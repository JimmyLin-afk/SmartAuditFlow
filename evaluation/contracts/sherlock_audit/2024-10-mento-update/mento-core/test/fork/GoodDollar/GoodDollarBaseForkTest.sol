// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

// Libraries / Helpers / Utils
import { console } from "forge-std/console.sol";
import { FixidityLib } from "celo/contracts/common/FixidityLib.sol";
import { TokenHelpers } from "../helpers/TokenHelpers.sol";
import { L0, L1, LG, min } from "../helpers/misc.sol";

// Interfaces
import { IBancorExchangeProvider } from "contracts/interfaces/IBancorExchangeProvider.sol";
import { IERC20 } from "contracts/interfaces/IERC20.sol";
import { IGoodDollar } from "contracts/goodDollar/interfaces/IGoodProtocol.sol";
import { IReserve } from "contracts/interfaces/IReserve.sol";
import { IStableTokenV2 } from "contracts/interfaces/IStableTokenV2.sol";
import { ITradingLimits } from "contracts/interfaces/ITradingLimits.sol";

// Contracts
import { BaseForkTest } from "../BaseForkTest.sol";
import { Broker } from "contracts/swap/Broker.sol";
import { GoodDollarExchangeProvider } from "contracts/goodDollar/GoodDollarExchangeProvider.sol";
import { GoodDollarExpansionController } from "contracts/goodDollar/GoodDollarExpansionController.sol";

contract GoodDollarBaseForkTest is BaseForkTest {
  using FixidityLib for FixidityLib.Fraction;
  using TokenHelpers for *;

  // Addresses
  address constant AVATAR_ADDRESS = 0x495d133B938596C9984d462F007B676bDc57eCEC;
  address constant CUSD_ADDRESS = 0x765DE816845861e75A25fCA122bb6898B8B1282a;
  address constant GOOD_DOLLAR_ADDRESS = 0x62B8B11039FcfE5aB0C56E502b1C372A3d2a9c7A;
  address constant REGISTRY_ADDRESS = 0x000000000000000000000000000000000000ce10;
  address ownerAddress;
  address distributionHelperAddress = makeAddr("distributionHelper");

  // GoodDollar Relaunch Config
  uint256 constant INITIAL_RESERVE_BALANCE = 200_000 * 1e18;
  uint256 constant INITIAL_GOOD_DOLLAR_TOKEN_SUPPLY = 7_000_000_000 * 1e18;
  uint32 constant INITIAL_RESERVE_RATIO = 0.28571428 * 1e8;
  uint32 constant INITIAL_EXIT_CONTRIBUTION = 0.1 * 1e8;
  uint64 constant INITIAL_EXPANSION_RATE = uint64(288617289022312); // == ~10% per year (assuming daily expansion)
  uint32 constant INITIAL_EXPANSION_FREQUENCY = uint32(1 days); // Daily expansion

  // Tokens
  IStableTokenV2 reserveToken;
  IGoodDollar goodDollarToken;

  // Contracts
  IReserve public goodDollarReserve;
  GoodDollarExchangeProvider goodDollarExchangeProvider;
  GoodDollarExpansionController expansionController;

  IBancorExchangeProvider.PoolExchange public poolExchange;
  bytes32 exchangeId;

  constructor(uint256 _chainId) BaseForkTest(_chainId) {}

  /* ======================================== */
  /* ================ Set Up ================ */
  /* ======================================== */

  function setUp() public virtual override {
    super.setUp();
    // Tokens
    reserveToken = IStableTokenV2(CUSD_ADDRESS);
    goodDollarToken = IGoodDollar(GOOD_DOLLAR_ADDRESS);

    // Contracts
    goodDollarExchangeProvider = new GoodDollarExchangeProvider(false);
    expansionController = new GoodDollarExpansionController(false);
    // deployCode() hack to deploy solidity v0.5 reserve contract from a v0.8 contract
    goodDollarReserve = IReserve(deployCode("Reserve", abi.encode(true)));

    // Addresses
    ownerAddress = makeAddr("owner");

    // Initialize GoodDollarExchangeProvider
    configureReserve();
    configureBroker();
    configureGoodDollarExchangeProvider();
    configureTokens();
    configureExpansionController();
    configureTradingLimits();
  }

  function configureReserve() public {
    bytes32[] memory initialAssetAllocationSymbols = new bytes32[](2);
    initialAssetAllocationSymbols[0] = bytes32("cGLD");
    initialAssetAllocationSymbols[1] = bytes32("cUSD");

    uint256[] memory initialAssetAllocationWeights = new uint256[](2);
    initialAssetAllocationWeights[0] = FixidityLib.newFixedFraction(1, 2).unwrap();
    initialAssetAllocationWeights[1] = FixidityLib.newFixedFraction(1, 2).unwrap();

    uint256 tobinTax = FixidityLib.newFixedFraction(5, 1000).unwrap();
    uint256 tobinTaxReserveRatio = FixidityLib.newFixedFraction(2, 1).unwrap();

    address[] memory collateralAssets = new address[](1);
    collateralAssets[0] = address(reserveToken);

    uint256[] memory collateralAssetDailySpendingRatios = new uint256[](1);
    collateralAssetDailySpendingRatios[0] = 1e24;

    vm.startPrank(ownerAddress);
    goodDollarReserve.initialize({
      registryAddress: REGISTRY_ADDRESS,
      _tobinTaxStalenessThreshold: 600, // deprecated
      _spendingRatioForCelo: 1000000000000000000000000,
      _frozenGold: 0,
      _frozenDays: 0,
      _assetAllocationSymbols: initialAssetAllocationSymbols,
      _assetAllocationWeights: initialAssetAllocationWeights,
      _tobinTax: tobinTax,
      _tobinTaxReserveRatio: tobinTaxReserveRatio,
      _collateralAssets: collateralAssets,
      _collateralAssetDailySpendingRatios: collateralAssetDailySpendingRatios
    });

    goodDollarReserve.addToken(address(goodDollarToken));
    goodDollarReserve.addExchangeSpender(address(broker));
    vm.stopPrank();
    require(
      goodDollarReserve.isStableAsset(address(goodDollarToken)),
      "GoodDollar is not a stable token in the reserve"
    );
    require(
      goodDollarReserve.isCollateralAsset(address(reserveToken)),
      "ReserveToken is not a collateral asset in the reserve"
    );
  }

  function configureTokens() public {
    vm.startPrank(AVATAR_ADDRESS);
    goodDollarToken.addMinter(address(broker));
    goodDollarToken.addMinter(address(expansionController));
    vm.stopPrank();

    deal({ token: address(reserveToken), to: address(goodDollarReserve), give: INITIAL_RESERVE_BALANCE });

    uint256 initialReserveGoodDollarBalanceInWei = (INITIAL_RESERVE_BALANCE /
      goodDollarExchangeProvider.currentPrice(exchangeId)) * 1e18;
    mintGoodDollar({ amount: initialReserveGoodDollarBalanceInWei, to: address(goodDollarReserve) });
  }

  function configureBroker() public {
    vm.prank(Broker(address(broker)).owner());
    broker.addExchangeProvider(address(goodDollarExchangeProvider), address(goodDollarReserve));

    require(
      broker.isExchangeProvider(address(goodDollarExchangeProvider)),
      "ExchangeProvider is not registered in the broker"
    );
    require(
      Broker(address(broker)).exchangeReserve(address(goodDollarExchangeProvider)) == address(goodDollarReserve),
      "Reserve is not registered in the broker"
    );
  }

  function configureGoodDollarExchangeProvider() public {
    vm.prank(ownerAddress);
    goodDollarExchangeProvider.initialize(
      address(broker),
      address(goodDollarReserve),
      address(expansionController),
      AVATAR_ADDRESS
    );

    poolExchange = IBancorExchangeProvider.PoolExchange({
      reserveAsset: address(reserveToken),
      tokenAddress: address(goodDollarToken),
      tokenSupply: INITIAL_GOOD_DOLLAR_TOKEN_SUPPLY,
      reserveBalance: INITIAL_RESERVE_BALANCE,
      reserveRatio: INITIAL_RESERVE_RATIO,
      exitContribution: INITIAL_EXIT_CONTRIBUTION
    });

    vm.prank(AVATAR_ADDRESS);
    exchangeId = IBancorExchangeProvider(address(goodDollarExchangeProvider)).createExchange(poolExchange);
  }

  function configureExpansionController() public {
    vm.prank(ownerAddress);
    expansionController.initialize({
      _goodDollarExchangeProvider: address(goodDollarExchangeProvider),
      _distributionHelper: distributionHelperAddress,
      _reserve: address(goodDollarReserve),
      _avatar: AVATAR_ADDRESS
    });

    vm.prank(AVATAR_ADDRESS);
    expansionController.setExpansionConfig({
      exchangeId: exchangeId,
      expansionRate: INITIAL_EXPANSION_RATE,
      expansionFrequency: INITIAL_EXPANSION_FREQUENCY
    });
  }

  function configureTradingLimits() internal {
    ITradingLimits.Config memory config = ITradingLimits.Config({
      // No more than 5,000 cUSD outflow within 5 minutes
      timestep0: 300,
      limit0: 5_000,
      // No more than 50,000 cUSD outflow within 1 day
      timestep1: 86_400,
      limit1: 50_000,
      // No more than 100,000 cUSD outflow in total
      limitGlobal: 100_000,
      flags: 1 | 2 | 4 // L0 = 1, L1 = 2, LG = 4
    });

    vm.prank(Broker(address(broker)).owner());
    broker.configureTradingLimit(exchangeId, address(reserveToken), config);
  }

  /**
   * @notice Manual deal helper because foundry's vm.deal() crashes
   * on the GoodDollar contract with "panic: assertion failed (0x01)"
   */
  function mintGoodDollar(uint256 amount, address to) public {
    vm.prank(AVATAR_ADDRESS);
    goodDollarToken.mint(to, amount);
  }

  function logHeader() internal view {
    string memory ticker = string(
      abi.encodePacked(IERC20(address(reserveToken)).symbol(), "/", IERC20(address(goodDollarToken)).symbol())
    );
    console.log("========================================");
    console.log(unicode"🔦 Testing pair:", ticker);
    console.log("========================================");
  }

  function getTradingLimitId(address tokenAddress) public view returns (bytes32 limitId) {
    bytes32 tokenInBytes32 = bytes32(uint256(uint160(tokenAddress)));
    bytes32 _limitId = exchangeId ^ tokenInBytes32;
    return _limitId;
  }

  function getTradingLimitsConfig(address tokenAddress) public view returns (ITradingLimits.Config memory config) {
    bytes32 limitId = getTradingLimitId(tokenAddress);
    ITradingLimits.Config memory _config;
    (_config.timestep0, _config.timestep1, _config.limit0, _config.limit1, _config.limitGlobal, _config.flags) = Broker(
      address(broker)
    ).tradingLimitsConfig(limitId);

    return _config;
  }

  function getTradingLimitsState(address tokenAddress) public view returns (ITradingLimits.State memory state) {
    bytes32 limitId = getTradingLimitId(tokenAddress);
    ITradingLimits.State memory _state;
    (_state.lastUpdated0, _state.lastUpdated1, _state.netflow0, _state.netflow1, _state.netflowGlobal) = Broker(
      address(broker)
    ).tradingLimitsState(limitId);

    return _state;
  }

  function getRefreshedTradingLimitsState(
    address tokenAddress
  ) public view returns (ITradingLimits.State memory state) {
    ITradingLimits.Config memory config = getTradingLimitsConfig(tokenAddress);
    // Netflow might be outdated because of a skip(...) call.
    // By doing an update(-1) and then update(1 ) we refresh the state without changing the state.
    // The reason we can't just update(0) is that 0 would be cast to -1 in the update function.
    state = tradingLimits.update(getTradingLimitsState(tokenAddress), config, -2, 1);
    state = tradingLimits.update(state, config, 1, 0);
  }

  function maxOutflow(address tokenAddress) internal view returns (int48) {
    ITradingLimits.Config memory config = getTradingLimitsConfig(tokenAddress);
    ITradingLimits.State memory state = getRefreshedTradingLimitsState(tokenAddress);
    int48 maxOutflowL0 = config.limit0 + state.netflow0;
    int48 maxOutflowL1 = config.limit1 + state.netflow1;
    int48 maxOutflowLG = config.limitGlobal + state.netflowGlobal;

    if (config.flags == L0 | L1 | LG) {
      return min(maxOutflowL0, maxOutflowL1, maxOutflowLG);
    } else if (config.flags == L0 | LG) {
      return min(maxOutflowL0, maxOutflowLG);
    } else if (config.flags == L0 | L1) {
      return min(maxOutflowL0, maxOutflowL1);
    } else if (config.flags == L0) {
      return maxOutflowL0;
    } else {
      revert("Unexpected limit config");
    }
  }

  function test_init_isDeployedAndInitializedCorrectly() public view {
    assertEq(goodDollarExchangeProvider.owner(), ownerAddress);
    assertEq(goodDollarExchangeProvider.broker(), address(broker));
    assertEq(address(goodDollarExchangeProvider.reserve()), address(goodDollarReserve));

    IBancorExchangeProvider.PoolExchange memory _poolExchange = goodDollarExchangeProvider.getPoolExchange(exchangeId);
    assertEq(_poolExchange.reserveAsset, _poolExchange.reserveAsset);
    assertEq(_poolExchange.tokenAddress, _poolExchange.tokenAddress);
    assertEq(_poolExchange.tokenSupply, _poolExchange.tokenSupply);
    assertEq(_poolExchange.reserveBalance, _poolExchange.reserveBalance);
    assertEq(_poolExchange.reserveRatio, _poolExchange.reserveRatio);
    assertEq(_poolExchange.exitContribution, _poolExchange.exitContribution);
  }
}
