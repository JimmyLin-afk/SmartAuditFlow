// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;
// solhint-disable func-name-mixedcase, var-name-mixedcase, state-visibility, max-line-length
// solhint-disable const-name-snakecase, max-states-count, contract-name-camelcase

import { Test } from "forge-std/Test.sol";
import { ERC20 } from "openzeppelin-contracts-next/contracts/token/ERC20/ERC20.sol";
import { BancorExchangeProvider } from "contracts/goodDollar/BancorExchangeProvider.sol";
import { IExchangeProvider } from "contracts/interfaces/IExchangeProvider.sol";
import { IBancorExchangeProvider } from "contracts/interfaces/IBancorExchangeProvider.sol";
import { IReserve } from "contracts/interfaces/IReserve.sol";

contract BancorExchangeProviderTest is Test {
  /* ------- Events from IBancorExchangeProvider ------- */

  event BrokerUpdated(address indexed newBroker);

  event ReserveUpdated(address indexed newReserve);

  event PowerUpdated(address indexed newPower);

  event ExchangeCreated(bytes32 indexed exchangeId, address indexed reserveAsset, address indexed tokenAddress);

  event ExchangeDestroyed(bytes32 indexed exchangeId, address indexed reserveAsset, address indexed tokenAddress);

  event ExitContributionSet(bytes32 indexed exchangeId, uint256 exitContribution);

  /* ------------------------------------------- */

  ERC20 public reserveToken;
  ERC20 public token;
  ERC20 public token2;

  address public reserveAddress;
  address public brokerAddress;
  IBancorExchangeProvider.PoolExchange public poolExchange1;
  IBancorExchangeProvider.PoolExchange public poolExchange2;

  function setUp() public virtual {
    reserveToken = new ERC20("cUSD", "cUSD");
    token = new ERC20("Good$", "G$");
    token2 = new ERC20("Good2$", "G2$");

    brokerAddress = makeAddr("Broker");
    reserveAddress = makeAddr("Reserve");

    poolExchange1 = IBancorExchangeProvider.PoolExchange({
      reserveAsset: address(reserveToken),
      tokenAddress: address(token),
      tokenSupply: 300_000 * 1e18,
      reserveBalance: 60_000 * 1e18,
      reserveRatio: 1e8 * 0.2,
      exitContribution: 1e8 * 0.01
    });

    poolExchange2 = IBancorExchangeProvider.PoolExchange({
      reserveAsset: address(reserveToken),
      tokenAddress: address(token2),
      tokenSupply: 300_000 * 1e18,
      reserveBalance: 60_000 * 1e18,
      reserveRatio: 1e8 * 0.2,
      exitContribution: 1e8 * 0.01
    });

    vm.mockCall(
      reserveAddress,
      abi.encodeWithSelector(IReserve(reserveAddress).isStableAsset.selector, address(token)),
      abi.encode(true)
    );
    vm.mockCall(
      reserveAddress,
      abi.encodeWithSelector(IReserve(reserveAddress).isStableAsset.selector, address(token2)),
      abi.encode(true)
    );
    vm.mockCall(
      reserveAddress,
      abi.encodeWithSelector(IReserve(reserveAddress).isCollateralAsset.selector, address(reserveToken)),
      abi.encode(true)
    );
  }

  function initializeBancorExchangeProvider() internal returns (BancorExchangeProvider) {
    BancorExchangeProvider bancorExchangeProvider = new BancorExchangeProvider(false);

    bancorExchangeProvider.initialize(brokerAddress, reserveAddress);
    return bancorExchangeProvider;
  }
}

contract BancorExchangeProviderTest_initilizerSettersGetters is BancorExchangeProviderTest {
  BancorExchangeProvider bancorExchangeProvider;

  function setUp() public override {
    super.setUp();
    bancorExchangeProvider = initializeBancorExchangeProvider();
  }

  /* ---------- Initializer ---------- */
  function test_initialize_shouldSetOwner() public view {
    assertEq(bancorExchangeProvider.owner(), address(this));
  }

  function test_initialize_shouldSetBroker() public view {
    assertEq(bancorExchangeProvider.broker(), brokerAddress);
  }

  function test_initialize_shouldSetReserve() public view {
    assertEq(address(bancorExchangeProvider.reserve()), reserveAddress);
  }

  /* ---------- Setters ---------- */
  function test_setBroker_whenSenderIsNotOwner_shouldRevert() public {
    vm.prank(makeAddr("NotOwner"));
    vm.expectRevert("Ownable: caller is not the owner");
    bancorExchangeProvider.setBroker(makeAddr("NewBroker"));
  }

  function test_setBroker_whenAddressIsZero_shouldRevert() public {
    vm.expectRevert("Broker address must be set");
    bancorExchangeProvider.setBroker(address(0));
  }

  function test_setBroker_whenSenderIsOwner_shouldUpdateAndEmit() public {
    address newBroker = makeAddr("NewBroker");

    vm.expectEmit(true, true, true, true);
    emit BrokerUpdated(newBroker);
    bancorExchangeProvider.setBroker(newBroker);

    assertEq(bancorExchangeProvider.broker(), newBroker);
  }

  function test_setReserve_whenSenderIsNotOwner_shouldRevert() public {
    vm.prank(makeAddr("NotOwner"));
    vm.expectRevert("Ownable: caller is not the owner");
    bancorExchangeProvider.setReserve(makeAddr("NewReserve"));
  }

  function test_setReserve_whenAddressIsZero_shouldRevert() public {
    vm.expectRevert("Reserve address must be set");
    bancorExchangeProvider.setReserve(address(0));
  }

  function test_setReserve_whenSenderIsOwner_shouldUpdateAndEmit() public {
    address newReserve = makeAddr("NewReserve");

    vm.expectEmit(true, true, true, true);
    emit ReserveUpdated(newReserve);
    bancorExchangeProvider.setReserve(newReserve);

    assertEq(address(bancorExchangeProvider.reserve()), newReserve);
  }

  function test_setExitContribution_whenSenderIsNotOwner_shouldRevert() public {
    vm.prank(makeAddr("NotOwner"));
    vm.expectRevert("Ownable: caller is not the owner");
    bytes32 exchangeId = "0xexchangeId";
    bancorExchangeProvider.setExitContribution(exchangeId, 1e5);
  }

  function test_setExitContribution_whenExchangeDoesNotExist_shouldRevert() public {
    bytes32 exchangeId = "0xexchangeId";
    vm.expectRevert("Exchange does not exist");
    bancorExchangeProvider.setExitContribution(exchangeId, 1e5);
  }

  function test_setExitContribution_whenExitContributionAbove100Percent_shouldRevert() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);

    uint32 maxWeight = bancorExchangeProvider.MAX_WEIGHT();
    vm.expectRevert("Exit contribution is too high");
    bancorExchangeProvider.setExitContribution(exchangeId, maxWeight + 1);
  }

  function test_setExitContribution_whenSenderIsOwner_shouldUpdateAndEmit() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);

    uint32 newExitContribution = 1e3;
    vm.expectEmit(true, true, true, true);
    emit ExitContributionSet(exchangeId, newExitContribution);
    bancorExchangeProvider.setExitContribution(exchangeId, newExitContribution);

    IBancorExchangeProvider.PoolExchange memory poolExchange = bancorExchangeProvider.getPoolExchange(exchangeId);
    assertEq(poolExchange.exitContribution, newExitContribution);
  }

  /* ---------- Getters ---------- */

  function test_getPoolExchange_whenExchangeDoesNotExist_shouldRevert() public {
    bytes32 exchangeId = "0xexchangeId";
    vm.expectRevert("Exchange does not exist");
    bancorExchangeProvider.getPoolExchange(exchangeId);
  }

  function test_getPoolExchange_whenPoolExists_shouldReturnPool() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);

    IBancorExchangeProvider.PoolExchange memory poolExchange = bancorExchangeProvider.getPoolExchange(exchangeId);
    assertEq(poolExchange.reserveAsset, poolExchange1.reserveAsset);
    assertEq(poolExchange.tokenAddress, poolExchange1.tokenAddress);
    assertEq(poolExchange.tokenSupply, poolExchange1.tokenSupply);
    assertEq(poolExchange.reserveBalance, poolExchange1.reserveBalance);
    assertEq(poolExchange.reserveRatio, poolExchange1.reserveRatio);
    assertEq(poolExchange.exitContribution, poolExchange1.exitContribution);
  }

  function test_getExchangeIds_whenNoExchanges_shouldReturnEmptyArray() public view {
    bytes32[] memory exchangeIds = bancorExchangeProvider.getExchangeIds();
    assertEq(exchangeIds.length, 0);
  }

  function test_getExchangeIds_whenExchangesExist_shouldReturnExchangeIds() public {
    bytes32 exchangeId1 = bancorExchangeProvider.createExchange(poolExchange1);

    bytes32[] memory exchangeIds = bancorExchangeProvider.getExchangeIds();
    assertEq(exchangeIds.length, 1);
    assertEq(exchangeIds[0], exchangeId1);
  }

  function test_getExchanges_whenNoExchanges_shouldReturnEmptyArray() public view {
    IExchangeProvider.Exchange[] memory exchanges = bancorExchangeProvider.getExchanges();
    assertEq(exchanges.length, 0);
  }

  function test_getExchanges_whenExchangesExist_shouldReturnExchange() public {
    bytes32 exchangeId1 = bancorExchangeProvider.createExchange(poolExchange1);

    IExchangeProvider.Exchange[] memory exchanges = bancorExchangeProvider.getExchanges();
    assertEq(exchanges.length, 1);
    assertEq(exchanges[0].exchangeId, exchangeId1);
    assertEq(exchanges[0].assets[0], poolExchange1.reserveAsset);
    assertEq(exchanges[0].assets[1], poolExchange1.tokenAddress);
  }
}

contract BancorExchangeProviderTest_createExchange is BancorExchangeProviderTest {
  BancorExchangeProvider bancorExchangeProvider;

  function setUp() public override {
    super.setUp();
    bancorExchangeProvider = initializeBancorExchangeProvider();
  }

  function test_createExchange_whenSenderIsNotOwner_shouldRevert() public {
    vm.prank(makeAddr("NotOwner"));
    vm.expectRevert("Ownable: caller is not the owner");
    bancorExchangeProvider.createExchange(poolExchange1);
  }

  function test_createExchange_whenReserveAssetIsZero_shouldRevert() public {
    poolExchange1.reserveAsset = address(0);
    vm.expectRevert("Invalid reserve asset");
    bancorExchangeProvider.createExchange(poolExchange1);
  }

  function test_createExchange_whenReserveAssetIsNotCollateral_shouldRevert() public {
    vm.mockCall(
      reserveAddress,
      abi.encodeWithSelector(IReserve(reserveAddress).isCollateralAsset.selector, address(reserveToken)),
      abi.encode(false)
    );
    vm.expectRevert("Reserve asset must be a collateral registered with the reserve");
    bancorExchangeProvider.createExchange(poolExchange1);
  }

  function test_createExchange_whenTokenAddressIsZero_shouldRevert() public {
    poolExchange1.tokenAddress = address(0);
    vm.expectRevert("Invalid token address");
    bancorExchangeProvider.createExchange(poolExchange1);
  }

  function test_createExchange_whenTokenAddressIsNotStable_shouldRevert() public {
    vm.mockCall(
      reserveAddress,
      abi.encodeWithSelector(IReserve(reserveAddress).isStableAsset.selector, address(token)),
      abi.encode(false)
    );
    vm.expectRevert("Token must be a stable registered with the reserve");
    bancorExchangeProvider.createExchange(poolExchange1);
  }

  function test_createExchange_whenReserveRatioIsSmaller2_shouldRevert() public {
    poolExchange1.reserveRatio = 0;
    vm.expectRevert("Reserve ratio is too low");
    bancorExchangeProvider.createExchange(poolExchange1);
    poolExchange1.reserveRatio = 1;
    vm.expectRevert("Reserve ratio is too low");
    bancorExchangeProvider.createExchange(poolExchange1);
  }

  function test_createExchange_whenReserveRatioAbove100Percent_shouldRevert() public {
    poolExchange1.reserveRatio = bancorExchangeProvider.MAX_WEIGHT() + 1;
    vm.expectRevert("Reserve ratio is too high");
    bancorExchangeProvider.createExchange(poolExchange1);
  }

  function test_createExchange_whenExitContributionAbove100Percent_shouldRevert() public {
    poolExchange1.exitContribution = bancorExchangeProvider.MAX_WEIGHT() + 1;
    vm.expectRevert("Exit contribution is too high");
    bancorExchangeProvider.createExchange(poolExchange1);
  }

  function test_createExchange_whenExchangeAlreadyExists_shouldRevert() public {
    bancorExchangeProvider.createExchange(poolExchange1);
    vm.expectRevert("Exchange already exists");
    bancorExchangeProvider.createExchange(poolExchange1);
  }

  function test_createExchanges_whenReserveTokenHasMoreDecimalsThan18_shouldRevert() public {
    vm.mockCall(address(reserveToken), abi.encodeWithSelector(reserveToken.decimals.selector), abi.encode(19));
    vm.expectRevert("Reserve asset decimals must be <= 18");
    bancorExchangeProvider.createExchange(poolExchange1);
  }

  function test_createExchange_whenTokenHasMoreDecimalsThan18_shouldRevert() public {
    vm.mockCall(address(token), abi.encodeWithSelector(token.decimals.selector), abi.encode(19));
    vm.expectRevert("Token decimals must be <= 18");
    bancorExchangeProvider.createExchange(poolExchange1);
  }

  function test_createExchange_whenExchangeDoesNotExist_shouldCreateExchangeAndEmit() public {
    vm.expectEmit(true, true, true, true);
    bytes32 expectedExchangeId = keccak256(abi.encodePacked(reserveToken.symbol(), token.symbol()));
    emit ExchangeCreated(expectedExchangeId, address(reserveToken), address(token));
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    assertEq(exchangeId, expectedExchangeId);

    IBancorExchangeProvider.PoolExchange memory poolExchange = bancorExchangeProvider.getPoolExchange(exchangeId);
    assertEq(poolExchange.reserveAsset, poolExchange1.reserveAsset);
    assertEq(poolExchange.tokenAddress, poolExchange1.tokenAddress);
    assertEq(poolExchange.tokenSupply, poolExchange1.tokenSupply);
    assertEq(poolExchange.reserveBalance, poolExchange1.reserveBalance);
    assertEq(poolExchange.reserveRatio, poolExchange1.reserveRatio);
    assertEq(poolExchange.exitContribution, poolExchange1.exitContribution);

    IExchangeProvider.Exchange[] memory exchanges = bancorExchangeProvider.getExchanges();
    assertEq(exchanges.length, 1);
    assertEq(exchanges[0].exchangeId, exchangeId);

    assertEq(bancorExchangeProvider.tokenPrecisionMultipliers(address(reserveToken)), 1);
    assertEq(bancorExchangeProvider.tokenPrecisionMultipliers(address(token)), 1);
  }
}

contract BancorExchangeProviderTest_destroyExchange is BancorExchangeProviderTest {
  BancorExchangeProvider bancorExchangeProvider;

  function setUp() public override {
    super.setUp();
    bancorExchangeProvider = initializeBancorExchangeProvider();
  }

  function test_destroyExchange_whenSenderIsNotOwner_shouldRevert() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    vm.prank(makeAddr("NotOwner"));
    vm.expectRevert("Ownable: caller is not the owner");
    bancorExchangeProvider.destroyExchange(exchangeId, 0);
  }

  function test_destroyExchange_whenIndexOutOfRange_shouldRevert() public {
    bytes32 exchangeId = "0xexchangeId";
    vm.expectRevert("exchangeIdIndex not in range");
    bancorExchangeProvider.destroyExchange(exchangeId, 10);
  }

  function test_destroyExchange_whenExchangeIdAndIndexDontMatch_shouldRevert() public {
    bytes32 exchangeId = "0xexchangeId";
    bancorExchangeProvider.createExchange(poolExchange1);
    vm.expectRevert("exchangeId at index doesn't match");
    bancorExchangeProvider.destroyExchange(exchangeId, 0);
  }

  function test_destroyExchange_whenExchangeExists_shouldDestroyExchangeAndEmit() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    bytes32 exchangeId2 = bancorExchangeProvider.createExchange(poolExchange2);

    vm.expectEmit(true, true, true, true);
    emit ExchangeDestroyed(exchangeId, poolExchange1.reserveAsset, poolExchange1.tokenAddress);
    bancorExchangeProvider.destroyExchange(exchangeId, 0);

    bytes32[] memory exchangeIds = bancorExchangeProvider.getExchangeIds();
    assertEq(exchangeIds.length, 1);

    IExchangeProvider.Exchange[] memory exchanges = bancorExchangeProvider.getExchanges();
    assertEq(exchanges.length, 1);
    assertEq(exchanges[0].exchangeId, exchangeId2);
  }
}

contract BancorExchangeProviderTest_getAmountIn is BancorExchangeProviderTest {
  BancorExchangeProvider bancorExchangeProvider;

  function setUp() public override {
    super.setUp();
    bancorExchangeProvider = initializeBancorExchangeProvider();
  }

  function test_getAmountIn_whenExchangeDoesNotExist_shouldRevert() public {
    bytes32 exchangeId = "0xexchangeId";
    vm.expectRevert("Exchange does not exist");
    bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountOut: 1e18
    });
  }

  function test_getAmountIn_whenTokenInNotInExchange_shouldRevert() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    vm.expectRevert("tokenIn and tokenOut must match exchange");
    bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(token2),
      tokenOut: address(token),
      amountOut: 1e18
    });
  }

  function test_getAmountIn_whenTokenOutNotInExchange_shouldRevert() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    vm.expectRevert("tokenIn and tokenOut must match exchange");
    bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(token2),
      amountOut: 1e18
    });
  }

  function test_getAmountIn_whenTokenInEqualsTokenOut_shouldRevert() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    vm.expectRevert("tokenIn and tokenOut must match exchange");
    bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(token),
      amountOut: 1e18
    });
  }

  function test_getAmountIn_whenTokenInIsTokenAndTokenSupplyIsZero_shouldRevert() public {
    poolExchange1.tokenSupply = 0;
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);

    vm.expectRevert("ERR_INVALID_SUPPLY");
    bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountOut: 1e18
    });
  }

  function test_getAmountIn_whenTokenInIsTokenAndReserveBalanceIsZero_shouldRevert() public {
    poolExchange1.reserveBalance = 0;
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);

    vm.expectRevert("ERR_INVALID_RESERVE_BALANCE");
    bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountOut: 1e18
    });
  }

  function test_getAmountIn_whenTokenInIsTokenAndAmountOutLargerThanReserveBalance_shouldRevert() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    vm.expectRevert("ERR_INVALID_AMOUNT");
    bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountOut: poolExchange1.reserveBalance + 1
    });
  }

  function test_getAmountIn_whenTokenInIsTokenAndAmountOutZero_shouldReturnZero() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    uint256 amountIn = bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountOut: 0
    });
    assertEq(amountIn, 0);
  }

  function test_getAmountIn_whenTokenInIsTokenAndAmountOutEqualReserveBalance_shouldReturnSupply() public {
    // need to set exit contribution to 0 to make the formula work otherwise amountOut would need to be adjusted
    // to be equal to reserveBalance after exit contribution is applied
    poolExchange1.exitContribution = 0;
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    uint256 expectedAmountIn = poolExchange1.tokenSupply;
    uint256 amountIn = bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountOut: poolExchange1.reserveBalance
    });
    assertEq(amountIn, expectedAmountIn);
  }

  function test_getAmountIn_whenTokenInIsTokenAndReserveRatioIs100Percent_shouldReturnCorrectAmount() public {
    poolExchange1.reserveRatio = 1e8;
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    uint256 amountOut = 12e18;
    // formula: amountIn = (amountOut / (1-e)) * tokenSupply / reserveBalance
    // calculation: (12 / 0.99) * 300_000 / 60_000 = 60.60606060606060606060606060606060606060
    uint256 expectedAmountIn = 60606060606060606060;

    uint256 amountIn = bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountOut: amountOut
    });

    assertEq(amountIn, expectedAmountIn);
  }

  function test_getAmountIn_whenTokenInIsReserveAssetAndSupplyIsZero_shouldRevert() public {
    poolExchange1.tokenSupply = 0;
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);

    vm.expectRevert("ERR_INVALID_SUPPLY");
    bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountOut: 1e18
    });
  }

  function test_getAmountIn_whenTokenInIsReserveAssetAndReserveBalanceIsZero_shouldRevert() public {
    poolExchange1.reserveBalance = 0;
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);

    vm.expectRevert("ERR_INVALID_RESERVE_BALANCE");
    bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountOut: 1e18
    });
  }

  function test_getAmountIn_whenTokenInIsReserveAssetAndAmountOutIsZero_shouldReturnZero() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    uint256 amountIn = bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountOut: 0
    });
    assertEq(amountIn, 0);
  }

  function test_getAmountIn_whenTokenInIsReserveAssetAndReserveRatioIs100Percent_shouldReturnCorrectAmount() public {
    poolExchange1.reserveRatio = 1e8;
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    uint256 amountOut = 12e18;
    // formula: amountIn = (amountOut * reserveBalance) / supply
    // calculation: (12 * 60_000) / 300_000 = 2.4
    uint256 expectedAmountIn = 1e18 * 2.4;

    uint256 amountIn = bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountOut: amountOut
    });

    assertEq(amountIn, expectedAmountIn);
  }

  function test_getAmountIn_whenTokenInIsReserveAsset_shouldReturnCorrectAmount() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    // formula: amountIn = reserveBalance * (( (tokenSupply + amountOut) / tokenSupply) ^ (1/reserveRatio) - 1)
    // calculation: 60_000 * ((300_001/300_000)^(1/0.2) - 1) ≈ 1.000006666688888926
    uint256 expectedAmountIn = 1000006666688888926;
    uint256 amountIn = bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountOut: 1e18
    });
    assertEq(amountIn, expectedAmountIn);
  }

  function test_getAmountIn_whenTokenInIsToken_shouldReturnCorrectAmount() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    // formula:          =   tokenSupply * (-1 + (reserveBalance / (reserveBalance - (amountOut/(1-e)))  )^reserveRatio )
    // formula: amountIn = ------------------------------------------------------------------------------------------------ this is a fractional line
    // formula:          =       (reserveBalance / (reserveBalance - (amountOut/(1-e)))  )^reserveRatio

    // calculation: (300000 * ( -1 + (60000 / (60000-(1/0.99)))^0.2))/(60000 / (60000-(1/0.99)))^0.2 = 1.010107812196722301
    uint256 expectedAmountIn = 1010107812196722302;
    uint256 amountIn = bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountOut: 1e18
    });
    assertEq(amountIn, expectedAmountIn);
  }

  function test_getAmountIn_whenTokenInIsReserveAssetAndAmountOutIsSmall_shouldReturnCorrectAmount() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    uint256 amountOut = 1e12; // 0.000001 token
    // formula: amountIn = reserveBalance * ((amountOut/tokenSupply + 1)^(1/reserveRatio) - 1)
    // calculation: 60_000 * ((0.000001/300_000 + 1)^(1/0.2) - 1) ≈ 0.00000100000000000666666
    uint256 expectedAmountIn = 1000000000007;
    uint256 amountIn = bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountOut: amountOut
    });
    assertEq(amountIn, expectedAmountIn);
  }

  function test_getAmountIn_whenTokenInIsTokenAndAmountOutIsSmall_shouldReturnCorrectAmount() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    uint256 amountOut = 1e12; // 0.000001 token
    // formula:          =   tokenSupply * (-1 + (reserveBalance / (reserveBalance - (amountOut/(1-e)))  )^reserveRatio )
    // formula: amountIn = ------------------------------------------------------------------------------------------------ this is a fractional line
    // formula:          =       (reserveBalance / (reserveBalance - (amountOut/(1-e)))  )^reserveRatio

    // calculation: (300000 * ( -1 + (60000 / (60000-(0.000001/0.99)))^0.2))/(60000 / (60000-(0.000001/0.99)))^0.2 ≈ 0.000001010101010107
    uint256 expectedAmountIn = 1010101010108;
    uint256 amountIn = bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountOut: amountOut
    });
    assertEq(amountIn, expectedAmountIn);
  }

  function test_getAmountIn_whenTokenInIsReserveAssetAndAmountOutIsLarge_shouldReturnCorrectAmount() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    uint256 amountOut = 1_000_000e18;
    // formula: amountIn = reserveBalance * ((amountOut/tokenSupply + 1)^(1/reserveRatio) - 1)
    // calculation: 60_000 * ((1_000_000/300_000 + 1)^(1/0.2) - 1) ≈ 91617283.9506172839506172839
    // 1 wei difference due to precision loss
    uint256 expectedAmountIn = 91617283950617283950617284;
    uint256 amountIn = bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountOut: amountOut
    });
    assertEq(amountIn, expectedAmountIn);
  }

  function test_getAmountIn_whenTokenInIsTokenAndAmountOutIsLarge_shouldReturnCorrectAmount() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    uint256 amountOut = 59000e18; // 59_000 since total reserve is 60k
    // formula:          =   tokenSupply * (-1 + (reserveBalance / (reserveBalance - (amountOut/(1-e)))  )^reserveRatio )
    // formula: amountIn = ------------------------------------------------------------------------------------------------ this is a fractional line
    // formula:          =       (reserveBalance / (reserveBalance - (amountOut/(1-e)))  )^reserveRatio

    // calculation: (300000 * ( -1 + (60000 / (60000-(59000/0.99)))^0.2))/(60000 / (60000-(59000/0.99)))^0.2 = 189649.078540006525698460
    uint256 expectedAmountIn = 189649078540006525698460;
    uint256 amountIn = bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountOut: amountOut
    });
    // we allow up to 1% difference due to precision loss
    assertApproxEqRel(amountIn, expectedAmountIn, 1e18 * 0.01);
  }

  function test_getAmountIn_whenTokenInIsTokenAndExitContributionIsNonZero_shouldReturnCorrectAmount() public {
    // Set exit contribution to 1% (1e6 out of 1e8) for exchange 1 and 0 for exchange 2
    // all other parameters are the same
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    bancorExchangeProvider.setExitContribution(exchangeId, 1e6);
    bytes32 exchangeId2 = bancorExchangeProvider.createExchange(poolExchange2);
    bancorExchangeProvider.setExitContribution(exchangeId2, 0);

    uint256 amountOut = 116e18;
    // formula: amountIn = (tokenSupply * (( (amountOut + reserveBalance)  / reserveBalance) ^ (reserveRatio) - 1)) / exitContribution
    uint256 amountIn = bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountOut: amountOut
    });

    // exit contribution is 1%
    uint256 amountOut2 = (amountOut * 100) / 99;
    assertTrue(amountOut < amountOut2);

    uint256 amountIn2 = bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId2,
      tokenIn: address(token2),
      tokenOut: address(reserveToken),
      amountOut: amountOut2
    });
    assertEq(amountIn, amountIn2);
  }

  function test_getAmountIn_whenDifferentTokenDecimals_shouldReturnCorrectAmount() public {
    // Create new tokens with different decimals
    ERC20 reserveToken6 = new ERC20("Reserve6", "RSV6");
    ERC20 stableToken18 = new ERC20("Stable18", "STB18");

    vm.mockCall(
      reserveAddress,
      abi.encodeWithSelector(IReserve(reserveAddress).isStableAsset.selector, address(stableToken18)),
      abi.encode(true)
    );
    vm.mockCall(
      reserveAddress,
      abi.encodeWithSelector(IReserve(reserveAddress).isCollateralAsset.selector, address(reserveToken6)),
      abi.encode(true)
    );

    // Mock decimals for these tokens
    vm.mockCall(address(reserveToken6), abi.encodeWithSelector(reserveToken6.decimals.selector), abi.encode(6));
    vm.mockCall(address(stableToken18), abi.encodeWithSelector(stableToken18.decimals.selector), abi.encode(18));

    IBancorExchangeProvider.PoolExchange memory newPoolExchange = IBancorExchangeProvider.PoolExchange({
      reserveAsset: address(reserveToken6),
      tokenAddress: address(stableToken18),
      tokenSupply: 100_000 * 1e18, // 100,000
      reserveBalance: 50_000 * 1e18, // 50,000
      reserveRatio: 1e8 * 0.5, // 50%
      exitContribution: 0
    });

    bytes32 newExchangeId = bancorExchangeProvider.createExchange(newPoolExchange);

    uint256 amountOut = 1e18; // 1 StableToken out

    // Formula: reserveBalance * ((amountOut/tokenSupply + 1) ^ (1/reserveRatio) - 1)
    // calculation: 50_000 * ((1/100_000 + 1) ^ (1/0.5) - 1) = 1.000005 in 6 decimals = 1000005
    uint256 expectedAmountIn = 1000005;

    uint256 amountIn = bancorExchangeProvider.getAmountIn({
      exchangeId: newExchangeId,
      tokenIn: address(reserveToken6),
      tokenOut: address(stableToken18),
      amountOut: amountOut
    });
    assertEq(amountIn, expectedAmountIn);
    // 100_000 * ((1 + 1.000005/50000)^0.5 - 1) = 1.000005 in 18 decimals = 1000005000000000000
    uint256 reversedAmountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: newExchangeId,
      tokenIn: address(reserveToken6),
      tokenOut: address(stableToken18),
      amountIn: amountIn
    });
    // we allow a 10 wei difference due to rounding errors
    assertApproxEqAbs(amountOut, reversedAmountOut, 10);
  }

  function test_getAmountIn_whenTokenInIsReserveAsset_fuzz(uint256 amountOut) public {
    // these values are closed to the ones in the real exchange will be initialized with
    IBancorExchangeProvider.PoolExchange memory poolExchange = IBancorExchangeProvider.PoolExchange({
      reserveAsset: address(reserveToken),
      tokenAddress: address(token),
      tokenSupply: 7_000_000_000 * 1e18,
      reserveBalance: 200_000 * 1e18,
      reserveRatio: uint32(28571428),
      exitContribution: 1e7
    });

    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange);

    // amountOut range between 1 and 10_000_000 tokens
    amountOut = bound(amountOut, 1e18, 10_000_000 * 1e18);

    uint256 amountIn = bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountOut: amountOut
    });

    // Basic sanity checks
    assertTrue(amountIn > 0, "Amount in should be positive");

    // Verify the reverse swap
    uint256 reversedAmountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountIn: amountIn
    });

    // we allow up to 0.01% difference due to precision loss
    assertApproxEqRel(reversedAmountOut, amountOut, 1e18 * 0.0001);
  }

  function test_getAmountIn_whenTokenInIsToken_fuzz(uint256 amountOut) public {
    // these values are closed to the ones in the real exchange will be initialized with
    IBancorExchangeProvider.PoolExchange memory poolExchange = IBancorExchangeProvider.PoolExchange({
      reserveAsset: address(reserveToken),
      tokenAddress: address(token),
      tokenSupply: 7_000_000_000 * 1e18,
      reserveBalance: 200_000 * 1e18,
      reserveRatio: uint32(28571428),
      exitContribution: 1e7
    });

    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange);

    // reserve balance is 200_000 and you can't get more than 90% of it because of the exit contribution
    amountOut = bound(amountOut, 1e18, (200_000 * 1e18 * 90) / 100);

    uint256 amountIn = bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountOut: amountOut
    });

    // Basic sanity checks
    assertTrue(0 < amountIn, "Amount in should be positive");

    uint256 reversedAmountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountIn: amountIn
    });

    // we allow up to 0.01% difference due to precision loss
    assertApproxEqRel(reversedAmountOut, amountOut, 1e18 * 0.0001);
  }

  function test_getAmountIn_whenTokenInIsToken_fullFuzz(
    uint256 amountOut,
    uint256 reserveBalance,
    uint256 tokenSupply,
    uint256 reserveRatio,
    uint256 exitContribution
  ) public {
    // reserveBalance range between 100 tokens and 10_000_000 tokens
    reserveBalance = bound(reserveBalance, 100e18, 100_000_000 * 1e18);
    // tokenSupply range between 100 tokens and 100_000_000 tokens
    tokenSupply = bound(tokenSupply, 100e18, 100_000_000 * 1e18);
    // reserveRatio range between 1% and 100%
    reserveRatio = bound(reserveRatio, 1e6, 1e8);
    // exitContribution range between 0% and 20%
    exitContribution = bound(exitContribution, 0, 2e7);

    IBancorExchangeProvider.PoolExchange memory poolExchange = IBancorExchangeProvider.PoolExchange({
      reserveAsset: address(reserveToken),
      tokenAddress: address(token),
      tokenSupply: tokenSupply,
      reserveBalance: reserveBalance,
      reserveRatio: uint32(reserveRatio),
      exitContribution: uint32(exitContribution)
    });

    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange);

    // amountOut range between 0.0001 tokens and 70% of reserveBalance
    amountOut = bound(amountOut, 0.0001e18, (reserveBalance * 7) / 10);

    uint256 amountIn = bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountOut: amountOut
    });

    // Basic sanity checks
    assertTrue(amountIn > 0, "Amount in should be positive");

    uint256 reversedAmountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountIn: amountIn
    });

    // we allow up to 0.01% difference due to precision loss
    assertApproxEqRel(reversedAmountOut, amountOut, 1e18 * 0.0001);
  }

  function test_getAmountIn_whenTokenInIsReserveAsset_fullFuzz(
    uint256 amountOut,
    uint256 reserveBalance,
    uint256 tokenSupply,
    uint256 reserveRatio
  ) public {
    // tokenSupply range between 100 tokens and 10_000_000 tokens
    tokenSupply = bound(tokenSupply, 100e18, 10_000_000 * 1e18);
    // reserveBalance range between 100 tokens and 10_000_000 tokens
    reserveBalance = bound(reserveBalance, 100e18, 10_000_000 * 1e18);
    // reserveRatio range between 5% and 100%
    reserveRatio = bound(reserveRatio, 5e6, 1e8);

    IBancorExchangeProvider.PoolExchange memory poolExchange = IBancorExchangeProvider.PoolExchange({
      reserveAsset: address(reserveToken),
      tokenAddress: address(token),
      tokenSupply: tokenSupply,
      reserveBalance: reserveBalance,
      reserveRatio: uint32(reserveRatio),
      exitContribution: 0 // no exit contribution because reserveToken in
    });

    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange);

    // amountOut range between 0.0001 tokens and 3 times the current tokenSupply
    amountOut = bound(amountOut, 0.0001e18, tokenSupply * 3);

    uint256 amountIn = bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountOut: amountOut
    });

    // Basic sanity checks
    assertTrue(amountIn > 0, "Amount in should be positive");

    uint256 reversedAmountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountIn: amountIn
    });

    // we allow up to 1% difference due to precision loss
    assertApproxEqRel(reversedAmountOut, amountOut, 1e18 * 0.01);
  }
}

contract BancorExchangeProviderTest_getAmountOut is BancorExchangeProviderTest {
  BancorExchangeProvider bancorExchangeProvider;

  function setUp() public override {
    super.setUp();
    bancorExchangeProvider = initializeBancorExchangeProvider();
  }

  function test_getAmountOut_whenExchangeDoesNotExist_shouldRevert() public {
    bytes32 exchangeId = "0xexchangeId";
    vm.expectRevert("Exchange does not exist");
    bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountIn: 1e18
    });
  }

  function test_getAmountOut_whenTokenInNotInExchange_shouldRevert() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    vm.expectRevert("tokenIn and tokenOut must match exchange");
    bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(token2),
      tokenOut: address(token),
      amountIn: 1e18
    });
  }

  function test_getAmountOut_whenTokenOutNotInExchange_shouldRevert() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    vm.expectRevert("tokenIn and tokenOut must match exchange");
    bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(token2),
      amountIn: 1e18
    });
  }

  function test_getAmountOut_whenTokenInEqualTokenOut_shouldRevert() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    vm.expectRevert("tokenIn and tokenOut must match exchange");
    bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(token),
      amountIn: 1e18
    });
  }

  function test_getAmountOut_whenTokenInIsReserveAssetAndTokenSupplyIsZero_shouldRevert() public {
    poolExchange1.tokenSupply = 0;
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    vm.expectRevert("ERR_INVALID_SUPPLY");
    bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountIn: 1e18
    });
  }

  function test_getAmountOut_whenTokenInIsReserveAssetAndReserveBalanceIsZero_shouldRevert() public {
    poolExchange1.reserveBalance = 0;
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    vm.expectRevert("ERR_INVALID_RESERVE_BALANCE");
    bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountIn: 1e18
    });
  }

  function test_getAmountOut_whenTokenInIsReserveAssetAndAmountInIsZero_shouldReturnZero() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    uint256 amountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountIn: 0
    });
    assertEq(amountOut, 0);
  }

  function test_getAmountOut_whenTokenInIsReserveAssetAndReserveRatioIs100Percent_shouldReturnCorrectAmount() public {
    poolExchange1.reserveRatio = 1e8;
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    uint256 amountIn = 1e18;
    // formula: amountOut = tokenSupply * amountIn / reserveBalance
    // calculation: 300_000 * 1 / 60_000 = 5
    uint256 expectedAmountOut = 5e18;
    uint256 amountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountIn: amountIn
    });
    assertEq(amountOut, expectedAmountOut);
  }

  function test_getAmountOut_whenTokenInIsTokenAndSupplyIsZero_shouldRevert() public {
    poolExchange1.tokenSupply = 0;
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    vm.expectRevert("ERR_INVALID_SUPPLY");
    bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountIn: 1e18
    });
  }

  function test_getAmountOut_whenTokenInIsTokenAndReserveBalanceIsZero_shouldRevert() public {
    poolExchange1.reserveBalance = 0;
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    vm.expectRevert("ERR_INVALID_RESERVE_BALANCE");
    bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountIn: 1e18
    });
  }

  function test_getAmountOut_whenTokenInIsTokenAndAmountLargerSupply_shouldRevert() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    vm.expectRevert("ERR_INVALID_AMOUNT");
    bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountIn: poolExchange1.tokenSupply + 1
    });
  }

  function test_getAmountOut_whenTokenInIsTokenAndAmountIsZero_shouldReturnZero() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    uint256 amountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountIn: 0
    });
    assertEq(amountOut, 0);
  }

  function test_getAmountOut_whenTokenInIsTokenAndAmountIsSupply_shouldReturnReserveBalanceMinusExitContribution()
    public
  {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    uint256 amountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountIn: poolExchange1.tokenSupply
    });
    assertEq(amountOut, (poolExchange1.reserveBalance * (1e8 - poolExchange1.exitContribution)) / 1e8);
  }

  function test_getAmountOut_whenTokenInIsTokenAndReserveRatioIs100Percent_shouldReturnCorrectAmount() public {
    poolExchange1.reserveRatio = 1e8;
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    uint256 amountIn = 1e18;
    // formula: amountOut = (reserveBalance * amountIn / tokenSupply) * (1-e)
    // calculation: (60_000 * 1 / 300_000) * 0.99 = 0.198
    uint256 expectedAmountOut = 198000000000000000;
    uint256 amountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountIn: amountIn
    });
    assertEq(amountOut, expectedAmountOut);
  }

  function test_getAmountOut_whenTokenInIsReserveAsset_shouldReturnCorrectAmount() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    // formula: amountOut = tokenSupply * ((1 + amountIn / reserveBalance) ^ reserveRatio - 1)
    // calculation: 300_000 * ((1 + 1 / 60_000) ^ 0.2 - 1) ≈ 0.999993333399999222
    uint256 expectedAmountOut = 999993333399999222;
    uint256 amountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountIn: 1e18
    });
    assertEq(amountOut, expectedAmountOut);
  }

  function test_getAmountOut_whenTokenInIsToken_shouldReturnCorrectAmount() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    // formula:           = reserveBalance * ( -1 + (tokenSupply/(tokenSupply - amountIn ))^(1/reserveRatio))
    // formula: amountOut = ---------------------------------------------------------------------------------- * (1 - e)
    // formula:           =          (tokenSupply/(tokenSupply - amountIn ))^(1/reserveRatio)

    // calculation: ((60_000 *(-1+(300_000/(300_000-1))^5) ) / (300_000/(300_000-1))^5)*0.99 = 0.989993400021999963
    // 1 wei difference due to precision loss
    uint256 expectedAmountOut = 989993400021999962;
    uint256 amountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountIn: 1e18
    });
    assertEq(amountOut, expectedAmountOut);
  }

  function test_getAmountOut_whenTokenInIsReserveAssetAndAmountOutIsSmall_shouldReturnCorrectAmount() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    uint256 amountIn = 1e12; // 0.000001 reserve token
    // formula: amountOut = tokenSupply * ((1 + amountIn / reserveBalance) ^ reserveRatio - 1)
    // calculation: 300_000 * ((1 + 0.000001 / 60_000) ^ 0.2 - 1) ≈ 0.00000099999999999333
    uint256 expectedAmountOut = 999999999993;
    uint256 amountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountIn: amountIn
    });
    assertEq(amountOut, expectedAmountOut);
  }

  function test_getAmountOut_whenTokenInIsTokenAndAmountOutIsSmall_shouldReturnCorrectAmount() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    uint256 amountIn = 1e12; // 0.000001 token
    // formula:           = reserveBalance * ( -1 + (tokenSupply/(tokenSupply - amountIn ))^(1/reserveRatio))
    // formula: amountOut = ---------------------------------------------------------------------------------- * (1 - e)
    // formula:           =          (tokenSupply/(tokenSupply - amountIn ))^(1/reserveRatio)

    // calculation: ((60_000 *(-1+(300_000/(300_000-0.000001))^5) )/(300_000/(300_000-0.000001))^5)*0.99 ≈ 0.0000009899999999934
    uint256 expectedAmountOut = 989999999993;
    uint256 amountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountIn: amountIn
    });
    assertEq(amountOut, expectedAmountOut);
  }

  function test_getAmountOut_whenTokenInIsReserveAssetAndAmountInIsLarge_shouldReturnCorrectAmount() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    uint256 amountIn = 1_000_000e18;
    // formula: amountOut = tokenSupply * ((1 + amountIn / reserveBalance) ^ reserveRatio - 1)
    // calculation: 300_000 * ((1 + 1_000_000 / 60_000) ^ 0.2 - 1) ≈ 232785.231205449318288038
    uint256 expectedAmountOut = 232785231205449318288038;
    uint256 amountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountIn: amountIn
    });
    assertEq(amountOut, expectedAmountOut);
  }

  function test_getAmountOut_whenTokenInIsTokenAndAmountInIsLarge_shouldReturnCorrectAmount() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    uint256 amountIn = 299_000 * 1e18; // 299,000 tokens only 300k supply
    // formula:           = reserveBalance * ( -1 + (tokenSupply/(tokenSupply - amountIn ))^(1/reserveRatio))
    // formula: amountOut = ---------------------------------------------------------------------------------- * (1 - e)
    // formula:           =          (tokenSupply/(tokenSupply - amountIn ))^(1/reserveRatio)

    // calculation: ((60_000 *(-1+(300_000/(300_000-299_000))^5) ) / (300_000/(300_000-299_000))^5)*0.99 ≈ 59399.999999975555555555
    uint256 expectedAmountOut = 59399999999975555555555;
    uint256 amountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountIn: amountIn
    });

    // we allow up to 1% difference due to precision loss
    assertApproxEqRel(amountOut, expectedAmountOut, 1e18 * 0.01);
  }

  function test_getAmountOut_whenTokenInIsTokenAndExitContributionIsNonZero_shouldReturnCorrectAmount(
    uint256 amountIn
  ) public {
    // Set exit contribution to 1% (1e6 out of 1e8) for exchange 1 and 0 for exchange 2
    // all other parameters are the same
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    bancorExchangeProvider.setExitContribution(exchangeId, 1e6);
    bytes32 exchangeId2 = bancorExchangeProvider.createExchange(poolExchange2);
    bancorExchangeProvider.setExitContribution(exchangeId2, 0);

    amountIn = bound(amountIn, 100, 299_000 * 1e18);
    uint256 amountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountIn: amountIn
    });
    uint256 amountOut2 = bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId2,
      tokenIn: address(token2),
      tokenOut: address(reserveToken),
      amountIn: amountIn
    });
    assertEq(amountOut, (amountOut2 * 99) / 100);
  }

  function test_getAmountOut_whenDifferentTokenDecimals_shouldReturnCorrectAmount() public {
    // Create new tokens with different decimals
    ERC20 reserveToken6 = new ERC20("Reserve6", "RSV6");
    ERC20 stableToken18 = new ERC20("Stable18", "STB18");

    vm.mockCall(
      reserveAddress,
      abi.encodeWithSelector(IReserve(reserveAddress).isStableAsset.selector, address(stableToken18)),
      abi.encode(true)
    );
    vm.mockCall(
      reserveAddress,
      abi.encodeWithSelector(IReserve(reserveAddress).isCollateralAsset.selector, address(reserveToken6)),
      abi.encode(true)
    );

    // Mock decimals for these tokens
    vm.mockCall(address(reserveToken6), abi.encodeWithSelector(reserveToken6.decimals.selector), abi.encode(6));
    vm.mockCall(address(stableToken18), abi.encodeWithSelector(stableToken18.decimals.selector), abi.encode(18));

    IBancorExchangeProvider.PoolExchange memory newPoolExchange = IBancorExchangeProvider.PoolExchange({
      reserveAsset: address(reserveToken6),
      tokenAddress: address(stableToken18),
      tokenSupply: 100_000 * 1e18, // 100,000
      reserveBalance: 50_000 * 1e18, // 50,000
      reserveRatio: 1e8 * 0.5, // 50%
      exitContribution: 0
    });

    bytes32 newExchangeId = bancorExchangeProvider.createExchange(newPoolExchange);

    uint256 amountIn = 1000000; // 1 ReserveToken in 6 decimals

    // formula: amountOut = tokenSupply * (-1 + (1 + (amountIn/reserveBalance))^reserveRatio)
    // calculation: 100_000 * (-1 + (1+ (1 / 50_000))^0.5) ≈ 0.999995000049999375 in 18 decimals
    uint256 expectedAmountOut = 999995000049999375;

    uint256 amountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: newExchangeId,
      tokenIn: address(reserveToken6),
      tokenOut: address(stableToken18),
      amountIn: amountIn
    });
    assertEq(amountOut, expectedAmountOut);

    uint256 reversedAmountIn = bancorExchangeProvider.getAmountIn({
      exchangeId: newExchangeId,
      tokenIn: address(reserveToken6),
      tokenOut: address(stableToken18),
      amountOut: amountOut
    });
    // we allow a 1 wei difference due to precision loss
    assertApproxEqAbs(amountIn, reversedAmountIn, 1);
  }

  function test_getAmountOut_whenTokenInIsReserveAsset_fuzz(uint256 amountIn) public {
    // these values are closed to the ones in the real exchange will be initialized with
    IBancorExchangeProvider.PoolExchange memory poolExchange = IBancorExchangeProvider.PoolExchange({
      reserveAsset: address(reserveToken),
      tokenAddress: address(token),
      tokenSupply: 7_000_000_000 * 1e18,
      reserveBalance: 200_000 * 1e18,
      reserveRatio: uint32(28571428),
      exitContribution: 1e7
    });

    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange);

    // amountIn range between 1 and 10_000_000 tokens
    amountIn = bound(amountIn, 1e18, 10_000_000 * 1e18);

    uint256 amountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountIn: amountIn
    });

    // Basic sanity checks
    assertTrue(0 < amountOut, "Amount out should be positive");

    // Verify the reverse swap
    uint256 reversedAmountIn = bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountOut: amountOut
    });
    // we allow up to 10 wei due to precision loss
    assertApproxEqAbs(reversedAmountIn, amountIn, 10, "Reversed swap should approximately equal original amount in");
  }

  function test_getAmountOut_whenTokenInIsToken_fuzz(uint256 amountIn) public {
    // these values are closed to the ones in the real exchange will be initialized with
    IBancorExchangeProvider.PoolExchange memory poolExchange = IBancorExchangeProvider.PoolExchange({
      reserveAsset: address(reserveToken),
      tokenAddress: address(token),
      tokenSupply: 7_000_000_000 * 1e18,
      reserveBalance: 200_000 * 1e18,
      reserveRatio: uint32(28571428),
      exitContribution: 1e7
    });

    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange);

    // amountIn range between 10_000wei and 3_500_000_000tokens
    amountIn = bound(amountIn, 1e18, (poolExchange.tokenSupply * 5) / 10);

    uint256 amountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountIn: amountIn
    });

    // Basic sanity checks
    assertTrue(amountOut > 0, "Amount out should be positive");

    uint256 reversedAmountIn = bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountOut: amountOut
    });

    // we allow up to 0.1% difference due to precision loss
    assertApproxEqRel(reversedAmountIn, amountIn, 1e18 * 0.001);
  }

  function test_getAmountOut_whenTokenInIsReserveAsset_fullFuzz(
    uint256 amountIn,
    uint256 reserveBalance,
    uint256 tokenSupply,
    uint256 reserveRatio
  ) public {
    // tokenSupply range between 100 tokens and 10_000_000 tokens
    tokenSupply = bound(tokenSupply, 100e18, 10_000_000 * 1e18);
    // reserveBalance range between 100 tokens and 10_000_000 tokens
    reserveBalance = bound(reserveBalance, 100e18, 10_000_000 * 1e18);
    // reserveRatio range between 1% and 100%
    reserveRatio = bound(reserveRatio, 1e6, 1e8);

    IBancorExchangeProvider.PoolExchange memory poolExchange = IBancorExchangeProvider.PoolExchange({
      reserveAsset: address(reserveToken),
      tokenAddress: address(token),
      tokenSupply: tokenSupply,
      reserveBalance: reserveBalance,
      reserveRatio: uint32(reserveRatio),
      exitContribution: 0 // no exit contribution because reserveToken in
    });

    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange);

    // amountIn range between 0.0001 tokens and 1_000_000 tokens
    amountIn = bound(amountIn, 0.0001e18, 1_000_000 * 1e18);

    uint256 amountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountIn: amountIn
    });

    // Basic sanity checks
    assertTrue(amountOut > 0, "Amount out should be positive");

    uint256 reversedAmountIn = bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountOut: amountOut
    });

    // we allow up to 0.01% difference due to precision loss
    assertApproxEqRel(reversedAmountIn, amountIn, 1e18 * 0.0001);
  }

  function test_getAmountOut_whenTokenInIsToken_fullFuzz(
    uint256 amountIn,
    uint256 reserveBalance,
    uint256 tokenSupply,
    uint256 reserveRatio,
    uint256 exitContribution
  ) public {
    // reserveBalance range between 100 tokens and 10_000_000 tokens
    reserveBalance = bound(reserveBalance, 100e18, 10_000_000 * 1e18);
    // tokenSupply range between 100 tokens and 100_000_000 tokens
    tokenSupply = bound(tokenSupply, 100e18, 10_000_000 * 1e18);
    // reserveRatio range between 5% and 100%
    reserveRatio = bound(reserveRatio, 5e6, 1e8);
    // exitContribution range between 0% and 20%
    exitContribution = bound(exitContribution, 0, 2e7);

    IBancorExchangeProvider.PoolExchange memory poolExchange = IBancorExchangeProvider.PoolExchange({
      reserveAsset: address(reserveToken),
      tokenAddress: address(token),
      tokenSupply: tokenSupply,
      reserveBalance: reserveBalance,
      reserveRatio: uint32(reserveRatio),
      exitContribution: uint32(exitContribution)
    });

    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange);

    // amountIn range between 0.0001 tokens and 80% of tokenSupply
    // if we would allow 100% of the tokenSupply, the precision loss can get higher
    amountIn = bound(amountIn, 0.0001e18, (tokenSupply * 8) / 10);

    uint256 amountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountIn: amountIn
    });
    // Basic sanity checks
    assertTrue(amountIn > 0, "Amount in should be positive");

    uint256 reversedAmountIn = bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountOut: amountOut
    });

    // we allow up to 1% difference due to precision loss
    assertApproxEqRel(reversedAmountIn, amountIn, 1e18 * 0.01);
  }
}

contract BancorExchangeProviderTest_currentPrice is BancorExchangeProviderTest {
  BancorExchangeProvider bancorExchangeProvider;

  function setUp() public override {
    super.setUp();
    bancorExchangeProvider = initializeBancorExchangeProvider();
  }

  function test_currentPrice_whenExchangeDoesNotExist_shouldRevert() public {
    bytes32 exchangeId = "0xexchangeId";
    vm.expectRevert("Exchange does not exist");
    bancorExchangeProvider.currentPrice(exchangeId);
  }

  function test_currentPrice_whenExchangeExists_shouldReturnCorrectPrice() public {
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    // formula: price = reserveBalance / tokenSupply * reserveRatio
    // calculation: 60_000 / 300_000 * 0.2 = 1
    uint256 expectedPrice = 1e18;
    uint256 price = bancorExchangeProvider.currentPrice(exchangeId);
    assertEq(price, expectedPrice);
  }

  function test_currentPrice_fuzz(uint256 reserveBalance, uint256 tokenSupply, uint256 reserveRatio) public {
    // reserveBalance range between 1 token and 10_000_000 tokens
    reserveBalance = bound(reserveBalance, 1e18, 10_000_000 * 1e18);
    // tokenSupply range between 1 token and 10_000_000 tokens
    tokenSupply = bound(tokenSupply, 1e18, 10_000_000 * 1e18);
    // reserveRatio range between 1% and 100%
    reserveRatio = bound(reserveRatio, 1e6, 1e8);

    IBancorExchangeProvider.PoolExchange memory poolExchange = IBancorExchangeProvider.PoolExchange({
      reserveAsset: address(reserveToken),
      tokenAddress: address(token),
      tokenSupply: tokenSupply,
      reserveBalance: reserveBalance,
      reserveRatio: uint32(reserveRatio),
      exitContribution: 0
    });

    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange);

    uint256 price = bancorExchangeProvider.currentPrice(exchangeId);
    assertTrue(0 < price, "Price should be positive");
  }
}

contract BancorExchangeProviderTest_swapIn is BancorExchangeProviderTest {
  function test_swapIn_whenCallerIsNotBroker_shouldRevert() public {
    BancorExchangeProvider bancorExchangeProvider = initializeBancorExchangeProvider();
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    vm.prank(makeAddr("NotBroker"));
    vm.expectRevert("Caller is not the Broker");
    bancorExchangeProvider.swapIn(exchangeId, address(reserveToken), address(token), 1e18);
  }

  function test_swapIn_whenExchangeDoesNotExist_shouldRevert() public {
    BancorExchangeProvider bancorExchangeProvider = initializeBancorExchangeProvider();
    vm.prank(brokerAddress);
    vm.expectRevert("Exchange does not exist");
    bancorExchangeProvider.swapIn("0xexchangeId", address(reserveToken), address(token), 1e18);
  }

  function test_swapIn_whenTokenInNotInexchange_shouldRevert() public {
    BancorExchangeProvider bancorExchangeProvider = initializeBancorExchangeProvider();
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    vm.prank(brokerAddress);
    vm.expectRevert("tokenIn and tokenOut must match exchange");
    bancorExchangeProvider.swapIn(exchangeId, address(token2), address(token), 1e18);
  }

  function test_swapIn_whenTokenOutNotInexchange_shouldRevert() public {
    BancorExchangeProvider bancorExchangeProvider = initializeBancorExchangeProvider();
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    vm.prank(brokerAddress);
    vm.expectRevert("tokenIn and tokenOut must match exchange");
    bancorExchangeProvider.swapIn(exchangeId, address(token), address(token2), 1e18);
  }

  function test_swapIn_whenTokenInEqualsTokenOut_itReverts() public {
    BancorExchangeProvider bancorExchangeProvider = initializeBancorExchangeProvider();
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    vm.prank(brokerAddress);
    vm.expectRevert("tokenIn and tokenOut must match exchange");
    bancorExchangeProvider.swapIn(exchangeId, address(token), address(token), 1e18);
  }

  function test_swapIn_whenTokenInIsReserveAsset_shouldSwapIn() public {
    BancorExchangeProvider bancorExchangeProvider = initializeBancorExchangeProvider();
    uint256 amountIn = 1e18;

    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    uint256 reserveBalanceBefore = poolExchange1.reserveBalance;
    uint256 tokenSupplyBefore = poolExchange1.tokenSupply;

    uint256 expectedAmountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountIn: amountIn
    });
    vm.prank(brokerAddress);
    uint256 amountOut = bancorExchangeProvider.swapIn(exchangeId, address(reserveToken), address(token), amountIn);
    assertEq(amountOut, expectedAmountOut);

    (, , uint256 tokenSupplyAfter, uint256 reserveBalanceAfter, , ) = bancorExchangeProvider.exchanges(exchangeId);

    assertEq(reserveBalanceAfter, reserveBalanceBefore + amountIn);
    assertEq(tokenSupplyAfter, tokenSupplyBefore + amountOut);
  }

  function test_swapIn_whenTokenInIsToken_shouldSwapIn() public {
    BancorExchangeProvider bancorExchangeProvider = initializeBancorExchangeProvider();
    uint256 amountIn = 1e18;

    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    uint256 reserveBalanceBefore = poolExchange1.reserveBalance;
    uint256 tokenSupplyBefore = poolExchange1.tokenSupply;

    uint256 expectedAmountOut = bancorExchangeProvider.getAmountOut({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountIn: amountIn
    });
    vm.prank(brokerAddress);
    uint256 amountOut = bancorExchangeProvider.swapIn(exchangeId, address(token), address(reserveToken), amountIn);
    assertEq(amountOut, expectedAmountOut);

    (, , uint256 tokenSupplyAfter, uint256 reserveBalanceAfter, , ) = bancorExchangeProvider.exchanges(exchangeId);

    assertEq(reserveBalanceAfter, reserveBalanceBefore - amountOut);
    assertEq(tokenSupplyAfter, tokenSupplyBefore - amountIn);
  }
}

contract BancorExchangeProviderTest_swapOut is BancorExchangeProviderTest {
  function test_swapOut_whenCallerIsNotBroker_shouldRevert() public {
    BancorExchangeProvider bancorExchangeProvider = initializeBancorExchangeProvider();
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    vm.prank(makeAddr("NotBroker"));
    vm.expectRevert("Caller is not the Broker");
    bancorExchangeProvider.swapOut(exchangeId, address(reserveToken), address(token), 1e18);
  }

  function test_swapOut_whenExchangeDoesNotExist_shouldRevert() public {
    BancorExchangeProvider bancorExchangeProvider = initializeBancorExchangeProvider();
    vm.prank(brokerAddress);
    vm.expectRevert("Exchange does not exist");
    bancorExchangeProvider.swapOut("0xexchangeId", address(reserveToken), address(token), 1e18);
  }

  function test_swapOut_whenTokenInNotInexchange_shouldRevert() public {
    BancorExchangeProvider bancorExchangeProvider = initializeBancorExchangeProvider();
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    vm.prank(brokerAddress);
    vm.expectRevert("tokenIn and tokenOut must match exchange");
    bancorExchangeProvider.swapOut(exchangeId, address(token2), address(token), 1e18);
  }

  function test_swapOut_whenTokenOutNotInexchange_shouldRevert() public {
    BancorExchangeProvider bancorExchangeProvider = initializeBancorExchangeProvider();
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    vm.prank(brokerAddress);
    vm.expectRevert("tokenIn and tokenOut must match exchange");
    bancorExchangeProvider.swapOut(exchangeId, address(token), address(token2), 1e18);
  }

  function test_swapOut_whenTokenInEqualsTokenOut_shouldRevert() public {
    BancorExchangeProvider bancorExchangeProvider = initializeBancorExchangeProvider();
    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    vm.prank(brokerAddress);
    vm.expectRevert("tokenIn and tokenOut must match exchange");
    bancorExchangeProvider.swapOut(exchangeId, address(token), address(token), 1e18);
  }

  function test_swapOut_whenTokenInIsReserveAsset_shouldSwapOut() public {
    BancorExchangeProvider bancorExchangeProvider = initializeBancorExchangeProvider();
    uint256 amountOut = 1e18;

    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    uint256 reserveBalanceBefore = poolExchange1.reserveBalance;
    uint256 tokenSupplyBefore = poolExchange1.tokenSupply;

    uint256 expectedAmountIn = bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(token),
      amountOut: amountOut
    });
    vm.prank(brokerAddress);
    uint256 amountIn = bancorExchangeProvider.swapOut(exchangeId, address(reserveToken), address(token), amountOut);
    assertEq(amountIn, expectedAmountIn);

    (, , uint256 tokenSupplyAfter, uint256 reserveBalanceAfter, , ) = bancorExchangeProvider.exchanges(exchangeId);

    assertEq(reserveBalanceAfter, reserveBalanceBefore + amountIn);
    assertEq(tokenSupplyAfter, tokenSupplyBefore + amountOut);
  }

  function test_swapOut_whenTokenInIsToken_shouldSwapOut() public {
    BancorExchangeProvider bancorExchangeProvider = initializeBancorExchangeProvider();
    uint256 amountOut = 1e18;

    bytes32 exchangeId = bancorExchangeProvider.createExchange(poolExchange1);
    uint256 reserveBalanceBefore = poolExchange1.reserveBalance;
    uint256 tokenSupplyBefore = poolExchange1.tokenSupply;

    uint256 expectedAmountIn = bancorExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(token),
      tokenOut: address(reserveToken),
      amountOut: amountOut
    });
    vm.prank(brokerAddress);
    uint256 amountIn = bancorExchangeProvider.swapOut(exchangeId, address(token), address(reserveToken), amountOut);
    assertEq(amountIn, expectedAmountIn);

    (, , uint256 tokenSupplyAfter, uint256 reserveBalanceAfter, , ) = bancorExchangeProvider.exchanges(exchangeId);

    assertEq(reserveBalanceAfter, reserveBalanceBefore - amountOut);
    assertEq(tokenSupplyAfter, tokenSupplyBefore - amountIn);
  }
}
