// SPDX-License-Identifier: MIT
pragma solidity 0.8.18;

import { IBancorExchangeProvider } from "contracts/interfaces/IBancorExchangeProvider.sol";
import { IExchangeProvider } from "contracts/interfaces/IExchangeProvider.sol";
import { IERC20 } from "contracts/interfaces/IERC20.sol";
import { IReserve } from "contracts/interfaces/IReserve.sol";

import { OwnableUpgradeable } from "openzeppelin-contracts-upgradeable/contracts/access/OwnableUpgradeable.sol";

import { BancorFormula } from "contracts/goodDollar/BancorFormula.sol";
import { UD60x18, unwrap, wrap } from "prb/math/UD60x18.sol";

/**
 * @title BancorExchangeProvider
 * @notice Provides exchange functionality for Bancor pools.
 */
contract BancorExchangeProvider is IExchangeProvider, IBancorExchangeProvider, BancorFormula, OwnableUpgradeable {
  /* ========================================================= */
  /* ==================== State Variables ==================== */
  /* ========================================================= */

  // Address of the broker contract.
  address public broker;

  // Address of the reserve contract.
  IReserve public reserve;

  // Maps an exchange id to the corresponding PoolExchange struct.
  // exchangeId is in the format "asset0Symbol:asset1Symbol"
  mapping(bytes32 => PoolExchange) public exchanges;
  bytes32[] public exchangeIds;

  // Token precision multiplier used to normalize values to the same precision when calculating amounts.
  mapping(address => uint256) public tokenPrecisionMultipliers;

  /* ===================================================== */
  /* ==================== Constructor ==================== */
  /* ===================================================== */

  /**
   * @dev Should be called with disable=true in deployments when it's accessed through a Proxy.
   * Call this with disable=false during testing, when used without a proxy.
   * @param disable Set to true to run `_disableInitializers()` inherited from
   * openzeppelin-contracts-upgradeable/Initializable.sol
   */
  constructor(bool disable) {
    if (disable) {
      _disableInitializers();
    }
  }

  /// @inheritdoc IBancorExchangeProvider
  function initialize(address _broker, address _reserve) public initializer {
    _initialize(_broker, _reserve);
  }

  function _initialize(address _broker, address _reserve) internal onlyInitializing {
    __Ownable_init();

    BancorFormula.init();
    setBroker(_broker);
    setReserve(_reserve);
  }

  /* =================================================== */
  /* ==================== Modifiers ==================== */
  /* =================================================== */

  modifier onlyBroker() {
    require(msg.sender == broker, "Caller is not the Broker");
    _;
  }

  modifier verifyExchangeTokens(address tokenIn, address tokenOut, PoolExchange memory exchange) {
    require(
      (tokenIn == exchange.reserveAsset && tokenOut == exchange.tokenAddress) ||
        (tokenIn == exchange.tokenAddress && tokenOut == exchange.reserveAsset),
      "tokenIn and tokenOut must match exchange"
    );
    _;
  }

  /* ======================================================== */
  /* ==================== View Functions ==================== */
  /* ======================================================== */

  /// @inheritdoc IBancorExchangeProvider
  function getPoolExchange(bytes32 exchangeId) public view returns (PoolExchange memory exchange) {
    exchange = exchanges[exchangeId];
    require(exchange.tokenAddress != address(0), "Exchange does not exist");
    return exchange;
  }

  /// @inheritdoc IBancorExchangeProvider
  function getExchangeIds() external view returns (bytes32[] memory) {
    return exchangeIds;
  }

  /**
   * @inheritdoc IExchangeProvider
   * @dev We don't expect the number of exchanges to grow to
   * astronomical values so this is safe gas-wise as is.
   */
  function getExchanges() public view returns (Exchange[] memory _exchanges) {
    uint256 numExchanges = exchangeIds.length;
    _exchanges = new Exchange[](numExchanges);
    for (uint256 i = 0; i < numExchanges; i++) {
      _exchanges[i].exchangeId = exchangeIds[i];
      _exchanges[i].assets = new address[](2);
      _exchanges[i].assets[0] = exchanges[exchangeIds[i]].reserveAsset;
      _exchanges[i].assets[1] = exchanges[exchangeIds[i]].tokenAddress;
    }
  }

  /// @inheritdoc IExchangeProvider
  function getAmountOut(
    bytes32 exchangeId,
    address tokenIn,
    address tokenOut,
    uint256 amountIn
  ) external view virtual returns (uint256 amountOut) {
    PoolExchange memory exchange = getPoolExchange(exchangeId);
    uint256 scaledAmountIn = amountIn * tokenPrecisionMultipliers[tokenIn];
    uint256 scaledAmountOut = _getScaledAmountOut(exchange, tokenIn, tokenOut, scaledAmountIn);
    amountOut = scaledAmountOut / tokenPrecisionMultipliers[tokenOut];
    return amountOut;
  }

  /// @inheritdoc IExchangeProvider
  function getAmountIn(
    bytes32 exchangeId,
    address tokenIn,
    address tokenOut,
    uint256 amountOut
  ) external view virtual returns (uint256 amountIn) {
    PoolExchange memory exchange = getPoolExchange(exchangeId);
    uint256 scaledAmountOut = amountOut * tokenPrecisionMultipliers[tokenOut];
    uint256 scaledAmountIn = _getScaledAmountIn(exchange, tokenIn, tokenOut, scaledAmountOut);
    amountIn = scaledAmountIn / tokenPrecisionMultipliers[tokenIn];
    return amountIn;
  }

  /// @inheritdoc IBancorExchangeProvider
  function currentPrice(bytes32 exchangeId) public view returns (uint256 price) {
    // calculates: reserveBalance / (tokenSupply * reserveRatio)
    require(exchanges[exchangeId].reserveAsset != address(0), "Exchange does not exist");
    PoolExchange memory exchange = getPoolExchange(exchangeId);
    uint256 scaledReserveRatio = uint256(exchange.reserveRatio) * 1e10;
    UD60x18 denominator = wrap(exchange.tokenSupply).mul(wrap(scaledReserveRatio));
    price = unwrap(wrap(exchange.reserveBalance).div(denominator));
    return price;
  }

  /* ============================================================ */
  /* ==================== Mutative Functions ==================== */
  /* ============================================================ */

  /// @inheritdoc IBancorExchangeProvider
  function setBroker(address _broker) public onlyOwner {
    require(_broker != address(0), "Broker address must be set");
    broker = _broker;
    emit BrokerUpdated(_broker);
  }

  /// @inheritdoc IBancorExchangeProvider
  function setReserve(address _reserve) public onlyOwner {
    require(address(_reserve) != address(0), "Reserve address must be set");
    reserve = IReserve(_reserve);
    emit ReserveUpdated(address(_reserve));
  }

  /// @inheritdoc IBancorExchangeProvider
  function setExitContribution(bytes32 exchangeId, uint32 exitContribution) external virtual onlyOwner {
    return _setExitContribution(exchangeId, exitContribution);
  }

  /// @inheritdoc IBancorExchangeProvider
  function createExchange(PoolExchange calldata _exchange) external virtual onlyOwner returns (bytes32 exchangeId) {
    return _createExchange(_exchange);
  }

  /// @inheritdoc IBancorExchangeProvider
  function destroyExchange(
    bytes32 exchangeId,
    uint256 exchangeIdIndex
  ) external virtual onlyOwner returns (bool destroyed) {
    return _destroyExchange(exchangeId, exchangeIdIndex);
  }

  /// @inheritdoc IExchangeProvider
  function swapIn(
    bytes32 exchangeId,
    address tokenIn,
    address tokenOut,
    uint256 amountIn
  ) public virtual onlyBroker returns (uint256 amountOut) {
    PoolExchange memory exchange = getPoolExchange(exchangeId);
    uint256 scaledAmountIn = amountIn * tokenPrecisionMultipliers[tokenIn];
    uint256 scaledAmountOut = _getScaledAmountOut(exchange, tokenIn, tokenOut, scaledAmountIn);
    executeSwap(exchangeId, tokenIn, scaledAmountIn, scaledAmountOut);

    amountOut = scaledAmountOut / tokenPrecisionMultipliers[tokenOut];
    return amountOut;
  }

  /// @inheritdoc IExchangeProvider
  function swapOut(
    bytes32 exchangeId,
    address tokenIn,
    address tokenOut,
    uint256 amountOut
  ) public virtual onlyBroker returns (uint256 amountIn) {
    PoolExchange memory exchange = getPoolExchange(exchangeId);
    uint256 scaledAmountOut = amountOut * tokenPrecisionMultipliers[tokenOut];
    uint256 scaledAmountIn = _getScaledAmountIn(exchange, tokenIn, tokenOut, scaledAmountOut);
    executeSwap(exchangeId, tokenIn, scaledAmountIn, scaledAmountOut);

    amountIn = scaledAmountIn / tokenPrecisionMultipliers[tokenIn];
    return amountIn;
  }

  /* =========================================================== */
  /* ==================== Private Functions ==================== */
  /* =========================================================== */

  function _createExchange(PoolExchange calldata _exchange) internal returns (bytes32 exchangeId) {
    PoolExchange memory exchange = _exchange;
    validateExchange(exchange);

    // slither-disable-next-line encode-packed-collision
    exchangeId = keccak256(
      abi.encodePacked(IERC20(exchange.reserveAsset).symbol(), IERC20(exchange.tokenAddress).symbol())
    );
    require(exchanges[exchangeId].reserveAsset == address(0), "Exchange already exists");

    uint256 reserveAssetDecimals = IERC20(exchange.reserveAsset).decimals();
    uint256 tokenDecimals = IERC20(exchange.tokenAddress).decimals();
    require(reserveAssetDecimals <= 18, "Reserve asset decimals must be <= 18");
    require(tokenDecimals <= 18, "Token decimals must be <= 18");

    tokenPrecisionMultipliers[exchange.reserveAsset] = 10 ** (18 - uint256(reserveAssetDecimals));
    tokenPrecisionMultipliers[exchange.tokenAddress] = 10 ** (18 - uint256(tokenDecimals));

    exchanges[exchangeId] = exchange;
    exchangeIds.push(exchangeId);
    emit ExchangeCreated(exchangeId, exchange.reserveAsset, exchange.tokenAddress);
  }

  function _destroyExchange(bytes32 exchangeId, uint256 exchangeIdIndex) internal returns (bool destroyed) {
    require(exchangeIdIndex < exchangeIds.length, "exchangeIdIndex not in range");
    require(exchangeIds[exchangeIdIndex] == exchangeId, "exchangeId at index doesn't match");
    PoolExchange memory exchange = exchanges[exchangeId];

    delete exchanges[exchangeId];
    exchangeIds[exchangeIdIndex] = exchangeIds[exchangeIds.length - 1];
    exchangeIds.pop();
    destroyed = true;

    emit ExchangeDestroyed(exchangeId, exchange.reserveAsset, exchange.tokenAddress);
  }

  function _setExitContribution(bytes32 exchangeId, uint32 exitContribution) internal {
    require(exchanges[exchangeId].reserveAsset != address(0), "Exchange does not exist");
    require(exitContribution <= MAX_WEIGHT, "Exit contribution is too high");

    PoolExchange storage exchange = exchanges[exchangeId];
    exchange.exitContribution = exitContribution;
    emit ExitContributionSet(exchangeId, exitContribution);
  }

  /**
   * @notice Execute a swap against the in-memory exchange and write the new exchange state to storage.
   * @param exchangeId The ID of the pool
   * @param tokenIn The token to be sold
   * @param scaledAmountIn The amount of tokenIn to be sold, scaled to 18 decimals
   * @param scaledAmountOut The amount of tokenOut to be bought, scaled to 18 decimals
   */
  function executeSwap(bytes32 exchangeId, address tokenIn, uint256 scaledAmountIn, uint256 scaledAmountOut) internal {
    PoolExchange memory exchange = getPoolExchange(exchangeId);
    if (tokenIn == exchange.reserveAsset) {
      exchange.reserveBalance += scaledAmountIn;
      exchange.tokenSupply += scaledAmountOut;
    } else {
      require(exchange.reserveBalance >= scaledAmountOut, "Insufficient reserve balance for swap");
      exchange.reserveBalance -= scaledAmountOut;
      exchange.tokenSupply -= scaledAmountIn;
    }
    exchanges[exchangeId].reserveBalance = exchange.reserveBalance;
    exchanges[exchangeId].tokenSupply = exchange.tokenSupply;
  }

  /**
   * @notice Calculate the scaledAmountIn of tokenIn for a given scaledAmountOut of tokenOut
   * @param exchange The pool exchange to operate on
   * @param tokenIn The token to be sold
   * @param tokenOut The token to be bought
   * @param scaledAmountOut The amount of tokenOut to be bought, scaled to 18 decimals
   * @return scaledAmountIn The amount of tokenIn to be sold, scaled to 18 decimals
   */
  function _getScaledAmountIn(
    PoolExchange memory exchange,
    address tokenIn,
    address tokenOut,
    uint256 scaledAmountOut
  ) internal view verifyExchangeTokens(tokenIn, tokenOut, exchange) returns (uint256 scaledAmountIn) {
    if (tokenIn == exchange.reserveAsset) {
      scaledAmountIn = fundCost(exchange.tokenSupply, exchange.reserveBalance, exchange.reserveRatio, scaledAmountOut);
    } else {
      // apply exit contribution
      scaledAmountOut = (scaledAmountOut * MAX_WEIGHT) / (MAX_WEIGHT - exchange.exitContribution);
      scaledAmountIn = saleCost(exchange.tokenSupply, exchange.reserveBalance, exchange.reserveRatio, scaledAmountOut);
    }
  }

  /**
   * @notice Calculate the scaledAmountOut of tokenOut received for a given scaledAmountIn of tokenIn
   * @param exchange The pool exchange to operate on
   * @param tokenIn The token to be sold
   * @param tokenOut The token to be bought
   * @param scaledAmountIn The amount of tokenIn to be sold, scaled to 18 decimals
   * @return scaledAmountOut The amount of tokenOut to be bought, scaled to 18 decimals
   */
  function _getScaledAmountOut(
    PoolExchange memory exchange,
    address tokenIn,
    address tokenOut,
    uint256 scaledAmountIn
  ) internal view verifyExchangeTokens(tokenIn, tokenOut, exchange) returns (uint256 scaledAmountOut) {
    if (tokenIn == exchange.reserveAsset) {
      scaledAmountOut = purchaseTargetAmount(
        exchange.tokenSupply,
        exchange.reserveBalance,
        exchange.reserveRatio,
        scaledAmountIn
      );
    } else {
      scaledAmountOut = saleTargetAmount(
        exchange.tokenSupply,
        exchange.reserveBalance,
        exchange.reserveRatio,
        scaledAmountIn
      );
      // apply exit contribution
      scaledAmountOut = (scaledAmountOut * (MAX_WEIGHT - exchange.exitContribution)) / MAX_WEIGHT;
    }
  }

  /**
   * @notice Validates a PoolExchange's parameters and configuration
   * @dev Reverts if not valid
   * @param exchange The PoolExchange to validate
   */
  function validateExchange(PoolExchange memory exchange) internal view {
    require(exchange.reserveAsset != address(0), "Invalid reserve asset");
    require(
      reserve.isCollateralAsset(exchange.reserveAsset),
      "Reserve asset must be a collateral registered with the reserve"
    );
    require(exchange.tokenAddress != address(0), "Invalid token address");
    require(reserve.isStableAsset(exchange.tokenAddress), "Token must be a stable registered with the reserve");
    require(exchange.reserveRatio > 1, "Reserve ratio is too low");
    require(exchange.reserveRatio <= MAX_WEIGHT, "Reserve ratio is too high");
    require(exchange.exitContribution <= MAX_WEIGHT, "Exit contribution is too high");
  }

  /**
   * @dev This empty reserved space is put in place to allow future versions to add new
   * variables without shifting down storage in the inheritance chain.
   * See https://docs.openzeppelin.com/contracts/4.x/upgradeable#storage_gaps
   */
  uint256[50] private __gap;
}
