// SPDX-License-Identifier: MIT
pragma solidity 0.8.18;

import { PausableUpgradeable } from "openzeppelin-contracts-upgradeable/contracts/security/PausableUpgradeable.sol";

import { IGoodDollarExchangeProvider } from "contracts/interfaces/IGoodDollarExchangeProvider.sol";
import { IGoodDollarExpansionController } from "contracts/interfaces/IGoodDollarExpansionController.sol";

import { BancorExchangeProvider } from "./BancorExchangeProvider.sol";
import { UD60x18, unwrap, wrap } from "prb/math/UD60x18.sol";

/**
 * @title GoodDollarExchangeProvider
 * @notice Provides exchange functionality for the GoodDollar system.
 */
contract GoodDollarExchangeProvider is IGoodDollarExchangeProvider, BancorExchangeProvider, PausableUpgradeable {
  /* ========================================================= */
  /* ==================== State Variables ==================== */
  /* ========================================================= */

  // Address of the Expansion Controller contract.
  IGoodDollarExpansionController public expansionController;

  // Address of the GoodDollar DAO contract.
  // solhint-disable-next-line var-name-mixedcase
  address public AVATAR;

  /* ===================================================== */
  /* ==================== Constructor ==================== */
  /* ===================================================== */

  /**
   * @dev Should be called with disable=true in deployments when it's accessed through a Proxy.
   * Call this with disable=false during testing, when used without a proxy.
   * @param disable Set to true to run `_disableInitializers()` inherited from
   * openzeppelin-contracts-upgradeable/Initializable.sol
   */
  constructor(bool disable) BancorExchangeProvider(disable) {}

  /// @inheritdoc IGoodDollarExchangeProvider
  function initialize(
    address _broker,
    address _reserve,
    address _expansionController,
    address _avatar
  ) public initializer {
    BancorExchangeProvider._initialize(_broker, _reserve);
    __Pausable_init();

    setExpansionController(_expansionController);
    setAvatar(_avatar);
  }

  /* =================================================== */
  /* ==================== Modifiers ==================== */
  /* =================================================== */

  modifier onlyAvatar() {
    require(msg.sender == AVATAR, "Only Avatar can call this function");
    _;
  }

  modifier onlyExpansionController() {
    require(msg.sender == address(expansionController), "Only ExpansionController can call this function");
    _;
  }

  /* ============================================================ */
  /* ==================== Mutative Functions ==================== */
  /* ============================================================ */

  /// @inheritdoc IGoodDollarExchangeProvider
  function setAvatar(address _avatar) public onlyOwner {
    require(_avatar != address(0), "Avatar address must be set");
    AVATAR = _avatar;
    emit AvatarUpdated(_avatar);
  }

  /// @inheritdoc IGoodDollarExchangeProvider
  function setExpansionController(address _expansionController) public onlyOwner {
    require(_expansionController != address(0), "ExpansionController address must be set");
    expansionController = IGoodDollarExpansionController(_expansionController);
    emit ExpansionControllerUpdated(_expansionController);
  }

  /**
   * @inheritdoc BancorExchangeProvider
   * @dev Only callable by the GoodDollar DAO contract.
   */
  function setExitContribution(bytes32 exchangeId, uint32 exitContribution) external override onlyAvatar {
    return _setExitContribution(exchangeId, exitContribution);
  }

  /**
   * @inheritdoc BancorExchangeProvider
   * @dev Only callable by the GoodDollar DAO contract.
   */
  function createExchange(PoolExchange calldata _exchange) external override onlyAvatar returns (bytes32 exchangeId) {
    return _createExchange(_exchange);
  }

  /**
   * @inheritdoc BancorExchangeProvider
   * @dev Only callable by the GoodDollar DAO contract.
   */
  function destroyExchange(
    bytes32 exchangeId,
    uint256 exchangeIdIndex
  ) external override onlyAvatar returns (bool destroyed) {
    return _destroyExchange(exchangeId, exchangeIdIndex);
  }

  /// @inheritdoc BancorExchangeProvider
  function swapIn(
    bytes32 exchangeId,
    address tokenIn,
    address tokenOut,
    uint256 amountIn
  ) public override onlyBroker whenNotPaused returns (uint256 amountOut) {
    amountOut = BancorExchangeProvider.swapIn(exchangeId, tokenIn, tokenOut, amountIn);
  }

  /// @inheritdoc BancorExchangeProvider
  function swapOut(
    bytes32 exchangeId,
    address tokenIn,
    address tokenOut,
    uint256 amountOut
  ) public override onlyBroker whenNotPaused returns (uint256 amountIn) {
    amountIn = BancorExchangeProvider.swapOut(exchangeId, tokenIn, tokenOut, amountOut);
  }

  /**
   * @inheritdoc IGoodDollarExchangeProvider
   * @dev Calculates the amount of G$ tokens that need to be minted as a result of the expansion
   *      while keeping the current price the same.
   *      calculation: amountToMint = (tokenSupply * reserveRatio - tokenSupply * newRatio) / newRatio
   */
  function mintFromExpansion(
    bytes32 exchangeId,
    uint256 reserveRatioScalar
  ) external onlyExpansionController whenNotPaused returns (uint256 amountToMint) {
    require(reserveRatioScalar > 0, "Reserve ratio scalar must be greater than 0");
    PoolExchange memory exchange = getPoolExchange(exchangeId);

    UD60x18 scaledRatio = wrap(uint256(exchange.reserveRatio) * 1e10);
    UD60x18 newRatio = scaledRatio.mul(wrap(reserveRatioScalar));

    uint32 newRatioUint = uint32(unwrap(newRatio) / 1e10);
    require(newRatioUint > 0, "New ratio must be greater than 0");

    UD60x18 numerator = wrap(exchange.tokenSupply).mul(scaledRatio);
    numerator = numerator.sub(wrap(exchange.tokenSupply).mul(newRatio));

    uint256 scaledAmountToMint = unwrap(numerator.div(newRatio));

    exchanges[exchangeId].reserveRatio = newRatioUint;
    exchanges[exchangeId].tokenSupply += scaledAmountToMint;

    amountToMint = scaledAmountToMint / tokenPrecisionMultipliers[exchange.tokenAddress];
    emit ReserveRatioUpdated(exchangeId, newRatioUint);

    return amountToMint;
  }

  /**
   * @inheritdoc IGoodDollarExchangeProvider
   * @dev Calculates the amount of G$ tokens that need to be minted as a result of the reserve interest
   *      flowing into the reserve while keeping the current price the same.
   *      calculation: amountToMint = reserveInterest * tokenSupply / reserveBalance
   */
  function mintFromInterest(
    bytes32 exchangeId,
    uint256 reserveInterest
  ) external onlyExpansionController whenNotPaused returns (uint256 amountToMint) {
    PoolExchange memory exchange = getPoolExchange(exchangeId);

    uint256 reserveinterestScaled = reserveInterest * tokenPrecisionMultipliers[exchange.reserveAsset];
    uint256 amountToMintScaled = unwrap(
      wrap(reserveinterestScaled).mul(wrap(exchange.tokenSupply)).div(wrap(exchange.reserveBalance))
    );
    amountToMint = amountToMintScaled / tokenPrecisionMultipliers[exchange.tokenAddress];

    exchanges[exchangeId].tokenSupply += amountToMintScaled;
    exchanges[exchangeId].reserveBalance += reserveinterestScaled;

    return amountToMint;
  }

  /**
   * @inheritdoc IGoodDollarExchangeProvider
   * @dev Calculates the new reserve ratio needed to mint the G$ reward while keeping the current price the same.
   *      calculation: newRatio = reserveBalance / (tokenSupply + reward) * currentPrice
   */
  function updateRatioForReward(bytes32 exchangeId, uint256 reward) external onlyExpansionController whenNotPaused {
    PoolExchange memory exchange = getPoolExchange(exchangeId);

    uint256 currentPriceScaled = currentPrice(exchangeId) * tokenPrecisionMultipliers[exchange.reserveAsset];
    uint256 rewardScaled = reward * tokenPrecisionMultipliers[exchange.tokenAddress];

    UD60x18 numerator = wrap(exchange.reserveBalance);
    UD60x18 denominator = wrap(exchange.tokenSupply + rewardScaled).mul(wrap(currentPriceScaled));
    uint256 newRatioScaled = unwrap(numerator.div(denominator));

    uint32 newRatioUint = uint32(newRatioScaled / 1e10);
    exchanges[exchangeId].reserveRatio = newRatioUint;
    exchanges[exchangeId].tokenSupply += rewardScaled;

    emit ReserveRatioUpdated(exchangeId, newRatioUint);
  }

  /**
   * @inheritdoc IGoodDollarExchangeProvider
   * @dev Only callable by the GoodDollar DAO contract.
   */
  function pause() external virtual onlyAvatar {
    _pause();
  }

  /**
   * @inheritdoc IGoodDollarExchangeProvider
   * @dev Only callable by the GoodDollar DAO contract.
   */
  function unpause() external virtual onlyAvatar {
    _unpause();
  }

  /**
   * @dev This empty reserved space is put in place to allow future versions to add new
   * variables without shifting down storage in the inheritance chain.
   * See https://docs.openzeppelin.com/contracts/4.x/upgradeable#storage_gaps
   */
  uint256[50] private __gap;
}
