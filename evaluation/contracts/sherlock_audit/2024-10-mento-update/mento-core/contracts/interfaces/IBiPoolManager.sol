// SPDX-License-Identifier: GPL-3.0-or-later
// solhint-disable func-name-mixedcase
pragma solidity >0.5.13 <0.9;
pragma experimental ABIEncoderV2;

import { IPricingModule } from "./IPricingModule.sol";
import { IReserve } from "./IReserve.sol";
import { ISortedOracles } from "./ISortedOracles.sol";
import { IBreakerBox } from "./IBreakerBox.sol";
import { IExchangeProvider } from "./IExchangeProvider.sol";

import { FixidityLib } from "celo/contracts/common/FixidityLib.sol";

/**
 * @title BiPool Manager interface
 * @notice An exchange provider implementation managing the state of all two-asset virtual pools.
 */
interface IBiPoolManager {
  /**
   * @title PoolExchange
   * @notice The PoolExchange is a type of asset exchange that
   * that implements an AMM with two virtual buckets.
   */
  struct PoolExchange {
    address asset0;
    address asset1;
    IPricingModule pricingModule;
    uint256 bucket0;
    uint256 bucket1;
    uint256 lastBucketUpdate;
    PoolConfig config;
  }

  /**
   * @notice Variables related to bucket updates and sizing.
   * @dev Broken down into a separate struct because the compiler
   * version doesn't support structs with too many members.
   * Sad reacts only.
   */
  struct PoolConfig {
    FixidityLib.Fraction spread;
    address referenceRateFeedID; // rateFeedID of the price that this pool follows (i.e. it's reference rate)
    uint256 referenceRateResetFrequency;
    uint256 minimumReports;
    uint256 stablePoolResetSize;
  }

  /**
   * @notice Emitted when a new PoolExchange has been created.
   * @param exchangeId The id of the new PoolExchange
   * @param asset0 The address of asset0
   * @param asset1 The address of asset1
   * @param pricingModule the address of the pricingModule
   */
  event ExchangeCreated(
    bytes32 indexed exchangeId,
    address indexed asset0,
    address indexed asset1,
    address pricingModule
  );

  /**
   * @notice Emitted when a PoolExchange has been destroyed.
   * @param exchangeId The id of the PoolExchange
   * @param asset0 The address of asset0
   * @param asset1 The address of asset1
   * @param pricingModule the address of the pricingModule
   */
  event ExchangeDestroyed(
    bytes32 indexed exchangeId,
    address indexed asset0,
    address indexed asset1,
    address pricingModule
  );

  /**
   * @notice Emitted when the broker address is updated.
   * @param newBroker The address of the new broker.
   */
  event BrokerUpdated(address indexed newBroker);

  /**
   * @notice Emitted when the reserve address is updated.
   * @param newReserve The address of the new reserve.
   */
  event ReserveUpdated(address indexed newReserve);

  /**
   * @notice Emitted when the breakerBox address is updated.
   * @param newBreakerBox The address of the new breakerBox.
   */
  event BreakerBoxUpdated(address newBreakerBox);

  /**
   * @notice Emitted when the sortedOracles address is updated.
   * @param newSortedOracles The address of the new sortedOracles.
   */
  event SortedOraclesUpdated(address indexed newSortedOracles);

  /**
   * @notice Emitted when the buckets for a specified exchange are updated.
   * @param exchangeId The id of the exchange
   * @param bucket0 The new bucket0 size
   * @param bucket1 The new bucket1 size
   */
  event BucketsUpdated(bytes32 indexed exchangeId, uint256 bucket0, uint256 bucket1);

  /**
   * @notice Emitted when the pricing modules have been updated.
   * @param newIdentifiers The new identifiers.
   * @param newAddresses The new pricing module addresses.
   */
  event PricingModulesUpdated(bytes32[] newIdentifiers, address[] newAddresses);

  /**
   * @notice Retrieves the pool with the specified exchangeId.
   * @param exchangeId The id of the pool to be retrieved.
   * @return exchange The PoolExchange with that ID.
   */
  function getPoolExchange(bytes32 exchangeId) external view returns (PoolExchange memory exchange);

  /**
   * @notice Get all exchange IDs.
   * @return _exchangeIds List of the exchangeIds.
   */
  function getExchangeIds() external view returns (bytes32[] memory _exchangeIds);

  /**
   * @notice Create a PoolExchange with the provided data.
   * @param exchange The PoolExchange to be created.
   * @return exchangeId The id of the exchange.
   */
  function createExchange(PoolExchange calldata exchange) external returns (bytes32 exchangeId);

  /**
   * @notice Delete a PoolExchange.
   * @param exchangeId The PoolExchange to be created.
   * @param exchangeIdIndex The index of the exchangeId in the exchangeIds array.
   * @return destroyed - true on successful delition.
   */
  function destroyExchange(bytes32 exchangeId, uint256 exchangeIdIndex) external returns (bool destroyed);

  /**
   * @notice Allows the contract to be upgradable via the proxy.
   * @param _broker The address of the broker contract.
   * @param _reserve The address of the reserve contract.
   * @param _sortedOracles The address of the sorted oracles contract.
   * @param _breakerBox The address of the breaker box contract.
   */
  function initialize(
    address _broker,
    IReserve _reserve,
    ISortedOracles _sortedOracles,
    IBreakerBox _breakerBox
  ) external;

  function swapIn(
    bytes32 exchangeId,
    address tokenIn,
    address tokenOut,
    uint256 amountIn
  ) external returns (uint256 amountOut);

  function swapOut(
    bytes32 exchangeId,
    address tokenIn,
    address tokenOut,
    uint256 amountOut
  ) external returns (uint256 amountIn);

  /**
   * @notice Updates the pricing modules for a list of identifiers
   * @dev This function can only be called by the owner of the contract.
   *      The number of identifiers and modules provided must be the same.
   * @param identifiers An array of identifiers for which the pricing modules are to be set.
   * @param addresses An array of module addresses corresponding to each identifier.
   */
  function setPricingModules(bytes32[] calldata identifiers, address[] calldata addresses) external;

  // @notice Getters:
  function broker() external view returns (address);

  function exchanges(bytes32) external view returns (PoolExchange memory);

  function exchangeIds(uint256) external view returns (bytes32);

  function reserve() external view returns (IReserve);

  function sortedOracles() external view returns (ISortedOracles);

  function breakerBox() external view returns (IBreakerBox);

  function tokenPrecisionMultipliers(address) external view returns (uint256);

  function CONSTANT_SUM() external view returns (bytes32);

  function CONSTANT_PRODUCT() external view returns (bytes32);

  function pricingModules(bytes32) external view returns (address);

  function getExchanges() external view returns (IExchangeProvider.Exchange[] memory);

  function getAmountOut(
    bytes32 exchangeId,
    address tokenIn,
    address tokenOut,
    uint256 amountIn
  ) external view returns (uint256 amountOut);

  function getAmountIn(
    bytes32 exchangeId,
    address tokenIn,
    address tokenOut,
    uint256 amountOut
  ) external view returns (uint256 amountIn);

  /// @notice Setters:
  function setBroker(address newBroker) external;

  function setReserve(IReserve newReserve) external;

  function setSortedOracles(ISortedOracles newSortedOracles) external;

  function setBreakerBox(IBreakerBox newBreakerBox) external;

  /// @notice IOwnable:
  function transferOwnership(address newOwner) external;

  function renounceOwnership() external;

  function owner() external view returns (address);
}
