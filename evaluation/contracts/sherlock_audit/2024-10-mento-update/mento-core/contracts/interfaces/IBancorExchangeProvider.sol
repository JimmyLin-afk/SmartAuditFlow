// SPDX-License-Identifier: MIT
pragma solidity >=0.5.17 <0.8.19;
pragma experimental ABIEncoderV2;

interface IBancorExchangeProvider {
  struct PoolExchange {
    address reserveAsset;
    address tokenAddress;
    uint256 tokenSupply;
    uint256 reserveBalance;
    uint32 reserveRatio;
    uint32 exitContribution;
  }

  /* ========================================== */
  /* ================= Events ================= */
  /* ========================================== */

  /**
   * @notice Emitted when the broker address is updated.
   * @param newBroker The address of the new broker.
   */
  event BrokerUpdated(address indexed newBroker);

  /**
   * @notice Emitted when the reserve contract is set.
   * @param newReserve The address of the new reserve.
   */
  event ReserveUpdated(address indexed newReserve);

  /**
   * @notice Emitted when a new pool has been created.
   * @param exchangeId The id of the new pool
   * @param reserveAsset The address of the reserve asset
   * @param tokenAddress The address of the token
   */
  event ExchangeCreated(bytes32 indexed exchangeId, address indexed reserveAsset, address indexed tokenAddress);

  /**
   * @notice Emitted when a pool has been destroyed.
   * @param exchangeId The id of the pool to destroy
   * @param reserveAsset The address of the reserve asset
   * @param tokenAddress The address of the token
   */
  event ExchangeDestroyed(bytes32 indexed exchangeId, address indexed reserveAsset, address indexed tokenAddress);

  /**
   * @notice Emitted when the exit contribution for a pool is set.
   * @param exchangeId The id of the pool
   * @param exitContribution The exit contribution
   */
  event ExitContributionSet(bytes32 indexed exchangeId, uint256 exitContribution);

  /* ======================================================== */
  /* ==================== View Functions ==================== */
  /* ======================================================== */

  /**
   * @notice Allows the contract to be upgradable via the proxy.
   * @param _broker The address of the broker contract.
   * @param _reserve The address of the reserve contract.
   */
  function initialize(address _broker, address _reserve) external;

  /**
   * @notice Retrieves the pool with the specified exchangeId.
   * @param exchangeId The ID of the pool to be retrieved.
   * @return exchange The pool with that ID.
   */
  function getPoolExchange(bytes32 exchangeId) external view returns (PoolExchange memory exchange);

  /**
   * @notice Gets all pool IDs.
   * @return exchangeIds List of the pool IDs.
   */
  function getExchangeIds() external view returns (bytes32[] memory exchangeIds);

  /**
   * @notice Gets the current price based of the Bancor formula
   * @param exchangeId The ID of the pool to get the price for
   * @return price The current continuous price of the pool
   */
  function currentPrice(bytes32 exchangeId) external view returns (uint256 price);

  /* ============================================================ */
  /* ==================== Mutative Functions ==================== */
  /* ============================================================ */
  /**
   * @notice Sets the address of the broker contract.
   * @param _broker The new address of the broker contract.
   */
  function setBroker(address _broker) external;

  /**
   * @notice Sets the address of the reserve contract.
   * @param _reserve The new address of the reserve contract.
   */
  function setReserve(address _reserve) external;

  /**
   * @notice Sets the exit contribution for a given pool
   * @param exchangeId The ID of the pool
   * @param exitContribution The exit contribution to be set
   */
  function setExitContribution(bytes32 exchangeId, uint32 exitContribution) external;

  /**
   * @notice Creates a new pool with the given parameters.
   * @param exchange The pool to be created.
   * @return exchangeId The ID of the new pool.
   */
  function createExchange(PoolExchange calldata exchange) external returns (bytes32 exchangeId);

  /**
   * @notice Destroys a pool with the given parameters if it exists.
   * @param exchangeId The ID of the pool to be destroyed.
   * @param exchangeIdIndex The index of the pool in the exchangeIds array.
   * @return destroyed A boolean indicating whether or not the exchange was successfully destroyed.
   */
  function destroyExchange(bytes32 exchangeId, uint256 exchangeIdIndex) external returns (bool destroyed);
}
