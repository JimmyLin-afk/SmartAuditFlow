// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity >=0.5.13 <0.8.19;

/*
 * @title Broker Admin Interface
 * @notice Contains admin functions to configure the broker that
 *         should be only callable by the owner.
 */
interface IBrokerAdmin {
  /**
   * @notice Emitted when an ExchangeProvider is added.
   * @param exchangeProvider The address of the ExchangeProvider.
   */
  event ExchangeProviderAdded(address indexed exchangeProvider);

  /**
   * @notice Emitted when an ExchangeProvider is removed.
   * @param exchangeProvider The address of the ExchangeProvider.
   */
  event ExchangeProviderRemoved(address indexed exchangeProvider);

  /**
   * @notice Emitted when the reserve is updated.
   * @param newAddress The new address.
   * @param prevAddress The previous address.
   */
  event ReserveSet(address indexed newAddress, address indexed prevAddress);

  /**
   * @notice Remove an ExchangeProvider at a specified index.
   * @param exchangeProvider The address of the ExchangeProvider to remove.
   * @param index The index in the exchange providers array.
   */
  function removeExchangeProvider(address exchangeProvider, uint256 index) external;

  /**
   * @notice Add an ExchangeProvider.
   * @param exchangeProvider The address of the ExchangeProvider to add.
   * @return index The index where the ExchangeProvider was inserted.
   */
  function addExchangeProvider(address exchangeProvider, address reserve) external returns (uint256 index);

  /**
   * @notice Set the reserves for the exchange providers.
   * @param _exchangeProviders The addresses of the ExchangeProvider contracts.
   * @param _reserves The addresses of the Reserve contracts.
   */
  function setReserves(address[] calldata _exchangeProviders, address[] calldata _reserves) external;
}
