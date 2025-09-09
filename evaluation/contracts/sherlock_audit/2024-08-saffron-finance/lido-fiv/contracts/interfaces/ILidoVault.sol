// SPDX-License-Identifier: MIT

pragma solidity 0.8.18;

import "./ILidoVaultInitializer.sol";

/// @title Saffron Fixed Income Lido Vault Interface
interface ILidoVault is ILidoVaultInitializer {
  /// @notice Initializes the vault
  /// @param params Initialization parameters
  /// @dev make sure this is only callable by the contract creator aka the vault factory
  function initialize(InitializationParams memory params) external;
}
