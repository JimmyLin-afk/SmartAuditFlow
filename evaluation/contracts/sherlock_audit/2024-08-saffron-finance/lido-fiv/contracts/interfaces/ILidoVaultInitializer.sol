// SPDX-License-Identifier: MIT

pragma solidity 0.8.18;

/// @notice Defines the parameter struct used during vault initialization
interface ILidoVaultInitializer {
  /// @notice All the required parameters to initialize a vault
  struct InitializationParams {
    uint256 vaultId;
    uint256 duration;
    uint256 fixedSideCapacity;
    uint256 variableSideCapacity;
    uint256 earlyExitFeeBps;
    uint256 protocolFeeBps;
    address protocolFeeReceiver;
  }
}