// SPDX-License-Identifier: MIT

pragma solidity 0.8.18;

import "../interfaces/ILidoVaultInitializer.sol";

import "../LidoVault.sol";

contract EchidnaCreateVault is ILidoVaultInitializer {
  LidoVault vault;

  uint256 private constant FIXED = 0;
  uint256 private constant VARIABLE = 1;

  function createVault(InitializationParams memory params) internal {
    vault = new LidoVault(false);

    vault.initialize(params);

    assert(vault.fixedSideCapacity() > 0);
  }

  function fixedDeposit(InitializationParams memory params) public payable {
    createVault(params);

    uint256 stETHBalanceBefore = vault.stakingBalance();
    uint256 stakeAmount = msg.value;

    vault.deposit{value: stakeAmount}(FIXED);

    assert(vault.stakingBalance() > 0);
    uint256 stETHReceived = (vault.stakingBalance() - stETHBalanceBefore);
    assert((stETHReceived >= stakeAmount) || (stakeAmount - stETHReceived <= 10 wei));
  }

  function variableDeposit(InitializationParams memory params) public payable {
    createVault(params);
    vault.deposit{value: msg.value}(VARIABLE);
    assert(address(vault).balance == msg.value);
  }

  function withdrawFixed(InitializationParams memory params) public payable {
    createVault(params);
    vault.deposit{value: msg.value}(FIXED);

    vault.withdraw(FIXED);
    assert(vault.stakingBalance() == 0);
  }

  function withdrawVariable(InitializationParams memory params) public payable {
    createVault(params);
    vault.deposit{value: msg.value}(VARIABLE);

    vault.withdraw(VARIABLE);
    assert(address(vault).balance == 0);
  }
}
