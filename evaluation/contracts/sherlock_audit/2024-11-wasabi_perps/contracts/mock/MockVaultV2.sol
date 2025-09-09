// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "../vaults/WasabiVault.sol";

contract MockVaultV2 is WasabiVault {
    uint256 public someNewValue;

    function setSomeNewValue(uint256 _someNewValue) external onlyAdmin {
        someNewValue = _someNewValue;
    }
}