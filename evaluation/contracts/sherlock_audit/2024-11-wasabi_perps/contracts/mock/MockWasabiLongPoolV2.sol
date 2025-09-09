// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "../WasabiLongPool.sol";

contract MockWasabiLongPoolV2 is WasabiLongPool {
    uint256 public someNewValue;

    function setSomeNewValue(uint256 _someNewValue) external onlyAdmin {
        someNewValue = _someNewValue;
    }
}