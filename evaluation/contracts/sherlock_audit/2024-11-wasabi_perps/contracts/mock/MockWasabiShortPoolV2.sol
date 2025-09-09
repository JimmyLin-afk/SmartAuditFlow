// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "../WasabiShortPool.sol";

contract MockWasabiShortPoolV2 is WasabiShortPool {
    uint256 public someNewValue;

    function setSomeNewValue(uint256 _someNewValue) external onlyAdmin {
        someNewValue = _someNewValue;
    }
}
