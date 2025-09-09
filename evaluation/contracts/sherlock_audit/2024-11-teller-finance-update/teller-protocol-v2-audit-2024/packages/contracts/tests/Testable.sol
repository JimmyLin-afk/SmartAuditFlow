pragma solidity >=0.8.0 <0.9.0;

// SPDX-License-Identifier: MIT

import "./util/FoundryTest.sol";

abstract contract Testable is Test {
    receive() external payable {}
}
