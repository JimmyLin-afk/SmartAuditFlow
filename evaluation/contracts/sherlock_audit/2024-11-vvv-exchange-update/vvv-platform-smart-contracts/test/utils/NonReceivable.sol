// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

//This contract can receive ether on creation, but not afterwards
//Created for testing failed transfers

contract NonReceivable {
    constructor() payable {}

    receive() external payable {
        revert("Cannot receive ETH");
    }

    fallback() external payable {
        revert("Cannot receive ETH");
    }
}
