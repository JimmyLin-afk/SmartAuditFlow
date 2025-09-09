// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

import {Initializable} from "@openzeppelin/contracts/proxy/utils/Initializable.sol";

//TESTING ONLY
contract TestWallet is Initializable {
    function initialize() external initializer {}

    function test() external {}

    function getTestSelector() external pure returns (bytes4) {
        return this.test.selector;
    }

    receive() external payable {}

    fallback() external {}
}
