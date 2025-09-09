// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";

//TESTING ONLY
contract TestAggregator {
    ERC20 public _telcoin;

    constructor(ERC20 telcoin_) {
        _telcoin = telcoin_;
    }

    function getSwapSelector() external pure returns (bytes4) {
        return this.swap.selector;
    }

    function getPOLSwapSelector() external pure returns (bytes4) {
        return this.POLSwap.selector;
    }

    function swap() external {
        _telcoin.transfer(msg.sender, 10);
    }

    function POLSwap() external payable {
        _telcoin.transfer(msg.sender, 10);
    }
}
