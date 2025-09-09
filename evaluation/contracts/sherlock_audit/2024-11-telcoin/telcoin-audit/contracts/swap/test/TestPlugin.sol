// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";

//TESTING ONLY
contract TestPlugin {
    ERC20 public _telcoin;

    constructor(ERC20 telcoin_) {
        _telcoin = telcoin_;
    }

    function increaseClaimableBy(
        address,
        uint256 amount
    ) external returns (bool) {
        _telcoin.transferFrom(msg.sender, address(this), amount);
        return true;
    }
}
