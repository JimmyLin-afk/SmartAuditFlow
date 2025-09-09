// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "../bera/StakingAccount.sol";

contract MockStakingAccountV2 is StakingAccount {
    uint256 public constant MAGIC_VALUE = 1337;
}