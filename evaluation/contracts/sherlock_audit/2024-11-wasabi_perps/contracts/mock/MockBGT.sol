// SPDX-License-Identifier: BUSL-1.1
pragma solidity 0.8.26;

import "@berachain/pol-contracts/src/pol/BGT.sol";

contract MockBGT is BGT {
    receive() external payable {}
}