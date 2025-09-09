// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;

import "./TestERC20.sol";

contract USDC is TestERC20 {
  constructor(string memory name, string memory symbol) TestERC20(name, symbol) {}

  function decimals() public pure override returns (uint8) {
    return 6;
  }
}
