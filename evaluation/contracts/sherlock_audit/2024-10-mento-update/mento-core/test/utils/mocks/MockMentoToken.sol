// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8.0;

import { ERC20 } from "openzeppelin-contracts-next/contracts/token/ERC20/ERC20.sol";

contract MockMentoToken is ERC20 {
  constructor() ERC20("Mock Mento Token", "MENTO") {}

  function mint(address to, uint256 amount) external {
    _mint(to, amount);
  }
}
