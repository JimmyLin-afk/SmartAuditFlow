// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;

import "openzeppelin-contracts-next/contracts/token/ERC20/ERC20.sol";

contract TestERC20 is ERC20 {
  constructor(string memory name, string memory symbol) ERC20(name, symbol) {}

  function mint(address to, uint256 amount) external returns (bool) {
    _mint(to, amount);
    return true;
  }

  function burn(uint256 amount) public returns (bool) {
    _burn(msg.sender, amount);
    return true;
  }
}
