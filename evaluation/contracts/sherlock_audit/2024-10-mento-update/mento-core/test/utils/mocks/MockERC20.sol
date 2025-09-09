// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

contract MockERC20 {
  string private _name;
  string private _symbol;
  uint256 private _decimals;

  constructor(string memory name_, string memory symbol_, uint256 decimals_) {
    _name = name_;
    _symbol = symbol_;
    _decimals = decimals_;
  }

  function name() public view returns (string memory) {
    return _name;
  }

  function symbol() public view returns (string memory) {
    return _symbol;
  }

  function decimals() public view returns (uint256) {
    return _decimals;
  }
}
