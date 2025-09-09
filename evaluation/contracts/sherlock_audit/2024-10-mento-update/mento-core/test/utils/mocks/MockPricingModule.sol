// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

import { IPricingModule } from "contracts/interfaces/IPricingModule.sol";

contract MockPricingModule is IPricingModule {
  string private _name;
  uint256 private _nextGetAmountOut;
  uint256 private _nextGetAmountIn;

  constructor(string memory name_) {
    _name = name_;
  }

  function name() external view returns (string memory) {
    return _name;
  }

  function mockNextGetAmountOut(uint256 amount) external {
    _nextGetAmountOut = amount;
  }

  function mockNextGetAmountIn(uint256 amount) external {
    _nextGetAmountIn = amount;
  }

  function getAmountOut(uint256, uint256, uint256, uint256) external view returns (uint256) {
    return _nextGetAmountOut;
  }

  function getAmountIn(uint256, uint256, uint256, uint256) external view returns (uint256) {
    return _nextGetAmountIn;
  }
}
