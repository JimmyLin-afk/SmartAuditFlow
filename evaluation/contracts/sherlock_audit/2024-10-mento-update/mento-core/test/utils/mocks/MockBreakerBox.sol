// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

contract MockBreakerBox {
  uint256 public tradingMode;

  function setTradingMode(uint256 _tradingMode) external {
    tradingMode = _tradingMode;
  }

  function getBreakers() external pure returns (address[] memory) {
    return new address[](0);
  }

  function isBreaker(address) external pure returns (bool) {
    return true;
  }

  function getRateFeedTradingMode(address) external pure returns (uint8) {
    return 0;
  }

  function checkAndSetBreakers(address) external {}
}
