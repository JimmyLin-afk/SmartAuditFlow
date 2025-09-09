// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

function min(uint256 a, uint256 b) pure returns (uint256) {
  return a > b ? b : a;
}

function min(uint256 a, uint256 b, uint256 c) pure returns (uint256) {
  return min(a, min(b, c));
}

function min(int48 a, int48 b) pure returns (int48) {
  return a > b ? b : a;
}

function min(int48 a, int48 b, int48 c) pure returns (int48) {
  return min(a, min(b, c));
}

uint8 constant L0 = 1; // 0b001 Limit0
uint8 constant L1 = 2; // 0b010 Limit1
uint8 constant LG = 4; // 0b100 LimitGlobal

function toRateFeed(string memory rateFeed) pure returns (address) {
  return address(uint160(uint256(keccak256(abi.encodePacked(rateFeed)))));
}
