// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;

import { Test } from "mento-std/Test.sol";

contract GovernanceTest is Test {
  address public owner = makeAddr("owner");
  address public alice = makeAddr("alice");
  address public bob = makeAddr("bob");
  address public charlie = makeAddr("charlie");

  uint256 public constant INITIAL_TOTAL_SUPPLY = 350_000_000 * 1e18;
  uint256 public constant EMISSION_SUPPLY = 650_000_000 * 1e18;

  uint256 public constant MONTH = 30 days;
  uint256 public constant YEAR = 365 days;

  uint256 public constant BLOCKS_DAY = 17_280; // in CELO
  uint256 public constant BLOCKS_WEEK = 120_960; // in CELO
}
