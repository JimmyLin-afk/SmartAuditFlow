// SPDX-License-Identifier: UNLICENSED
pragma solidity 0.8.18;
// solhint-disable state-visibility

import { Test } from "mento-std/Test.sol";
import "openzeppelin-contracts-upgradeable/contracts/token/ERC20/IERC20Upgradeable.sol";

import { TestERC20 } from "test/utils/mocks/TestERC20.sol";
import { LockingHarness } from "test/utils/harnesses/LockingHarness.sol";

/**
 * @notice https://github.com/rarible/locking-contracts/tree/4f189a96b3e85602dedfbaf69d9a1f5056d835eb
 */
contract FuzzTestLocking is Test {
  TestERC20 public testERC20;

  LockingHarness locking;

  address user0;
  address user1;

  function setUp() public {
    user0 = address(100);
    vm.deal(user0, 100 ether);
    user1 = address(200);
    vm.deal(user1, 100 ether);
    testERC20 = new TestERC20("Test", "TST");

    locking = new LockingHarness();
    locking.__Locking_init(IERC20Upgradeable(address(testERC20)), 0, 1, 3);
    locking.incrementBlock(locking.WEEK() + 1);
  }

  function testFuzz_lockAmount(uint96 amount) public {
    vm.assume(amount < 2 ** 95);
    vm.assume(amount > 1e18);
    prepareTokens(user0, amount);
    lockTokens(user0, user0, uint96(amount), 100, 100);
  }

  function testFuzz_lockSlope(uint32 slope) public {
    vm.assume(slope >= locking.minSlopePeriod());
    vm.assume(slope <= locking.getMaxSlopePeriod());

    uint96 amount = 100 * (10 ** 18);

    prepareTokens(user0, amount);
    lockTokens(user0, user0, uint96(amount), slope, 100);
  }

  function testFuzz_lockCliff(uint32 cliff) public {
    vm.assume(cliff >= locking.minCliffPeriod());
    vm.assume(cliff <= locking.getMaxCliffPeriod());

    uint96 amount = 100 * (10 ** 18);

    prepareTokens(user0, amount);
    lockTokens(user0, user0, uint96(amount), 100, cliff);
  }

  function prepareTokens(address user, uint256 amount) public {
    testERC20.mint(user, amount);
    vm.prank(user);
    testERC20.approve(address(locking), amount);
    assertEq(testERC20.balanceOf(user), amount);
    assertEq(testERC20.allowance(user, address(locking)), amount);
  }

  function lockTokens(address user, address delegate, uint96 amount, uint32 slopePeriod, uint32 cliff) public {
    vm.prank(user);
    locking.lock(user, delegate, amount, slopePeriod, cliff);

    assertEq(locking.locked(user), amount);
  }
}
