// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;
// solhint-disable func-name-mixedcase, contract-name-camelcase

import { LockingTest } from "./LockingTest.sol";

contract DelegateTo_LockingTest is LockingTest {
  uint256 public lockId;

  function test_delegateTo_whenDelegateZero_shouldRevert() public {
    mentoToken.mint(alice, 100000e18);

    vm.prank(alice);
    lockId = locking.lock(alice, bob, 60000e18, 30, 0);

    _incrementBlock(20 * weekInBlocks);

    vm.prank(alice);
    vm.expectRevert("delegate is zero");
    locking.delegateTo(lockId, address(0));
  }

  function test_delegateTo_whenReDelegateToDifferentAccount_shouldDelegateCorrectly() public {
    mentoToken.mint(alice, 100000e18);

    vm.prank(alice);
    lockId = locking.lock(alice, bob, 60000e18, 30, 0);

    _incrementBlock(20 * weekInBlocks);
    // 60000e18 * (30 / 104) - 20 * ((60000e18 * (30 / 104) - 1) / 30 + 1)= 5769
    assertApproxEqAbs(locking.balanceOf(bob), 5769e18, 1e18);

    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 20000e18);
    assertEq(mentoToken.balanceOf(alice), 80000e18);

    vm.prank(alice);
    locking.delegateTo(lockId, charlie);

    assertEq(locking.balanceOf(bob), 0);
    assertApproxEqAbs(locking.balanceOf(charlie), 5769e18, 1e18);

    _incrementBlock(10 * weekInBlocks);

    assertEq(locking.balanceOf(charlie), 0);

    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 0);
    assertEq(mentoToken.balanceOf(alice), 100000e18);
  }

  function test_delegateTo_whenRedelegateToSameAccount_shouldDelegateCorrectly() public {
    mentoToken.mint(alice, 100000e18);

    vm.prank(alice);
    lockId = locking.lock(alice, bob, 60000e18, 30, 0);

    _incrementBlock(20 * weekInBlocks);

    vm.prank(alice);
    locking.withdraw();

    _incrementBlock(5 * weekInBlocks);

    vm.prank(alice);
    locking.delegateTo(lockId, bob);
    // 60000e18 * (30 / 104) - 25 * ((60000e18 * (30 / 104) - 1) / 30 + 1) = 2884
    assertApproxEqAbs(locking.balanceOf(bob), 2884e18, 1e18);
    assertEq(locking.balanceOf(charlie), 0);

    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 10000e18);
    assertEq(mentoToken.balanceOf(alice), 90000e18);

    _incrementBlock(5 * weekInBlocks);

    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 0);
    assertEq(mentoToken.balanceOf(alice), 100000e18);
    assertEq(locking.totalSupply(), 0);
  }

  function test_delegateTo_whenInTail_shouldReDelegateVotesToNewDelegate() public {
    mentoToken.mint(alice, 100000e18);

    vm.prank(alice);
    lockId = locking.lock(alice, bob, 6300e18, 7, 0);

    _incrementBlock(6 * weekInBlocks);
    //6300e18 * (7 / 104) - 6 * ((6300e18 * (7 / 104) - 1) / 7 + 1) = 60e18
    assertApproxEqAbs(locking.balanceOf(bob), 60e18, 1e18);

    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 900e18);
    assertEq(mentoToken.balanceOf(alice), 99100e18);

    vm.prank(alice);
    locking.delegateTo(lockId, charlie);

    assertEq(locking.balanceOf(bob), 0);
    assertApproxEqAbs(locking.balanceOf(charlie), 60e18, 1e18);

    _incrementBlock(weekInBlocks);

    assertEq(locking.balanceOf(charlie), 0);

    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 0);
    assertEq(mentoToken.balanceOf(alice), 100000e18);
  }

  function test_delegateTo_whenInCliff_shouldReDelegateVotes() public {
    mentoToken.mint(alice, 1000000e18);

    vm.prank(alice);
    lockId = locking.lock(alice, bob, 630000e18, 7, 2);

    _incrementBlock(weekInBlocks);
    // 630000e18 * (7 / 104 + 2 / 103) = 54636
    assertApproxEqAbs(locking.balanceOf(bob), 54636e18, 1e18);

    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 630000e18);
    assertEq(mentoToken.balanceOf(alice), 370000e18);

    vm.prank(alice);
    locking.delegateTo(lockId, charlie);

    assertEq(locking.balanceOf(bob), 0);
    assertApproxEqAbs(locking.balanceOf(charlie), 54636e18, 1e18);

    _incrementBlock(weekInBlocks);

    assertApproxEqAbs(locking.balanceOf(charlie), 54636e18, 1e18);

    _incrementBlock(7 * weekInBlocks);
    assertEq(locking.balanceOf(charlie), 0);

    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 0);
    assertEq(mentoToken.balanceOf(alice), 1000000e18);
  }

  function test_delegateTo_wheninSlope_shouldReDelegateVotes() public {
    mentoToken.mint(alice, 1000000e18);

    vm.prank(alice);
    lockId = locking.lock(alice, bob, 630000e18, 7, 2);

    _incrementBlock(4 * weekInBlocks);

    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 450000e18);
    assertEq(mentoToken.balanceOf(alice), 550000e18);

    vm.prank(alice);
    locking.delegateTo(lockId, charlie);

    // 630000e18 * (7 / 104 + 2 / 103) - 2 * ((630000e18 * (7 / 104 + 2 / 103) - 1) / 7 + 1) = 39026
    assertEq(locking.balanceOf(bob), 0);
    assertApproxEqAbs(locking.balanceOf(charlie), 39026e18, 1e18);

    _incrementBlock(5 * weekInBlocks);

    assertEq(locking.balanceOf(charlie), 0);

    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 0);
    assertEq(mentoToken.balanceOf(alice), 1000000e18);
  }

  function test_delegateTo_whenInTail_shouldReDelegateVotes() public {
    mentoToken.mint(alice, 1000000e18);

    vm.prank(alice);
    lockId = locking.lock(alice, bob, 630000e18, 7, 2);

    _incrementBlock(8 * weekInBlocks);

    vm.prank(alice);
    locking.withdraw();
    assertEq(mentoToken.balanceOf(address(locking)), 90000e18);
    assertEq(mentoToken.balanceOf(alice), 910000e18);

    vm.prank(alice);
    locking.delegateTo(lockId, charlie);

    assertEq(locking.balanceOf(bob), 0);
    // (630000e18 * (7 / 104 + 2 / 103)) - (((630000e18 * (7 / 104 + 2 / 103)) -1) / 7 + 1) * 6 = 7805e18
    assertApproxEqAbs(locking.balanceOf(charlie), 7805e18, 1e18);

    _incrementBlock(weekInBlocks);

    assertEq(locking.balanceOf(charlie), 0);

    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 0);
    assertEq(mentoToken.balanceOf(alice), 1000000e18);
  }

  function test_delegateTo_whenAfterFinishTime_shouldRevert() public {
    mentoToken.mint(alice, 1000000e18);

    vm.prank(alice);
    lockId = locking.lock(alice, bob, 630000e18, 7, 2);

    _incrementBlock(10 * weekInBlocks);

    vm.prank(alice);
    vm.expectRevert("Slope == 0, unacceptable value for slope");
    locking.delegateTo(lockId, charlie);
  }
}
