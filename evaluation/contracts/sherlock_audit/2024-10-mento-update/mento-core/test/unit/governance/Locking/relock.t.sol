// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;
// solhint-disable func-name-mixedcase, contract-name-camelcase

import { LockingTest } from "./LockingTest.sol";

contract Relock_LockingTest is LockingTest {
  uint256 public lockId;

  function test_relock_whenDelegateZero_shouldRevert() public {
    mentoToken.mint(alice, 100e18);

    vm.prank(alice);
    lockId = locking.lock(alice, alice, 30e18, 3, 3);

    _incrementBlock(2 * weekInBlocks);

    vm.expectRevert("delegate is zero");
    vm.prank(alice);
    locking.relock(lockId, address(0), 30e18, 3, 3);
  }

  function test_relock_whenInCliff_shouldRelockWithIncreasedAmount() public {
    mentoToken.mint(alice, 100e18);

    vm.prank(alice);
    lockId = locking.lock(alice, alice, 30e18, 3, 3);

    _incrementBlock(2 * weekInBlocks);

    vm.prank(alice);
    locking.relock(lockId, alice, 45e18, 9, 0);

    assertEq(mentoToken.balanceOf(address(locking)), 45e18);
    assertEq(mentoToken.balanceOf(alice), 55e18);

    _incrementBlock(9 * weekInBlocks);
    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 0);
    assertEq(mentoToken.balanceOf(alice), 100e18);
  }

  function test_relock_whenInCliff_shouldRelockWithNewCliff() public {
    mentoToken.mint(alice, 100e18);

    vm.prank(alice);
    lockId = locking.lock(alice, alice, 30e18, 3, 3);

    _incrementBlock(2 * weekInBlocks);

    vm.prank(alice);
    locking.relock(lockId, alice, 30e18, 3, 4);

    assertEq(mentoToken.balanceOf(address(locking)), 30e18);
    assertEq(mentoToken.balanceOf(alice), 70e18);

    _incrementBlock(7 * weekInBlocks);
    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 0);
    assertEq(mentoToken.balanceOf(alice), 100e18);
  }

  function test_relock_whenInCliff_sholdRelockWithNewSlope() public {
    mentoToken.mint(alice, 100e18);

    vm.prank(alice);
    lockId = locking.lock(alice, alice, 30e18, 3, 3);

    _incrementBlock(2 * weekInBlocks);

    vm.prank(alice);
    locking.relock(lockId, alice, 35e18, 18, 0);

    assertEq(mentoToken.balanceOf(address(locking)), 35e18);
    assertEq(mentoToken.balanceOf(alice), 65e18);

    _incrementBlock(18 * weekInBlocks);
    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 0);
    assertEq(mentoToken.balanceOf(alice), 100e18);
  }

  function test_relock_whenInCliff_shouldRelockWithANewSchedle() public {
    mentoToken.mint(alice, 100e18);

    vm.prank(alice);
    lockId = locking.lock(alice, alice, 30e18, 3, 3);

    _incrementBlock(2 * weekInBlocks);

    vm.prank(alice);
    locking.relock(lockId, alice, 80e18, 16, 6);

    assertEq(mentoToken.balanceOf(address(locking)), 80e18);
    assertEq(mentoToken.balanceOf(alice), 20e18);

    _incrementBlock(6 * weekInBlocks);
    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 80e18);
    assertEq(mentoToken.balanceOf(alice), 20e18);

    _incrementBlock(16 * weekInBlocks);
    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 0);
    assertEq(mentoToken.balanceOf(alice), 100e18);
  }

  function test_relock_wheInSlope_shouldRelockWithNewSchedule() public {
    mentoToken.mint(alice, 1000e18);

    vm.startPrank(alice);
    lockId = locking.lock(alice, alice, 300e18, 3, 0);
    _incrementBlock(2 * weekInBlocks);

    locking.relock(lockId, alice, 800e18, 16, 6);

    assertEq(mentoToken.balanceOf(address(locking)), 800e18);
    assertEq(mentoToken.balanceOf(alice), 200e18);

    _incrementBlock(6 * weekInBlocks);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 800e18);
    assertEq(mentoToken.balanceOf(alice), 200e18);

    _incrementBlock(16 * weekInBlocks);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 0);
    assertEq(mentoToken.balanceOf(alice), 1000e18);

    vm.stopPrank();
  }

  function test_relock_whenRelocked_shouldReleaseByTheNewScheduleAfterWithdraw() public {
    mentoToken.mint(alice, 1000e18);

    vm.startPrank(alice);
    lockId = locking.lock(alice, alice, 300e18, 3, 0);
    _incrementBlock(2 * weekInBlocks);

    locking.relock(lockId, alice, 800e18, 16, 6);

    assertEq(mentoToken.balanceOf(address(locking)), 800e18);
    assertEq(mentoToken.balanceOf(alice), 200e18);

    _incrementBlock(6 * weekInBlocks);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 800e18);
    assertEq(mentoToken.balanceOf(alice), 200e18);

    _incrementBlock(16 * weekInBlocks);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 0);
    assertEq(mentoToken.balanceOf(alice), 1000e18);

    vm.stopPrank();
  }

  function test_relock_whenInTail_shouldRelockWithNewSchedule() public {
    mentoToken.mint(alice, 1000e18);

    vm.startPrank(alice);
    lockId = locking.lock(alice, alice, 370e18, 4, 3);

    _incrementBlock(6 * weekInBlocks);

    locking.relock(lockId, alice, 100e18, 2, 2);

    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 100e18);
    assertEq(mentoToken.balanceOf(alice), 900e18);

    _incrementBlock(2 * weekInBlocks);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 100e18);
    assertEq(mentoToken.balanceOf(alice), 900e18);

    _incrementBlock(2 * weekInBlocks);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 0);
    assertEq(mentoToken.balanceOf(alice), 1000e18);

    vm.stopPrank();
  }

  function test_relock_whenAmountLessThanCurrent_shouldRevert() public {
    mentoToken.mint(alice, 100e18);

    vm.prank(alice);
    lockId = locking.lock(alice, alice, 38e18, 4, 3);
    _incrementBlock(6 * weekInBlocks);

    vm.expectRevert("Impossible to relock: less amount, then now is");
    vm.prank(alice);
    locking.relock(lockId, alice, 5e18, 1, 1);
  }

  function test_relock_whenPeriodTooShort_shouldRevert() public {
    mentoToken.mint(alice, 100e18);

    vm.prank(alice);
    lockId = locking.lock(alice, alice, 38e18, 4, 3);
    _incrementBlock(2 * weekInBlocks);

    vm.expectRevert("new line period lock too short");
    vm.prank(alice);
    locking.relock(lockId, alice, 5e18, 1, 1);
  }

  function test_relock_whenAmountIsZero_shouldRevert() public {
    mentoToken.mint(alice, 100e18);

    vm.prank(alice);
    lockId = locking.lock(alice, alice, 38e18, 4, 3);
    _incrementBlock(2 * weekInBlocks);

    vm.expectRevert("zero amount");
    vm.prank(alice);
    locking.relock(lockId, alice, 0, 5, 1);
  }

  function test_relock_whenSlopeIsZero_shouldRevert() public {
    mentoToken.mint(alice, 100e18);

    vm.prank(alice);
    lockId = locking.lock(alice, alice, 38e18, 4, 3);
    _incrementBlock(2 * weekInBlocks);

    vm.expectRevert("slope period equal 0");
    vm.prank(alice);
    locking.relock(lockId, alice, 60e18, 0, 2);
  }

  function test_relock_whenCliffIsBig_shouldRevert() public {
    mentoToken.mint(alice, 100e18);

    vm.prank(alice);
    lockId = locking.lock(alice, alice, 38e18, 4, 3);
    _incrementBlock(2 * weekInBlocks);

    vm.expectRevert("cliff too big");
    vm.prank(alice);
    locking.relock(lockId, alice, 60e18, 12, 105);
  }

  function test_relock_whenSlopeIsBig_shouldRevert() public {
    mentoToken.mint(alice, 100e18);

    vm.prank(alice);
    lockId = locking.lock(alice, alice, 38e18, 4, 3);
    _incrementBlock(2 * weekInBlocks);

    vm.expectRevert("slope period too big");
    vm.prank(alice);
    locking.relock(lockId, alice, 60e18, 210, 10);
  }

  function test_delegate_withoutRelock() public {
    mentoToken.mint(alice, 100000e18);

    vm.prank(alice);
    lockId = locking.lock(alice, bob, 60000e18, 30, 0);
    // 60000 * 30 / 104 = 17307
    assertApproxEqAbs(locking.balanceOf(bob), 17307e18, 1e18);
    assertEq(mentoToken.balanceOf(address(locking)), 60000e18);
    assertEq(mentoToken.balanceOf(alice), 40000e18);

    _incrementBlock(29 * weekInBlocks);
    // 60000e18 * (30 / 104 ) - 29* ((60000e18 * (30 / 104) - 1) / 30 + 1) = 577e18
    assertApproxEqAbs(locking.balanceOf(bob), 577e18, 1e18);

    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 2000e18);
    assertEq(mentoToken.balanceOf(alice), 98000e18);

    _incrementBlock(weekInBlocks);

    vm.prank(alice);
    locking.withdraw();

    assertEq(locking.balanceOf(bob), 0);
    assertEq(mentoToken.balanceOf(address(locking)), 0);
    assertEq(mentoToken.balanceOf(alice), 100000e18);
  }

  function test_relock_shouldAccountDelegatedAmountCorrectly() public {
    mentoToken.mint(alice, 100000e18);

    vm.prank(alice);
    lockId = locking.lock(alice, bob, 60000e18, 30, 0);

    // 60000 * 30 / 104 = 17307
    assertApproxEqAbs(locking.balanceOf(bob), 17307e18, 1e18);

    _incrementBlock(20 * weekInBlocks);
    // 60000e18 * (30 / 104 ) - 20 * ((60000e18 * (30 / 104) - 1) / 30 + 1)= 5769
    assertApproxEqAbs(locking.balanceOf(bob), 5769e18, 1e18);

    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 20000e18);
    assertEq(mentoToken.balanceOf(alice), 80000e18);

    vm.prank(alice);
    locking.relock(lockId, charlie, 20000e18, 10, 0);

    assertEq(locking.balanceOf(bob), 0);
    // 20000 * 10 / 104 = 1923
    assertApproxEqAbs(locking.balanceOf(charlie), 1923e18, 1e18);

    _incrementBlock(10 * weekInBlocks);

    assertEq(locking.balanceOf(bob), 0);
    assertEq(locking.balanceOf(charlie), 0);

    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 0);
    assertEq(mentoToken.balanceOf(alice), 100000e18);
  }

  function test_relock_whenUnknownLock_shouldRevert() public {
    mentoToken.mint(alice, 100000e18);

    vm.prank(alice);
    lockId = locking.lock(alice, bob, 60000e18, 30, 0);

    _incrementBlock(20 * weekInBlocks);
    // 60000e18 * (30 / 104 ) - 20 * ((60000e18 * (30 / 104) - 1) / 30 + 1)= 5769
    assertApproxEqAbs(locking.balanceOf(bob), 5769e18, 1e18);

    vm.prank(alice);
    locking.withdraw();

    lockId = 1337; // does not exist

    vm.expectRevert("caller not a lock owner");
    vm.prank(alice);
    locking.relock(lockId, charlie, 20000e18, 10, 0);
  }

  function test_relock_whenMultiLinesInCliff_shouldRelockCorrectLine() public {
    mentoToken.mint(alice, 100e18);

    vm.prank(alice);
    locking.lock(alice, alice, 30e18, 3, 3);

    vm.prank(alice);
    uint256 lockId2 = locking.lock(alice, alice, 50e18, 5, 3);

    _incrementBlock(2 * weekInBlocks);

    vm.prank(alice);
    locking.relock(lockId2, alice, 60e18, 6, 0);

    assertEq(mentoToken.balanceOf(address(locking)), 90e18);
    assertEq(mentoToken.balanceOf(alice), 10e18);

    _incrementBlock(weekInBlocks);
    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 80e18);
    assertEq(mentoToken.balanceOf(alice), 20e18);

    _incrementBlock(5 * weekInBlocks);
    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 0);
    assertEq(mentoToken.balanceOf(alice), 100e18);
  }

  function test_relock_whenMultiLinesInCliff_shouldRelockCorrectLine2() public {
    mentoToken.mint(alice, 100e18);

    vm.prank(alice);
    uint256 lockId1 = locking.lock(alice, alice, 30e18, 3, 3);

    vm.prank(alice);
    locking.lock(alice, alice, 50e18, 5, 3);

    _incrementBlock(2 * weekInBlocks);

    vm.prank(alice);
    locking.relock(lockId1, alice, 50e18, 10, 0);

    assertEq(mentoToken.balanceOf(address(locking)), 100e18);
    assertEq(mentoToken.balanceOf(alice), 0);

    _incrementBlock(weekInBlocks);
    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 95e18);
    assertEq(mentoToken.balanceOf(alice), 5e18);

    _incrementBlock(9 * weekInBlocks);
    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 0);
    assertEq(mentoToken.balanceOf(alice), 100e18);
  }

  function test_relock_whenMultiLinesCutCorner_shouldRevert() public {
    mentoToken.mint(alice, 100e18);

    vm.prank(alice);
    locking.lock(alice, alice, 30e18, 3, 3);

    vm.prank(alice);
    uint256 lockId2 = locking.lock(alice, alice, 50e18, 5, 3);

    _incrementBlock(2 * weekInBlocks);

    vm.prank(alice);
    vm.expectRevert("detect cut deposit corner");
    locking.relock(lockId2, alice, 50e18, 10, 0);
  }

  function test_relock_whenMultiLinesInSlope_shouldRelockCorrectly() public {
    mentoToken.mint(alice, 1000e18);

    vm.prank(alice);
    locking.lock(alice, alice, 300e18, 3, 0);

    vm.prank(alice);
    uint256 lockId2 = locking.lock(alice, alice, 300e18, 3, 0);

    _incrementBlock(2 * weekInBlocks);

    vm.prank(alice);
    locking.relock(lockId2, alice, 600e18, 12, 0);

    assertEq(mentoToken.balanceOf(address(locking)), 700e18);
    assertEq(mentoToken.balanceOf(alice), 300e18);

    _incrementBlock(weekInBlocks);
    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 550e18);
    assertEq(mentoToken.balanceOf(alice), 450e18);

    _incrementBlock(11 * weekInBlocks);
    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 0);
    assertEq(mentoToken.balanceOf(alice), 1000e18);
  }

  function test_relock_whenMultiAccountsInCliff_shouldRelockCorrectly() public {
    mentoToken.mint(alice, 100e18);
    mentoToken.mint(bob, 100e18);
    mentoToken.mint(charlie, 100e18);

    vm.prank(alice);
    mentoToken.approve(address(locking), type(uint256).max);
    vm.prank(bob);
    mentoToken.approve(address(locking), type(uint256).max);
    vm.prank(charlie);
    mentoToken.approve(address(locking), type(uint256).max);

    vm.prank(alice);
    locking.lock(alice, alice, 20e18, 4, 2);

    vm.prank(bob);
    uint256 lockId2 = locking.lock(bob, bob, 30e18, 3, 3);

    vm.prank(charlie);
    locking.lock(charlie, charlie, 40e18, 4, 4);

    _incrementBlock(4 * weekInBlocks);

    vm.prank(bob);
    locking.relock(lockId2, bob, 30e18, 6, 1);

    vm.prank(alice);
    locking.withdraw();
    vm.prank(bob);
    locking.withdraw();
    vm.prank(charlie);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 80e18);
    assertEq(mentoToken.balanceOf(alice), 90e18);
    assertEq(mentoToken.balanceOf(bob), 70e18);
    assertEq(mentoToken.balanceOf(charlie), 60e18);

    _incrementBlock(weekInBlocks);

    vm.prank(alice);
    locking.withdraw();
    vm.prank(bob);
    locking.withdraw();
    vm.prank(charlie);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 65e18);
    assertEq(mentoToken.balanceOf(alice), 95e18);
    assertEq(mentoToken.balanceOf(bob), 70e18);
    assertEq(mentoToken.balanceOf(charlie), 70e18);

    _incrementBlock(weekInBlocks);

    vm.prank(alice);
    locking.withdraw();
    vm.prank(bob);
    locking.withdraw();
    vm.prank(charlie);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 45e18);
    assertEq(mentoToken.balanceOf(alice), 100e18);
    assertEq(mentoToken.balanceOf(bob), 75e18);
    assertEq(mentoToken.balanceOf(charlie), 80e18);

    _incrementBlock(2 * weekInBlocks);

    vm.prank(alice);
    locking.withdraw();
    vm.prank(bob);
    locking.withdraw();
    vm.prank(charlie);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 15e18);
    assertEq(mentoToken.balanceOf(alice), 100e18);
    assertEq(mentoToken.balanceOf(bob), 85e18);
    assertEq(mentoToken.balanceOf(charlie), 100e18);

    _incrementBlock(3 * weekInBlocks);

    vm.prank(alice);
    locking.withdraw();
    vm.prank(bob);
    locking.withdraw();
    vm.prank(charlie);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 0);
    assertEq(mentoToken.balanceOf(alice), 100e18);
    assertEq(mentoToken.balanceOf(bob), 100e18);
    assertEq(mentoToken.balanceOf(charlie), 100e18);
  }

  function test_relock_whenMultiAccountsInTail_shouldRelockCorrectly() public {
    mentoToken.mint(alice, 100e18);
    mentoToken.mint(bob, 100e18);
    mentoToken.mint(charlie, 100e18);

    vm.prank(alice);
    mentoToken.approve(address(locking), type(uint256).max);
    vm.prank(bob);
    mentoToken.approve(address(locking), type(uint256).max);
    vm.prank(charlie);
    mentoToken.approve(address(locking), type(uint256).max);

    vm.prank(alice);
    locking.lock(alice, alice, 20e18, 4, 2);

    vm.prank(bob);
    uint256 lockId2 = locking.lock(bob, bob, 32e18, 4, 3);

    vm.prank(charlie);
    locking.lock(charlie, charlie, 40e18, 4, 4);

    _incrementBlock(6 * weekInBlocks);

    vm.prank(bob);
    locking.relock(lockId2, bob, 22e18, 5, 1);

    vm.prank(alice);
    locking.withdraw();
    vm.prank(bob);
    locking.withdraw();
    vm.prank(charlie);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 42e18);
    assertEq(mentoToken.balanceOf(alice), 100e18);
    assertEq(mentoToken.balanceOf(bob), 78e18);
    assertEq(mentoToken.balanceOf(charlie), 80e18);

    _incrementBlock(weekInBlocks);

    vm.prank(alice);
    locking.withdraw();
    vm.prank(bob);
    locking.withdraw();
    vm.prank(charlie);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 32e18);
    assertEq(mentoToken.balanceOf(alice), 100e18);
    assertEq(mentoToken.balanceOf(bob), 78e18);
    assertEq(mentoToken.balanceOf(charlie), 90e18);

    _incrementBlock(weekInBlocks);

    vm.prank(alice);
    locking.withdraw();
    vm.prank(bob);
    locking.withdraw();
    vm.prank(charlie);
    locking.withdraw();

    assertApproxEqAbs(mentoToken.balanceOf(address(locking)), 17e18, 1e18);
    assertEq(mentoToken.balanceOf(alice), 100e18);
    assertApproxEqAbs(mentoToken.balanceOf(bob), 83e18, 1e18);
    assertEq(mentoToken.balanceOf(charlie), 100e18);

    _incrementBlock(4 * weekInBlocks);

    vm.prank(alice);
    locking.withdraw();
    vm.prank(bob);
    locking.withdraw();
    vm.prank(charlie);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 0);
    assertEq(mentoToken.balanceOf(alice), 100e18);
    assertEq(mentoToken.balanceOf(bob), 100e18);
    assertEq(mentoToken.balanceOf(charlie), 100e18);
  }
}
