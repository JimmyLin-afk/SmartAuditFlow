// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;
// solhint-disable func-name-mixedcase, contract-name-camelcase

import { LockingTest } from "./LockingTest.sol";

contract Lock_LockingTest is LockingTest {
  function test_init_shouldSetState() public view {
    assertEq(address(locking.token()), address(mentoToken));

    assertEq(locking.startingPointWeek(), 0);
    assertEq(locking.minCliffPeriod(), 0);
    assertEq(locking.minSlopePeriod(), 0);
    assertEq(locking.owner(), owner);
  }

  function test_lock_whenSlopeIsLarge_shouldRevert() public {
    mentoToken.mint(alice, 1500e18);

    vm.expectRevert("period too big");
    vm.prank(alice);
    locking.lock(alice, alice, 1000e18, 105, 0);
  }

  function test_lock_whenCliffeIsLarge_shouldRevert() public {
    mentoToken.mint(alice, 1500e18);

    vm.expectRevert("cliff too big");
    vm.prank(alice);
    locking.lock(alice, alice, 1000e18, 11, 105);
  }

  function test_lock_whenAmountIsZero_shouldRevert() public {
    mentoToken.mint(alice, 1500e18);

    vm.expectRevert("amount is less than minimum");
    vm.prank(alice);
    locking.lock(alice, alice, 0, 10, 10);
  }

  function test_lock_whenSlopeIsZero_shouldRevert() public {
    mentoToken.mint(alice, 1500e18);

    vm.expectRevert();
    vm.prank(alice);
    locking.lock(alice, alice, 10000e18, 0, 10);
  }

  function test_lock_whenAccountZero_shouldRevert() public {
    mentoToken.mint(alice, 1500e18);

    vm.expectRevert("account is zero");
    vm.prank(alice);
    locking.lock(address(0), alice, 20e18, 40, 0);
  }

  function test_lock_whenDelegateZero_shouldRevert() public {
    mentoToken.mint(alice, 1500e18);

    vm.expectRevert("delegate is zero");
    vm.prank(alice);
    locking.lock(alice, address(0), 20e18, 40, 0);
  }

  function test_withdraw_whenInSlope_shouldReleaseCorrectAmount() public {
    mentoToken.mint(alice, 1000e18);

    vm.prank(alice);
    locking.lock(alice, alice, 300e18, 3, 3);

    _incrementBlock(3 * weekInBlocks);

    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 300e18);
    assertEq(mentoToken.balanceOf(alice), 700e18);

    _incrementBlock(weekInBlocks);

    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 200e18);
    assertEq(mentoToken.balanceOf(alice), 800e18);
  }

  function test_withdraw_whenCalledFromAnotherAccount_shouldNotSendTokens() public {
    mentoToken.mint(alice, 1000e18);

    vm.prank(alice);
    locking.lock(alice, alice, 300e18, 3, 0);

    _incrementBlock(weekInBlocks);

    vm.prank(bob);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 300e18);
    assertEq(mentoToken.balanceOf(alice), 700e18);
  }

  function test_withdraw_whenTheLockIsCreatedForSomeoneElse_shouldNotReleaseTokens() public {
    mentoToken.mint(alice, 1000e18);

    vm.prank(alice);
    locking.lock(bob, bob, 300e18, 3, 0);

    _incrementBlock(weekInBlocks);

    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 300e18);
    assertEq(mentoToken.balanceOf(alice), 700e18);
  }

  function test_withdraw_whenCalledByTheOwnerOfTheLock_shouldReleaseTokens() public {
    mentoToken.mint(alice, 1000e18);

    vm.prank(alice);
    locking.lock(bob, bob, 300e18, 3, 0);

    _incrementBlock(weekInBlocks);

    vm.prank(bob);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 200e18);
    assertEq(mentoToken.balanceOf(alice), 700e18);
    assertEq(mentoToken.balanceOf(bob), 100e18);
  }

  function test_withdraw_whenTailInVeToken_shouldReleaseCorrectAmounts() public {
    mentoToken.mint(alice, 6000e18);

    vm.prank(alice);
    locking.lock(alice, alice, 5200e18, 52, 53);

    assertEq(mentoToken.balanceOf(address(locking)), 5200e18);
    assertEq(mentoToken.balanceOf(alice), 800e18);
    // (52 / 104 + 53 / 103) * 5200 = 5275 > 5200
    assertEq(locking.balanceOf(alice), 5200e18);

    _incrementBlock(103 * weekInBlocks);

    vm.prank(alice);
    locking.withdraw();

    assertEq(locking.balanceOf(alice), 200e18);

    _incrementBlock(weekInBlocks);

    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 100e18);
    assertEq(mentoToken.balanceOf(alice), 5900e18);
    assertEq(locking.balanceOf(alice), 100e18);

    _incrementBlock(weekInBlocks);

    vm.prank(alice);
    locking.withdraw();

    assertEq(mentoToken.balanceOf(address(locking)), 0);
    assertEq(mentoToken.balanceOf(alice), 6000e18);
    assertEq(locking.balanceOf(alice), 0);
  }

  function test_getLock_shouldReturnCorrectValues() public view {
    uint96 amount = 60000e18;
    uint32 slopePeriod = 30;
    uint32 cliff = 30;
    (uint96 lockAmount, uint96 lockSlope) = locking.getLock(amount, slopePeriod, cliff);
    // floor ((30 / 104 + 30 / 103) * 60000) = 34783
    assertApproxEqAbs(lockAmount, 34783e18, 1e18);
    // divUp (34783, 30) = 1160
    assertApproxEqAbs(lockSlope, 1160e18, 1e18);

    amount = 96000e18;
    slopePeriod = 48;
    cliff = 48;
    (lockAmount, lockSlope) = locking.getLock(amount, slopePeriod, cliff);
    // floor ((48 / 104 + 48 / 103) * 96000) = 89045
    assertApproxEqAbs(lockAmount, 89045e18, 1e18);
    // divUp (89045, 48) = 1856
    assertApproxEqAbs(lockSlope, 1856e18, 1e18);

    amount = 104000e18;
    slopePeriod = 104;
    cliff = 0;
    (lockAmount, lockSlope) = locking.getLock(amount, slopePeriod, cliff);
    // floor ((104 / 104 + 0 / 103) * 104000) = 104000
    assertEq(lockAmount, 104000e18);
    // divUp (104000, 104) = 1000
    assertEq(lockSlope, 1000e18);

    amount = 104000e18;
    slopePeriod = 24;
    cliff = 103;
    (lockAmount, lockSlope) = locking.getLock(amount, slopePeriod, cliff);
    // floor ((24 / 104 + 103 / 103) * 104000) = 128000 > 104000
    assertEq(lockAmount, 104000e18);
    // divUp (104000, 24) = 4334
    assertApproxEqAbs(lockSlope, 4334e18, 1e18);
  }

  function test_getWeek_shouldReturnCorrectWeekNo() public {
    uint32 dayInBlocks = weekInBlocks / 7;
    uint32 currentBlock = 21664044; // (Sep-29-2023 11:59:59 AM +UTC) Friday
    locking.setBlock(currentBlock);

    // without shifting, it is week #179 with 12_204 blocks reminder
    assertEq(locking.getWeek(), 179);

    locking.setEpochShift(89_964);

    // since we shift more than remainder(89_964 > 12_204), we are now in the previous week, #178
    assertEq(locking.getWeek(), 178);

    // FRI 12:00 -> WED 00:00 = 4.5 days
    assertEq(locking.blockTillNextPeriod(), (9 * dayInBlocks) / 2);

    _incrementBlock(3 * dayInBlocks);
    assertEq(locking.getWeek(), 178);
    _incrementBlock(dayInBlocks);
    assertEq(locking.getWeek(), 178);
    _incrementBlock(dayInBlocks);
    assertEq(locking.getWeek(), 179);
  }

  function test_getAvailableForWithdraw_shouldReturnCorrectAmount() public {
    mentoToken.mint(alice, 1000e18);

    vm.prank(alice);
    locking.lock(alice, alice, 300e18, 3, 0);

    _incrementBlock(2 * weekInBlocks);

    vm.prank(alice);
    uint256 availableForWithdraw = locking.getAvailableForWithdraw(alice);

    assertEq(mentoToken.balanceOf(address(locking)), 300e18);
    assertEq(mentoToken.balanceOf(alice), 700e18);
    assertEq(availableForWithdraw, 200e18);
  }

  function test_viewFuctions_shouldReturnCorrectValues() public {
    uint256 lockId;
    uint256 lockBlock;
    uint256 voteBlock;
    uint256 delegateBlock;
    uint256 relockBlock;

    _incrementBlock(weekInBlocks + 1);
    mentoToken.mint(alice, 1000000e18);

    _incrementBlock(weekInBlocks / 2);

    lockBlock = locking.blockNumberMocked();

    vm.prank(alice);
    lockId = locking.lock(alice, alice, 3000e18, 3, 0);
    // WEEK 1
    assertEq(mentoToken.balanceOf(address(locking)), 3000e18);
    // 3 / 104 * 3000 = 86
    assertApproxEqAbs(locking.balanceOf(alice), 86e18, 1e18);
    assertApproxEqAbs(locking.getVotes(alice), 86e18, 1e18);

    // WEEK 2
    _incrementBlock(weekInBlocks / 2 + weekInBlocks / 4);

    delegateBlock = locking.blockNumberMocked();
    voteBlock = delegateBlock + weekInBlocks / 4;
    // 86 * 2 / 3 = 57
    assertApproxEqAbs(locking.balanceOf(alice), 57e18, 1e18);
    assertApproxEqAbs(locking.getVotes(alice), 57e18, 1e18);

    vm.prank(alice);
    locking.delegateTo(lockId, bob);

    assertEq(locking.balanceOf(alice), 0);
    assertEq(locking.getVotes(alice), 0);
    assertApproxEqAbs(locking.balanceOf(bob), 57e18, 1e18);
    assertApproxEqAbs(locking.getVotes(bob), 57e18, 1e18);

    assertApproxEqAbs(locking.getPastVotes(alice, delegateBlock - 1), 57e18, 1e18);
    assertEq(locking.getPastVotes(bob, delegateBlock - 1), 0);
    assertApproxEqAbs(locking.getPastTotalSupply(delegateBlock - 1), 57e18, 1e18);

    _incrementBlock(weekInBlocks / 2);

    relockBlock = locking.blockNumberMocked();

    assertEq(locking.balanceOf(alice), 0);
    assertEq(locking.getVotes(alice), 0);
    assertApproxEqAbs(locking.balanceOf(bob), 57e18, 1e18);
    assertApproxEqAbs(locking.getVotes(bob), 57e18, 1e18);

    assertEq(locking.getPastVotes(alice, voteBlock), 0);
    assertApproxEqAbs(locking.getPastVotes(bob, voteBlock), 57e18, 1e18);
    assertApproxEqAbs(locking.getPastTotalSupply(voteBlock), 57e18, 1e18);

    assertApproxEqAbs(locking.getPastVotes(alice, delegateBlock - 1), 57e18, 1e18);
    assertEq(locking.getPastVotes(bob, delegateBlock - 1), 0);
    assertApproxEqAbs(locking.getPastTotalSupply(delegateBlock - 1), 57e18, 1e18);

    vm.prank(alice);
    locking.relock(lockId, charlie, 4000e18, 4, 0);

    assertEq(mentoToken.balanceOf(address(locking)), 4000e18);
    assertEq(locking.balanceOf(alice), 0);
    assertEq(locking.getVotes(alice), 0);
    assertEq(locking.balanceOf(bob), 0);
    assertEq(locking.getVotes(bob), 0);
    // 4 / 104 * 4000 = 153
    assertApproxEqAbs(locking.balanceOf(charlie), 153e18, 1e18);
    assertApproxEqAbs(locking.getVotes(charlie), 153e18, 1e18);

    assertEq(locking.getPastVotes(alice, voteBlock), 0);
    assertApproxEqAbs(locking.getPastVotes(bob, voteBlock), 57e18, 1e18);
    assertApproxEqAbs(locking.getPastTotalSupply(voteBlock), 57e18, 1e18);

    assertApproxEqAbs(locking.getPastVotes(alice, delegateBlock - 1), 57e18, 1e18);
    assertEq(locking.getPastVotes(bob, delegateBlock - 1), 0);
    assertApproxEqAbs(locking.getPastTotalSupply(delegateBlock - 1), 57e18, 1e18);

    assertEq(locking.getPastVotes(alice, relockBlock - 1), 0);
    assertApproxEqAbs(locking.getPastVotes(bob, relockBlock - 1), 57e18, 1e18);
    assertApproxEqAbs(locking.getPastTotalSupply(relockBlock - 1), 57e18, 1e18);

    // WEEK 3

    _incrementBlock(weekInBlocks / 4);
    assertEq(locking.balanceOf(alice), 0);
    assertEq(locking.getVotes(alice), 0);
    assertEq(locking.balanceOf(bob), 0);
    assertEq(locking.getVotes(bob), 0);
    // 153 * 3 / 4 = 114
    assertApproxEqAbs(locking.balanceOf(charlie), 114e18, 2e18);
    assertApproxEqAbs(locking.getVotes(charlie), 114e18, 2e18);

    uint256 currentBlock = locking.blockNumberMocked();

    vm.expectRevert("block not yet mined");
    locking.getPastVotes(alice, currentBlock);

    vm.expectRevert("block not yet mined");
    locking.getPastTotalSupply(currentBlock);
  }

  function test_setMinCliffPeriod_setMinSlope_whenNotOwner_shouldRevert() public {
    vm.expectRevert("Ownable: caller is not the owner");
    locking.setMinCliffPeriod(3);

    vm.expectRevert("Ownable: caller is not the owner");
    locking.setMinSlopePeriod(3);
  }

  function test_setMinCliffPeriod_setMinSlope_whenExceedsMax_shouldRevert() public {
    vm.prank(owner);
    vm.expectRevert("new cliff period > 2 years");
    locking.setMinCliffPeriod(104);

    vm.prank(owner);
    vm.expectRevert("new slope period > 2 years");
    locking.setMinSlopePeriod(105);
  }

  function test_setMinCliffPeriod_shouldSetCliff() public {
    vm.prank(owner);
    locking.setMinCliffPeriod(5);
    assertEq(locking.minCliffPeriod(), 5);

    vm.prank(owner);
    locking.setMinCliffPeriod(103);
    assertEq(locking.minCliffPeriod(), 103);

    vm.prank(owner);
    locking.setMinCliffPeriod(0);
    assertEq(locking.minCliffPeriod(), 0);
  }

  function test_setMinSlopePeriod_shouldSetSlope() public {
    vm.prank(owner);
    locking.setMinSlopePeriod(5);
    assertEq(locking.minSlopePeriod(), 5);

    vm.prank(owner);
    locking.setMinSlopePeriod(104);
    assertEq(locking.minSlopePeriod(), 104);

    vm.prank(owner);
    locking.setMinSlopePeriod(1);
    assertEq(locking.minSlopePeriod(), 1);
  }
}
