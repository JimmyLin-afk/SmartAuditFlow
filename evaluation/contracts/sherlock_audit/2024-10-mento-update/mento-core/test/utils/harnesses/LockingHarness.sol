// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

import { Locking } from "contracts/governance/locking/Locking.sol";

contract LockingHarness is Locking {
  uint32 public blockNumberMocked;
  uint32 public epochShift;

  function incrementBlock(uint32 _amount) external {
    blockNumberMocked = blockNumberMocked + _amount;
  }

  function getBlockNumber() internal view override returns (uint32) {
    return blockNumberMocked;
  }

  function getMaxSlopePeriod() public pure returns (uint32) {
    return MAX_SLOPE_PERIOD;
  }

  function getMaxCliffPeriod() public pure returns (uint32) {
    return MAX_CLIFF_PERIOD;
  }

  function getLockTest(
    uint96 amount,
    uint32 slope,
    uint32 cliff
  ) external view returns (uint96 lockAmount, uint96 lockSlope) {
    (lockAmount, lockSlope) = getLock(amount, slope, cliff);
  }

  function getEpochShift() internal view override returns (uint32) {
    return epochShift;
  }

  function setEpochShift(uint32 _epochShift) external {
    epochShift = _epochShift;
  }

  function setBlock(uint32 _block) external {
    blockNumberMocked = _block;
  }

  function blockTillNextPeriod() external view returns (uint256) {
    uint256 currentWeek = this.getWeek();
    return (WEEK * (currentWeek + 1)) + getEpochShift() - getBlockNumber();
  }
}
