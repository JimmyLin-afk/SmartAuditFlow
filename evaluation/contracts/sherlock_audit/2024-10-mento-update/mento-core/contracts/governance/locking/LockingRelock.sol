// SPDX-License-Identifier: MIT
// solhint-disable gas-custom-errors
pragma solidity 0.8.18;

import "./LockingBase.sol";

/**
 * @notice https://github.com/rarible/locking-contracts/tree/4f189a96b3e85602dedfbaf69d9a1f5056d835eb
 */
abstract contract LockingRelock is LockingBase {
  using LibBrokenLine for LibBrokenLine.BrokenLine;

  /**
   * @notice Relocking tokens allows to changing lock parameters
   * @param id lock id of lock to relock
   * @param newDelegate new delegate address
   * @param newAmount new amount to lock
   * @param newSlopePeriod new slope period
   * @param newCliff new cliff period
   * @return counter new lock id
   */
  function relock(
    uint256 id,
    address newDelegate,
    uint96 newAmount,
    uint32 newSlopePeriod,
    uint32 newCliff
  ) external returns (uint256) {
    require(newDelegate != address(0), "delegate is zero");

    address account = verifyLockOwner(id);
    uint32 currentBlock = getBlockNumber();
    uint32 time = roundTimestamp(currentBlock);
    verification(account, id, newAmount, newSlopePeriod, newCliff, time);

    address _delegate = locks[id].delegate;
    accounts[account].locked.update(time);

    rebalance(id, account, accounts[account].locked.initial.bias, removeLines(id, account, _delegate, time), newAmount);

    counter++;

    addLines(account, newDelegate, newAmount, newSlopePeriod, newCliff, time, currentBlock);
    emit Relock(id, account, newDelegate, counter, time, newAmount, newSlopePeriod, newCliff);

    return counter;
  }

  /**
   * @notice Verifies parameters for relock
   * @dev Verification parameters:
   *      1. amount > 0, slope > 0
   *      2. cliff period and slope period less or equal two years
   *      3. newFinishTime more or equal oldFinishTime
   * @param account address of account that owns the old lock
   * @param id lock id of lock to relock
   * @param newAmount new amount to lock
   * @param newSlopePeriod new slope period
   * @param newCliff new cliff period
   * @param toTime current week number
   */
  function verification(
    address account,
    uint256 id,
    uint96 newAmount,
    uint32 newSlopePeriod,
    uint32 newCliff,
    uint32 toTime
  ) internal view {
    require(newAmount > 0, "zero amount");
    require(newCliff <= MAX_CLIFF_PERIOD, "cliff too big");
    require(newSlopePeriod <= MAX_SLOPE_PERIOD, "slope period too big");
    require(newSlopePeriod > 0, "slope period equal 0");

    //check Line with new parameters don`t finish earlier than old Line
    uint32 newEnd = toTime + (newCliff) + (newSlopePeriod);
    LibBrokenLine.Line memory line = accounts[account].locked.initiatedLines[id];
    uint32 oldSlopePeriod = uint32(divUp(line.bias, line.slope));
    uint32 oldEnd = line.start + (line.cliff) + (oldSlopePeriod);
    require(oldEnd <= newEnd, "new line period lock too short");

    //check Line with new parameters don`t cut corner old Line
    uint32 oldCliffEnd = line.start + (line.cliff);
    uint32 newCliffEnd = toTime + (newCliff);
    if (oldCliffEnd > newCliffEnd) {
      uint32 balance = oldCliffEnd - (newCliffEnd);
      uint32 newSlope = uint32(divUp(newAmount, newSlopePeriod));
      uint96 newBias = newAmount - (balance * (newSlope));
      require(newBias >= line.bias, "detect cut deposit corner");
    }
  }

  /**
   * @notice Removes a given lock from the lock owner, delegate and total supply
   * @param id lock id of lock to remove
   * @param account address of account that owns the lock
   * @param delegate address of delegate that owns the voting power
   * @return residue amount of tokens still locked in the old lock
   */
  function removeLines(uint256 id, address account, address delegate, uint32 toTime) internal returns (uint96 residue) {
    updateLines(account, delegate, toTime);
    uint32 currentBlock = getBlockNumber();
    // slither-disable-start unused-return
    accounts[delegate].balance.remove(id, toTime, currentBlock);
    totalSupplyLine.remove(id, toTime, currentBlock);
    (residue, , ) = accounts[account].locked.remove(id, toTime, currentBlock);
    // slither-disable-end unused-return
  }

  /**
   * @notice Rebalances additional tokens for the relock
   * @param id lock id of lock to relock
   * @param account address of account that owns the old lock
   * @param bias bias of the old lock
   * @param residue amount of tokens still locked in the old lock
   * @param newAmount new amount to lock
   */
  function rebalance(uint256 id, address account, uint96 bias, uint96 residue, uint96 newAmount) internal {
    require(residue <= newAmount, "Impossible to relock: less amount, then now is");
    uint96 addAmount = newAmount - (residue);
    uint96 amount = accounts[account].amount;
    uint96 balance = amount - (bias);
    if (addAmount > balance) {
      //need more, than balance, so need transfer tokens to this
      uint96 transferAmount = addAmount - (balance);
      accounts[account].amount = accounts[account].amount + (transferAmount);
      // slither-disable-start arbitrary-send-erc20
      // slither-disable-start reentrancy-events
      // slither-disable-start reentrancy-benign
      // slither-disable-next-line reentrancy-no-eth
      require(token.transferFrom(locks[id].account, address(this), transferAmount), "transfer failed");
      // slither-disable-end arbitrary-send-erc20
      // slither-disable-end reentrancy-events
      // slither-disable-end reentrancy-benign
    }
  }

  uint256[50] private __gap;
}
