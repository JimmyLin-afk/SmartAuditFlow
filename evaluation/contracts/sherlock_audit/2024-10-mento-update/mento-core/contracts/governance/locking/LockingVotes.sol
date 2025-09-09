// SPDX-License-Identifier: MIT
pragma solidity 0.8.18;
// solhint-disable no-unused-vars, gas-custom-errors

import "./LockingBase.sol";

/**
 * @notice https://github.com/rarible/locking-contracts/tree/4f189a96b3e85602dedfbaf69d9a1f5056d835eb
 */
contract LockingVotes is LockingBase {
  using LibBrokenLine for LibBrokenLine.BrokenLine;

  /**
   * @dev Returns the current amount of votes that `account` has.
   */
  function getVotes(address account) external view override returns (uint256) {
    uint32 currentBlock = getBlockNumber();
    uint32 currentWeek = roundTimestamp(currentBlock);
    return accounts[account].balance.actualValue(currentWeek, currentBlock);
  }

  /**
   * @dev Returns the amount of votes that `account` had
   * at the end of the last period
   */
  function getPastVotes(address account, uint256 blockNumber) external view override returns (uint256) {
    uint32 currentWeek = roundTimestamp(uint32(blockNumber));
    require(blockNumber < getBlockNumber() && currentWeek > 0, "block not yet mined");

    return accounts[account].balance.actualValue(currentWeek, uint32(blockNumber));
  }

  /**
   * @dev Returns the total supply of votes available
   * at the end of the last period
   */
  function getPastTotalSupply(uint256 blockNumber) external view override returns (uint256) {
    uint32 currentWeek = roundTimestamp(uint32(blockNumber));
    require(blockNumber < getBlockNumber() && currentWeek > 0, "block not yet mined");

    return totalSupplyLine.actualValue(currentWeek, uint32(blockNumber));
  }

  /**
   * @dev Returns the delegate that `account` has chosen.
   */
  function delegates(address /* account */) external pure override returns (address) {
    revert("not implemented");
  }

  /**
   * @dev Delegates votes from the sender to `delegatee`.
   */
  function delegate(address /* delegatee */) external pure override {
    revert("not implemented");
  }

  /**
   * @dev Delegates votes from signer to `delegatee`.
   */
  function delegateBySig(
    address, // delegatee
    uint256, // nonce
    uint256, // expiry
    uint8, // v
    bytes32, // r
    bytes32 // s
  ) external pure override {
    revert("not implemented");
  }

  uint256[50] private __gap;
}
