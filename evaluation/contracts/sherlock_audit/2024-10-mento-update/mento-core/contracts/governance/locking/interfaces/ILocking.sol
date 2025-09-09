// SPDX-License-Identifier: MIT
pragma solidity 0.8.18;

/**
 * @title ILocking
 * @notice Interface for the Locking contract
 */

interface ILocking {
  /**
   * @notice Locks a specified amount of tokens for a given period
   * @param account Account for which tokens are being locked
   * @param delegate Address that will receive the voting power from the locked tokens
   * If address(0) passed, voting power will be lost
   * @param amount Amount of tokens to lock
   * @param slope Period over which the tokens will unlock
   * @param cliff Initial period during which tokens remain locked and do not start unlocking
   * @return Id for the created lock
   */
  function lock(
    address account,
    address delegate,
    uint96 amount,
    uint32 slope,
    uint32 cliff
  ) external returns (uint256);
}
