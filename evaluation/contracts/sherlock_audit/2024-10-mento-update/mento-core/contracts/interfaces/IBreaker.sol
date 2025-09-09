// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity >0.5.13 <0.9;

/**
 * @title Breaker Interface
 * @notice Defines the basic interface for a Breaker
 */
interface IBreaker {
  /**
   * @notice Emitted when the sortedOracles address is updated.
   * @param newSortedOracles The address of the new sortedOracles.
   */
  event SortedOraclesUpdated(address newSortedOracles);

  /**
   * @notice Retrieve the cooldown time for the breaker.
   * @param rateFeedID The rate feed to get the cooldown for
   * @return cooldown The amount of time that must pass before the breaker can reset.
   * @dev when cooldown is 0 auto reset will not be attempted.
   */
  function getCooldown(address rateFeedID) external view returns (uint256 cooldown);

  /**
   * @notice Check if the criteria have been met, by a specified rateFeedID, to trigger the breaker.
   * @param rateFeedID The address of the rate feed to run the check against.
   * @return triggerBreaker A boolean indicating whether or not the breaker
   *                        should be triggered for the given rate feed.
   */
  function shouldTrigger(address rateFeedID) external returns (bool triggerBreaker);

  /**
   * @notice Check if the criteria to automatically reset the breaker have been met.
   * @param rateFeedID The address of rate feed the criteria should be checked against.
   * @return resetBreaker A boolean indicating whether the breaker
   *                      should be reset for the given rate feed.
   * @dev Allows the definition of additional critera to check before reset.
   *      If no additional criteria is needed set to !shouldTrigger();
   */
  function shouldReset(address rateFeedID) external returns (bool resetBreaker);
}
