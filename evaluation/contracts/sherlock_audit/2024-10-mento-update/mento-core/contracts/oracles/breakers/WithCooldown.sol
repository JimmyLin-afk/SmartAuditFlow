// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.5.13;

/**
 * @title   Breaker With Cooldown
 * @notice  Utility portion of a Breaker contract which deals with the
 *          cooldown component.
 */
contract WithCooldown {
  /* ==================== Events ==================== */
  /**
   * @notice Emitted after the cooldownTime has been updated.
   * @param newCooldownTime The new cooldownTime of the breaker.
   */
  event DefaultCooldownTimeUpdated(uint256 newCooldownTime);

  /**
   * @notice Emitted after the cooldownTime has been updated.
   * @param rateFeedID The rateFeedID targeted.
   * @param newCooldownTime The new cooldownTime of the breaker.
   */
  event RateFeedCooldownTimeUpdated(address rateFeedID, uint256 newCooldownTime);

  /* ==================== State Variables ==================== */

  // The amount of time that must pass before the breaker can be reset for a rate feed.
  // Should be set to 0 to force a manual reset.
  uint256 public defaultCooldownTime;
  mapping(address => uint256) public rateFeedCooldownTime;

  /* ==================== View Functions ==================== */

  /**
   * @notice Get the cooldown time for a rateFeedID
   * @param rateFeedID the targeted rate feed.
   * @return the rate specific or default cooldown
   */
  function getCooldown(address rateFeedID) public view returns (uint256) {
    uint256 _rateFeedCooldownTime = rateFeedCooldownTime[rateFeedID];
    if (_rateFeedCooldownTime == 0) {
      return defaultCooldownTime;
    }
    return _rateFeedCooldownTime;
  }

  /* ==================== Internal Functions ==================== */

  /**
   * @notice Sets the cooldown time to the specified value for a rate feed.
   * @param rateFeedIDs the targeted rate feed.
   * @param cooldownTimes The new cooldownTime value.
   * @dev Should be set to 0 to force a manual reset.
   */
  function _setCooldownTimes(address[] memory rateFeedIDs, uint256[] memory cooldownTimes) internal {
    require(rateFeedIDs.length == cooldownTimes.length, "array length missmatch");
    for (uint256 i = 0; i < rateFeedIDs.length; i++) {
      require(rateFeedIDs[i] != address(0), "rate feed invalid");
      rateFeedCooldownTime[rateFeedIDs[i]] = cooldownTimes[i];
      emit RateFeedCooldownTimeUpdated(rateFeedIDs[i], cooldownTimes[i]);
    }
  }

  /**
   * @notice Sets the cooldownTime to the specified value for a rate feed.
   * @param cooldownTime The new cooldownTime value.
   * @dev Should be set to 0 to force a manual reset.
   */
  function _setDefaultCooldownTime(uint256 cooldownTime) internal {
    defaultCooldownTime = cooldownTime;
    emit DefaultCooldownTimeUpdated(cooldownTime);
  }
}
