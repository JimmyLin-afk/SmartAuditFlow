// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.5.13;

import { Ownable } from "openzeppelin-solidity/contracts/ownership/Ownable.sol";
import { SafeMath } from "openzeppelin-solidity/contracts/math/SafeMath.sol";
import { FixidityLib } from "celo/contracts/common/FixidityLib.sol";

import { IBreaker } from "../../interfaces/IBreaker.sol";
import { ISortedOracles } from "../../interfaces/ISortedOracles.sol";

import { WithCooldown } from "./WithCooldown.sol";
import { WithThreshold } from "./WithThreshold.sol";

/**
 * @title   Value Delta Breaker
 * @notice  Breaker contract that will trigger when the current oracle median rate change
 *          relative to a reference value is greater than a calculated threshold. If this
 *          breaker is triggered for a rate feed it should be set to no trading mode.
 */
contract ValueDeltaBreaker is IBreaker, WithCooldown, WithThreshold, Ownable {
  using SafeMath for uint256;
  using FixidityLib for FixidityLib.Fraction;

  /* ==================== State Variables ==================== */

  // Address of the Mento SortedOracles contract
  ISortedOracles public sortedOracles;

  // The reference value to check against
  mapping(address => uint256) public referenceValues;

  /* ==================== Events ==================== */

  // Emitted when the reference value is updated
  event ReferenceValueUpdated(address rateFeedID, uint256 referenceValue);

  /* ==================== Constructor ==================== */

  constructor(
    uint256 _defaultCooldownTime,
    uint256 _defaultRateChangeThreshold,
    ISortedOracles _sortedOracles,
    address[] memory rateFeedIDs,
    uint256[] memory rateChangeThresholds,
    uint256[] memory cooldownTimes
  ) public {
    _transferOwnership(msg.sender);
    setSortedOracles(_sortedOracles);

    _setDefaultCooldownTime(_defaultCooldownTime);
    _setDefaultRateChangeThreshold(_defaultRateChangeThreshold);
    _setRateChangeThresholds(rateFeedIDs, rateChangeThresholds);
    _setCooldownTimes(rateFeedIDs, cooldownTimes);
  }

  /* ==================== Restricted Functions ==================== */

  /**
   * @notice Sets the cooldown time to the specified value for a rate feed.
   * @param rateFeedIDs the targeted rate feed.
   * @param cooldownTimes The new cooldownTime value.
   * @dev Should be set to 0 to force a manual reset.
   */
  function setCooldownTimes(address[] calldata rateFeedIDs, uint256[] calldata cooldownTimes) external onlyOwner {
    _setCooldownTimes(rateFeedIDs, cooldownTimes);
  }

  /**
   * @notice Sets the cooldownTime to the specified value for a rate feed.
   * @param cooldownTime The new cooldownTime value.
   * @dev Should be set to 0 to force a manual reset.
   */
  function setDefaultCooldownTime(uint256 cooldownTime) external onlyOwner {
    _setDefaultCooldownTime(cooldownTime);
  }

  /**
   * @notice Sets rateChangeThreshold.
   * @param _defaultRateChangeThreshold The new rateChangeThreshold value.
   */
  function setDefaultRateChangeThreshold(uint256 _defaultRateChangeThreshold) external onlyOwner {
    _setDefaultRateChangeThreshold(_defaultRateChangeThreshold);
  }

  /**
   * @notice Configures rate feed to rate threshold pairs.
   * @param rateFeedIDs Collection of the addresses rate feeds.
   * @param rateChangeThresholds Collection of the rate thresholds.
   */
  function setRateChangeThresholds(
    address[] calldata rateFeedIDs,
    uint256[] calldata rateChangeThresholds
  ) external onlyOwner {
    _setRateChangeThresholds(rateFeedIDs, rateChangeThresholds);
  }

  /**
   * @notice Configures rate feed to reference value pairs.
   * @param rateFeedIDs Collection of the addresses rate feeds.
   * @param _referenceValues Collection of referance values.
   */
  function setReferenceValues(address[] calldata rateFeedIDs, uint256[] calldata _referenceValues) external onlyOwner {
    require(rateFeedIDs.length == _referenceValues.length, "array length missmatch");
    for (uint256 i = 0; i < rateFeedIDs.length; i++) {
      require(rateFeedIDs[i] != address(0), "rate feed invalid");
      referenceValues[rateFeedIDs[i]] = _referenceValues[i];
      emit ReferenceValueUpdated(rateFeedIDs[i], _referenceValues[i]);
    }
  }

  /**
   * @notice Sets the address of the sortedOracles contract.
   * @param _sortedOracles The new address of the sorted oracles contract.
   */
  function setSortedOracles(ISortedOracles _sortedOracles) public onlyOwner {
    require(address(_sortedOracles) != address(0), "SortedOracles address must be set");
    sortedOracles = _sortedOracles;
    emit SortedOraclesUpdated(address(_sortedOracles));
  }

  /* ==================== Public Functions ==================== */

  /**
   * @notice  Check if the current median report rate change, for a rate feed, relative
   *          to the last median report is greater than a calculated threshold.
   *          If the change is greater than the threshold the breaker will be triggered.
   * @param   rateFeedID The rate feed to be checked.
   * @return  triggerBreaker  A bool indicating whether or not this breaker
   *                          should be tripped for the rate feed.
   */
  function shouldTrigger(address rateFeedID) public returns (bool triggerBreaker) {
    // slither-disable-next-line unused-return
    (uint256 currentMedian, ) = sortedOracles.medianRate(rateFeedID);
    uint256 referenceValue = referenceValues[rateFeedID];

    if (referenceValue == 0) {
      // Never trigger if reference value is not set
      return false;
    }

    return exceedsThreshold(referenceValue, currentMedian, rateFeedID);
  }

  /**
   * @notice  Checks whether or not the conditions have been met
   *          for the specifed rate feed to be reset.
   * @return  resetBreaker A bool indicating whether or not
   *          this breaker can be reset for the given rate feed.
   */
  function shouldReset(address rateFeedID) external returns (bool resetBreaker) {
    return !shouldTrigger(rateFeedID);
  }
}
