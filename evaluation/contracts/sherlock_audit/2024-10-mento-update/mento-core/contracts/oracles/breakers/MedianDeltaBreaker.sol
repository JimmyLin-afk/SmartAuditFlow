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
 * @title   Median Delta Breaker
 * @notice  Breaker contract that will trigger when an updated oracle median rate changes
 *          more than a configured relative threshold from the previous one. If this
 *          breaker is triggered for a rate feed it should be set to no trading mode.
 */
contract MedianDeltaBreaker is IBreaker, WithCooldown, WithThreshold, Ownable {
  using SafeMath for uint256;
  using FixidityLib for FixidityLib.Fraction;

  /* ==================== Events ==================== */
  event SmoothingFactorSet(address rateFeedId, uint256 smoothingFactor);
  event BreakerBoxUpdated(address breakerBox);

  event MedianRateEMAReset(address rateFeedID);

  /* ==================== State Variables ==================== */
  // Address of the Mento SortedOracles contract
  ISortedOracles public sortedOracles;

  // Address of the BreakerBox contract
  address public breakerBox;

  // Default smoothing factor for EMA as a Fixidity value
  uint256 public constant DEFAULT_SMOOTHING_FACTOR = 1e24;

  // Smoothing factor per rate feed
  mapping(address => FixidityLib.Fraction) public smoothingFactors;

  // EMA of the median rates per rate feed
  mapping(address => uint256) public medianRatesEMA;

  /* ==================== Constructor ==================== */

  constructor(
    uint256 _defaultCooldownTime,
    uint256 _defaultRateChangeThreshold,
    ISortedOracles _sortedOracles,
    address _breakerBox,
    address[] memory rateFeedIDs,
    uint256[] memory rateChangeThresholds,
    uint256[] memory cooldownTimes
  ) public {
    _transferOwnership(msg.sender);
    setSortedOracles(_sortedOracles);
    setBreakerBox(_breakerBox);

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
  function setCooldownTime(address[] calldata rateFeedIDs, uint256[] calldata cooldownTimes) external onlyOwner {
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
   * @notice Sets the address of the sortedOracles contract.
   * @param _sortedOracles The new address of the sorted oracles contract.
   */
  function setSortedOracles(ISortedOracles _sortedOracles) public onlyOwner {
    require(address(_sortedOracles) != address(0), "SortedOracles address must be set");
    sortedOracles = _sortedOracles;
    emit SortedOraclesUpdated(address(_sortedOracles));
  }

  /**
   * @notice Sets the address of the BreakerBox contract.
   * @param _breakerBox The new address of the breaker box contract.
   */
  function setBreakerBox(address _breakerBox) public onlyOwner {
    require(_breakerBox != address(0), "BreakerBox address must be set");
    breakerBox = _breakerBox;
    emit BreakerBoxUpdated(_breakerBox);
  }

  /*
   * @notice Sets the smoothing factor for a rate feed.
   * @param rateFeedID The rate feed to be updated.
   * @param smoothingFactor The new smoothingFactor value.
   */
  function setSmoothingFactor(address rateFeedID, uint256 newSmoothingFactor) external onlyOwner {
    FixidityLib.Fraction memory _newSmoothingFactor = FixidityLib.wrap(newSmoothingFactor);
    require(_newSmoothingFactor.lte(FixidityLib.fixed1()), "Smoothing factor must be <= 1");
    smoothingFactors[rateFeedID] = _newSmoothingFactor;
    emit SmoothingFactorSet(rateFeedID, newSmoothingFactor);
  }

  /**
   * @notice Resets the median rates EMA for a rate feed.
   * @param rateFeedID the targeted rate feed.
   * @dev Should be called when the breaker is disabled for a rate feed.
   */
  function resetMedianRateEMA(address rateFeedID) external onlyOwner {
    require(rateFeedID != address(0), "RateFeed address must be set");
    medianRatesEMA[rateFeedID] = 0;
    emit MedianRateEMAReset(rateFeedID);
  }

  /* ==================== View Functions ==================== */

  /**
   * @notice  Get the smoothing factor for a rate feed.
   * @param   rateFeedID The rate feed to be checked.
   * @return  smoothingFactor The smoothingFactor for the rate feed.
   */
  function getSmoothingFactor(address rateFeedID) public view returns (uint256) {
    uint256 factor = smoothingFactors[rateFeedID].unwrap();
    if (factor == 0) {
      return DEFAULT_SMOOTHING_FACTOR;
    }
    return factor;
  }

  /**
   * @notice  Check if the current median report rate for a rate feed change, relative
   *          to the last median report, is greater than the configured threshold.
   *          If the change is greater than the threshold the breaker will be triggered.
   * @param   rateFeedID The rate feed to be checked.
   * @return  triggerBreaker  A bool indicating whether or not this breaker
   *                          should be tripped for the rate feed.
   */
  function shouldTrigger(address rateFeedID) public returns (bool triggerBreaker) {
    require(msg.sender == breakerBox, "Caller must be the BreakerBox contract");

    // slither-disable-next-line unused-return
    (uint256 currentMedian, ) = sortedOracles.medianRate(rateFeedID);

    uint256 previousRatesEMA = medianRatesEMA[rateFeedID];
    if (previousRatesEMA == 0) {
      // Previous recorded EMA will be 0 the first time this rate feed is checked.
      medianRatesEMA[rateFeedID] = currentMedian;
      return false;
    }

    FixidityLib.Fraction memory smoothingFactor = FixidityLib.wrap(getSmoothingFactor(rateFeedID));
    medianRatesEMA[rateFeedID] = FixidityLib
      .wrap(currentMedian)
      .multiply(smoothingFactor)
      .add(FixidityLib.wrap(previousRatesEMA).multiply(FixidityLib.fixed1().subtract(smoothingFactor)))
      .unwrap();

    return exceedsThreshold(previousRatesEMA, currentMedian, rateFeedID);
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
