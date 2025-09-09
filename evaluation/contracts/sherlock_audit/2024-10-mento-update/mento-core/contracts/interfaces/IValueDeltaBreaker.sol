// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

import { IBreaker } from "./IBreaker.sol";
import { IOwnable } from "./IOwnable.sol";
import { ISortedOracles } from "./ISortedOracles.sol";

interface IValueDeltaBreaker is IBreaker, IOwnable {
  function sortedOracles() external view returns (address);

  function referenceValues(address) external view returns (uint256);

  function breakerBox() external view returns (address);

  function setSortedOracles(ISortedOracles _sortedOracles) external;

  function setReferenceValues(address[] calldata rateFeedIDs, uint256[] calldata _referenceValues) external;

  function setBreakerBox(address _breakerBox) external;

  function setCooldownTimes(address[] calldata rateFeedIDs, uint256[] calldata cooldownTime) external;

  function getCoolDown(address rateFeedID) external view returns (uint256);

  function setDefaultCooldownTime(uint256 cooldownTime) external;

  function setDefaultRateChangeThreshold(uint256 _rateChangeTreshold) external;

  function setRateChangeThresholds(address[] calldata rateFeedIDs, uint256[] calldata rateChangeThresholds) external;

  function defaultCooldownTime() external view returns (uint256);

  function defaultRateChangeThreshold() external view returns (uint256);

  function rateChangeThreshold(address) external view returns (uint256);
}
