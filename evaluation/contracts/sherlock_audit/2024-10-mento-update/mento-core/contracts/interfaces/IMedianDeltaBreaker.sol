// SPDX-License-Identifier: GPL-3.0-or-later
// solhint-disable func-name-mixedcase
pragma solidity ^0.8;

import { IBreaker } from "./IBreaker.sol";
import { IOwnable } from "./IOwnable.sol";
import { ISortedOracles } from "./ISortedOracles.sol";

interface IMedianDeltaBreaker is IBreaker, IOwnable {
  function sortedOracles() external view returns (address);

  function breakerBox() external view returns (address);

  function DEFAULT_SMOOTHING_FACTOR() external view returns (uint256);

  function smoothingFactors(address) external view returns (uint256);

  function medianRatesEMA(address) external view returns (uint256);

  function setSortedOracles(ISortedOracles _sortedOracles) external;

  function setBreakerBox(address _breakerBox) external;

  function setCooldownTime(address[] calldata rateFeedIDs, uint256 cooldownTime) external;

  function getCoolDown(address rateFeedID) external view returns (uint256);

  function setDefaultCooldownTime(uint256 cooldownTime) external;

  function setDefaultRateChangeThreshold(uint256) external;

  function setRateChangeThresholds(address[] calldata rateFeedIDs, uint256[] calldata rateChangeThresholds) external;

  function setSmoothingFactor(address rateFeedID, uint256 smoothingFactor) external;

  function setMedianRateEMA(address rateFeedID) external;

  function getSmoothingFactor(address rateFeedID) external view returns (uint256);

  function defaultCooldownTime() external view returns (uint256);

  function defaultRateChangeThreshold() external view returns (uint256);

  function rateChangeThreshold(address) external view returns (uint256);

  function resetMedianRateEMA(address rateFeedID) external;
}
