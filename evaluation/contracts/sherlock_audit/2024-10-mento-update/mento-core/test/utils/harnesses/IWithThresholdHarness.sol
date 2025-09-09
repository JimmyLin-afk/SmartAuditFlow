// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

interface IWithThresholdHarness {
  function setDefaultRateChangeThreshold(uint256 testThreshold) external;

  function setRateChangeThresholds(address[] calldata rateFeedIDs, uint256[] calldata thresholds) external;

  function rateChangeThreshold(address rateFeedID) external view returns (uint256);

  function defaultRateChangeThreshold() external view returns (uint256);

  function rateFeedRateChangeThreshold(address rateFeedID) external view returns (uint256);

  function exceedsThreshold(
    uint256 referenceValue,
    uint256 currentValue,
    address rateFeedID
  ) external view returns (bool);
}
