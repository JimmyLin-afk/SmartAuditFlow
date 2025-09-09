// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

interface IWithCooldownHarness {
  function setDefaultCooldownTime(uint256 cooldownTime) external;

  function setCooldownTimes(address[] calldata rateFeedIDs, uint256[] calldata cooldownTimes) external;

  function getCooldown(address rateFeedID) external view returns (uint256);

  function defaultCooldownTime() external view returns (uint256);

  function rateFeedCooldownTime(address rateFeedID) external view returns (uint256);
}
