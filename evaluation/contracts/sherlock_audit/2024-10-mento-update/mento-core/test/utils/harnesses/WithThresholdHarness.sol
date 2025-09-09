// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.5.13;

import { WithThreshold } from "contracts/oracles/breakers/WithThreshold.sol";

contract WithThresholdHarness is WithThreshold {
  function setDefaultRateChangeThreshold(uint256 testThreshold) external {
    _setDefaultRateChangeThreshold(testThreshold);
  }

  function setRateChangeThresholds(address[] calldata rateFeedIDs, uint256[] calldata thresholds) external {
    _setRateChangeThresholds(rateFeedIDs, thresholds);
  }
}
