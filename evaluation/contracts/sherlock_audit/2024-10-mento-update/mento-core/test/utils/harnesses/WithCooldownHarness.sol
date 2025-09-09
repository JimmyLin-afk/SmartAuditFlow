// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.5.13;

import { WithCooldown } from "contracts/oracles/breakers/WithCooldown.sol";

contract WithCooldownHarness is WithCooldown {
  function setDefaultCooldownTime(uint256 cooldownTime) external {
    _setDefaultCooldownTime(cooldownTime);
  }

  function setCooldownTimes(address[] calldata rateFeedIDs, uint256[] calldata cooldownTimes) external {
    _setCooldownTimes(rateFeedIDs, cooldownTimes);
  }
}
