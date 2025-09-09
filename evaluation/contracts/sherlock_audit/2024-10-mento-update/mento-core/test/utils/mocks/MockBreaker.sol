// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

import { IBreaker } from "contracts/interfaces/IBreaker.sol";

contract MockBreaker is IBreaker {
  uint256 public cooldown;
  bool public trigger;
  bool public reset;

  constructor(uint256 _cooldown, bool _trigger, bool _reset) {
    cooldown = _cooldown;
    trigger = _trigger;
    reset = _reset;
  }

  function getCooldown(address) external view returns (uint256) {
    return cooldown;
  }

  function setCooldown(uint256 _cooldown) external {
    cooldown = _cooldown;
  }

  function shouldTrigger(address) external view returns (bool) {
    return trigger;
  }

  function setTrigger(bool _trigger) external {
    trigger = _trigger;
  }

  function shouldReset(address) external view returns (bool) {
    return reset;
  }

  function setReset(bool _reset) external {
    reset = _reset;
  }
}
