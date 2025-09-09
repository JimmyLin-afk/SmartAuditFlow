// SPDX-License-Identifier: GPL-3.0-or-later
// solhint-disable func-name-mixedcase, var-name-mixedcase, state-visibility
// solhint-disable const-name-snakecase, max-states-count, contract-name-camelcase
pragma solidity ^0.8;

import { Test } from "mento-std/Test.sol";
import { IWithCooldownHarness } from "test/utils/harnesses/IWithCooldownHarness.sol";

contract WithCooldownTest is Test {
  event DefaultCooldownTimeUpdated(uint256 newCooldownTime);
  event RateFeedCooldownTimeUpdated(address rateFeedID, uint256 newCooldownTime);

  IWithCooldownHarness harness;

  function setUp() public {
    harness = IWithCooldownHarness(deployCode("WithCooldownHarness"));
  }

  function test_setDefaultCooldownTime() public {
    uint256 testCooldown = 39 minutes;
    vm.expectEmit(true, true, true, true);
    emit DefaultCooldownTimeUpdated(testCooldown);
    harness.setDefaultCooldownTime(testCooldown);
    assertEq(harness.defaultCooldownTime(), testCooldown);
  }

  function test_setCooldownTimes_withZeroAddress_reverts() public {
    address[] memory rateFeedIDs = new address[](1);
    uint256[] memory cooldownTimes = new uint256[](1);

    vm.expectRevert("rate feed invalid");
    harness.setCooldownTimes(rateFeedIDs, cooldownTimes);
  }

  function test_setCooldownTimes_withMismatchingArrays_reverts() public {
    address[] memory rateFeedIDs = new address[](2);
    uint256[] memory cooldownTimes = new uint256[](1);

    vm.expectRevert("array length missmatch");
    harness.setCooldownTimes(rateFeedIDs, cooldownTimes);
  }

  function test_setCoolDownTimes_emitsEvents() public {
    address[] memory rateFeedIDs = new address[](2);
    uint256[] memory cooldownTimes = new uint256[](2);

    rateFeedIDs[0] = address(1111);
    rateFeedIDs[1] = address(2222);
    cooldownTimes[0] = 1;
    cooldownTimes[1] = 2;

    vm.expectEmit(true, true, true, true);
    emit RateFeedCooldownTimeUpdated(rateFeedIDs[0], cooldownTimes[0]);
    vm.expectEmit(true, true, true, true);
    emit RateFeedCooldownTimeUpdated(rateFeedIDs[1], cooldownTimes[1]);

    harness.setCooldownTimes(rateFeedIDs, cooldownTimes);

    assertEq(harness.rateFeedCooldownTime(rateFeedIDs[0]), cooldownTimes[0]);
    assertEq(harness.rateFeedCooldownTime(rateFeedIDs[1]), cooldownTimes[1]);
  }

  function test_getCooldown_whenNoRateFeedSpecific_usesDefault() public {
    uint256 testCooldown = 39 minutes;
    address rateFeedID = address(1111);
    harness.setDefaultCooldownTime(testCooldown);

    assertEq(harness.getCooldown(rateFeedID), testCooldown);
  }

  function test_getCooldown_whenRateFeedSpecific_usesRateSpecific() public {
    uint256 testCooldown = 39 minutes;
    uint256 defaultCooldown = 10 minutes;
    address rateFeedID = address(1111);

    address[] memory rateFeedIDs = new address[](1);
    rateFeedIDs[0] = rateFeedID;
    uint256[] memory cooldownTimes = new uint256[](1);
    cooldownTimes[0] = testCooldown;

    harness.setCooldownTimes(rateFeedIDs, cooldownTimes);
    harness.setDefaultCooldownTime(defaultCooldown);

    assertEq(harness.getCooldown(rateFeedID), testCooldown);
  }
}
