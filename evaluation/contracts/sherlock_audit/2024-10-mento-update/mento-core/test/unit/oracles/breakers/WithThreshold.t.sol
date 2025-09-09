// SPDX-License-Identifier: GPL-3.0-or-later
// solhint-disable func-name-mixedcase, var-name-mixedcase, state-visibility
// solhint-disable const-name-snakecase, max-states-count, contract-name-camelcase
pragma solidity ^0.8;

import { Test } from "mento-std/Test.sol";
import { IWithThresholdHarness } from "test/utils/harnesses/IWithThresholdHarness.sol";

contract WithThresholdTest is Test {
  event DefaultRateChangeThresholdUpdated(uint256 defaultRateChangeThreshold);
  event RateChangeThresholdUpdated(address rateFeedID, uint256 rateChangeThreshold);

  IWithThresholdHarness harness;

  function setUp() public virtual {
    harness = IWithThresholdHarness(deployCode("WithThresholdHarness"));
  }

  function test_setDefaultRateChangeThreshold() public {
    uint256 testThreshold = 1e20;
    vm.expectEmit(true, true, true, true);
    emit DefaultRateChangeThresholdUpdated(testThreshold);
    harness.setDefaultRateChangeThreshold(testThreshold);
    assertEq(harness.defaultRateChangeThreshold(), testThreshold);
  }

  function test_setRateChangeThresholds_withZeroAddress_reverts() public {
    address[] memory rateFeedIDs = new address[](1);
    uint256[] memory thresholds = new uint256[](1);

    vm.expectRevert("rate feed invalid");
    harness.setRateChangeThresholds(rateFeedIDs, thresholds);
  }

  function test_setRateChangeThresholds_withMismatchingArrays_reverts() public {
    address[] memory rateFeedIDs = new address[](2);
    uint256[] memory thresholds = new uint256[](1);

    vm.expectRevert("array length missmatch");
    harness.setRateChangeThresholds(rateFeedIDs, thresholds);
  }

  function test_setRateChangeThresholds_emitsEvents() public {
    address[] memory rateFeedIDs = new address[](2);
    uint256[] memory thresholds = new uint256[](2);

    rateFeedIDs[0] = address(1111);
    rateFeedIDs[1] = address(2222);
    thresholds[0] = 1e20;
    thresholds[1] = 2e20;

    vm.expectEmit(true, true, true, true);
    emit RateChangeThresholdUpdated(rateFeedIDs[0], thresholds[0]);
    vm.expectEmit(true, true, true, true);
    emit RateChangeThresholdUpdated(rateFeedIDs[1], thresholds[1]);

    harness.setRateChangeThresholds(rateFeedIDs, thresholds);

    assertEq(harness.rateChangeThreshold(rateFeedIDs[0]), thresholds[0]);
    assertEq(harness.rateChangeThreshold(rateFeedIDs[1]), thresholds[1]);
  }
}

contract WithThresholdTest_exceedsThreshold is WithThresholdTest {
  uint256 constant _1PC = 0.01 * 1e24; // 1%
  uint256 constant _10PC = 0.1 * 1e24; // 10%
  uint256 constant _20PC = 0.2 * 1e24; // 20%

  address rateFeedID0 = makeAddr("rateFeedID0-10%");
  address rateFeedID1 = makeAddr("rateFeedID2-1%");
  address rateFeedID2 = makeAddr("rateFeedID3-default-20%");

  function setUp() public override {
    super.setUp();
    uint256[] memory ts = new uint256[](2);
    ts[0] = _10PC;
    ts[1] = _1PC;
    address[] memory rateFeedIDs = new address[](2);
    rateFeedIDs[0] = rateFeedID0;
    rateFeedIDs[1] = rateFeedID1;

    harness.setDefaultRateChangeThreshold(_20PC);
    harness.setRateChangeThresholds(rateFeedIDs, ts);
  }

  function test_exceedsThreshold_withDefault_whenWithin_isFalse() public view {
    assertEq(harness.exceedsThreshold(1e24, 1.1 * 1e24, rateFeedID2), false);
    assertEq(harness.exceedsThreshold(1e24, 0.9 * 1e24, rateFeedID2), false);
  }

  function test_exceedsThreshold_withDefault_whenNotWithin_isTrue() public view {
    assertEq(harness.exceedsThreshold(1e24, 1.3 * 1e24, rateFeedID2), true);
    assertEq(harness.exceedsThreshold(1e24, 0.7 * 1e24, rateFeedID2), true);
  }

  function test_exceedsThreshold_withOverride_whenWithin_isTrue() public view {
    assertEq(harness.exceedsThreshold(1e24, 1.1 * 1e24, rateFeedID1), true);
    assertEq(harness.exceedsThreshold(1e24, 0.9 * 1e24, rateFeedID1), true);
    assertEq(harness.exceedsThreshold(1e24, 1.11 * 1e24, rateFeedID0), true);
    assertEq(harness.exceedsThreshold(1e24, 0.89 * 1e24, rateFeedID0), true);
  }

  function test_exceedsThreshold_withOverride_whenNotWithin_isFalse() public view {
    assertEq(harness.exceedsThreshold(1e24, 1.01 * 1e24, rateFeedID1), false);
    assertEq(harness.exceedsThreshold(1e24, 1.01 * 1e24, rateFeedID0), false);
  }
}
