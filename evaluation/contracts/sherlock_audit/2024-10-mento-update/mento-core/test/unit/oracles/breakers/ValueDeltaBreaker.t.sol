// SPDX-License-Identifier: GPL-3.0-or-later
// solhint-disable func-name-mixedcase, var-name-mixedcase, state-visibility
// solhint-disable const-name-snakecase, max-states-count, contract-name-camelcase
pragma solidity ^0.8;

import { Test } from "mento-std/Test.sol";

import { FixidityLib } from "celo/contracts/common/FixidityLib.sol";

import { MockSortedOracles } from "test/utils/mocks/MockSortedOracles.sol";
import { IValueDeltaBreaker } from "contracts/interfaces/IValueDeltaBreaker.sol";
import { ISortedOracles } from "contracts/interfaces/ISortedOracles.sol";

contract ValueDeltaBreakerTest is Test {
  using FixidityLib for FixidityLib.Fraction;

  address notDeployer;

  address rateFeedID1;
  address rateFeedID2;
  address rateFeedID3;
  MockSortedOracles sortedOracles;
  IValueDeltaBreaker breaker;

  uint256 defaultThreshold = 0.15 * 10 ** 24; // 15%
  uint256 defaultCooldownTime = 5 minutes;

  address[] rateFeedIDs = new address[](1);
  uint256[] rateChangeThresholds = new uint256[](1);
  uint256[] cooldownTimes = new uint256[](1);

  event BreakerTriggered(address indexed rateFeedID);
  event BreakerReset(address indexed rateFeedID);
  event DefaultCooldownTimeUpdated(uint256 newCooldownTime);
  event CooldownTimeUpdated(address indexed rateFeedID, uint256 newCooldownTime);
  event DefaultRateChangeThresholdUpdated(uint256 newMinRateChangeThreshold);
  event SortedOraclesUpdated(address newSortedOracles);
  event RateChangeThresholdUpdated(address rateFeedID1, uint256 rateChangeThreshold);

  function setUp() public virtual {
    notDeployer = makeAddr("notDeployer");
    rateFeedID1 = makeAddr("rateFeedID1");
    rateFeedID2 = makeAddr("rateFeedID2");
    rateFeedID3 = makeAddr("rateFeedID3");

    rateFeedIDs[0] = rateFeedID2;
    rateChangeThresholds[0] = 0.9 * 10 ** 24;
    cooldownTimes[0] = 10 minutes;

    sortedOracles = new MockSortedOracles();

    sortedOracles.addOracle(rateFeedID1, makeAddr("OracleClient"));
    sortedOracles.addOracle(rateFeedID2, makeAddr("oracleClient"));
    sortedOracles.addOracle(rateFeedID3, makeAddr("oracleClient1"));

    breaker = IValueDeltaBreaker(
      deployCode(
        "ValueDeltaBreaker",
        abi.encode(
          defaultCooldownTime,
          defaultThreshold,
          ISortedOracles(address(sortedOracles)),
          rateFeedIDs,
          rateChangeThresholds,
          cooldownTimes
        )
      )
    );
  }
}

contract ValueDeltaBreakerTest_constructorAndSetters is ValueDeltaBreakerTest {
  /* ---------- Constructor ---------- */

  function test_constructor_shouldSetOwner() public view {
    assertEq(breaker.owner(), address(this));
  }

  function test_constructor_shouldSetDefaultCooldownTime() public view {
    assertEq(breaker.defaultCooldownTime(), defaultCooldownTime);
  }

  function test_constructor_shouldSetDefaultRateChangeThreshold() public view {
    assertEq(breaker.defaultRateChangeThreshold(), defaultThreshold);
  }

  function test_constructor_shouldSetSortedOracles() public view {
    assertEq(address(breaker.sortedOracles()), address(sortedOracles));
  }

  function test_constructor_shouldSetRateChangeThresholds() public view {
    assertEq(breaker.rateChangeThreshold(rateFeedIDs[0]), rateChangeThresholds[0]);
  }

  function test_constructor_shouldSetCooldownTimes() public view {
    assertEq(breaker.getCooldown(rateFeedIDs[0]), cooldownTimes[0]);
  }

  /* ---------- Setters ---------- */

  function test_setDefaultCooldownTime_whenCallerIsNotOwner_shouldRevert() public {
    vm.expectRevert("Ownable: caller is not the owner");
    vm.prank(notDeployer);
    breaker.setDefaultCooldownTime(2 minutes);
  }

  function test_setDefaultCooldownTime_whenCallerIsOwner_shouldUpdateAndEmit() public {
    uint256 testCooldown = 39 minutes;
    vm.expectEmit(false, false, false, true);
    emit DefaultCooldownTimeUpdated(testCooldown);

    breaker.setDefaultCooldownTime(testCooldown);

    assertEq(breaker.defaultCooldownTime(), testCooldown);
  }

  function test_setRateChangeThreshold_whenCallerIsNotOwner_shouldRevert() public {
    vm.expectRevert("Ownable: caller is not the owner");
    vm.prank(notDeployer);

    breaker.setDefaultRateChangeThreshold(123456);
  }

  function test_setDefaultRateChangeThreshold_whenValueGreaterThanOne_shouldRevert() public {
    vm.expectRevert("value must be less than 1");
    breaker.setDefaultRateChangeThreshold(1 * 10 ** 24);
  }

  function test_setDefaultRateChangeThreshold_whenCallerIsOwner_shouldUpdateAndEmit() public {
    uint256 testThreshold = 0.1 * 10 ** 24;
    vm.expectEmit(false, false, false, true);
    emit DefaultRateChangeThresholdUpdated(testThreshold);

    breaker.setDefaultRateChangeThreshold(testThreshold);

    assertEq(breaker.defaultRateChangeThreshold(), testThreshold);
  }

  function test_setSortedOracles_whenSenderIsNotOwner_shouldRevert() public {
    vm.prank(notDeployer);
    vm.expectRevert("Ownable: caller is not the owner");
    breaker.setSortedOracles(ISortedOracles(address(0)));
  }

  function test_setSortedOracles_whenAddressIsZero_shouldRevert() public {
    vm.expectRevert("SortedOracles address must be set");
    breaker.setSortedOracles(ISortedOracles(address(0)));
  }

  function test_setSortedOracles_whenSenderIsOwner_shouldUpdateAndEmit() public {
    address newSortedOracles = makeAddr("newSortedOracles");
    vm.expectEmit(true, true, true, true);
    emit SortedOraclesUpdated(newSortedOracles);

    breaker.setSortedOracles(ISortedOracles(newSortedOracles));

    assertEq(address(breaker.sortedOracles()), newSortedOracles);
  }

  function test_setRateChangeThresholds_whenSenderIsNotOwner_shouldRevert() public {
    vm.prank(notDeployer);
    vm.expectRevert("Ownable: caller is not the owner");
    breaker.setRateChangeThresholds(rateFeedIDs, rateChangeThresholds);
  }

  function test_setRateChangeThresholds_whenValuesAreDifferentLengths_shouldRevert() public {
    address[] memory rateFeedIDs2 = new address[](2);
    rateFeedIDs2[0] = makeAddr("randomRateFeed");
    rateFeedIDs2[1] = makeAddr("randomRateFeed2");
    vm.expectRevert("array length missmatch");
    breaker.setRateChangeThresholds(rateFeedIDs2, rateChangeThresholds);
  }

  function test_setRateChangeThresholds_whenThresholdIsMoreThan0_shouldRevert() public {
    rateChangeThresholds[0] = 1 * 10 ** 24;
    vm.expectRevert("value must be less than 1");
    breaker.setRateChangeThresholds(rateFeedIDs, rateChangeThresholds);
  }

  function test_setRateChangeThresholds_whenSenderIsOwner_shouldUpdateAndEmit() public {
    vm.expectEmit(true, true, true, true);
    emit RateChangeThresholdUpdated(rateFeedIDs[0], rateChangeThresholds[0]);
    breaker.setRateChangeThresholds(rateFeedIDs, rateChangeThresholds);
    assertEq(breaker.rateChangeThreshold(rateFeedIDs[0]), rateChangeThresholds[0]);
  }

  /* ---------- Getters ---------- */
  function test_getCooldown_withDefault_shouldReturnDefaultCooldown() public view {
    assertEq(breaker.getCooldown(rateFeedID1), defaultCooldownTime);
  }

  function test_getCooldown_withoutdefault_shouldReturnSpecificCooldown() public view {
    assertEq(breaker.getCooldown(rateFeedIDs[0]), cooldownTimes[0]);
  }
}

contract ValueDeltaBreakerTest_shouldTrigger is ValueDeltaBreakerTest {
  using FixidityLib for FixidityLib.Fraction;

  function setUp() public override {
    super.setUp();

    address[] memory _rateFeedIDs = new address[](3);
    uint256[] memory _referenceValues = new uint256[](3);
    _rateFeedIDs[0] = rateFeedID1;
    _rateFeedIDs[1] = rateFeedID2;
    _rateFeedIDs[2] = rateFeedID3;
    _referenceValues[0] = FixidityLib.fixed1().unwrap();
    _referenceValues[1] = FixidityLib.fixed1().unwrap();
    _referenceValues[2] = FixidityLib.fixed1().unwrap();

    breaker.setReferenceValues(_rateFeedIDs, _referenceValues);
  }

  function updateMedianByPercent(uint256 medianChangeScaleFactor, address _rateFeedID) public {
    uint256 previousMedianRate = 10 ** 24;
    uint256 currentMedianRate = (previousMedianRate * medianChangeScaleFactor) / 10 ** 24;
    vm.mockCall(
      address(sortedOracles),
      abi.encodeWithSelector(sortedOracles.medianRate.selector),
      abi.encode(currentMedianRate, 1)
    );
    vm.expectCall(address(sortedOracles), abi.encodeWithSelector(sortedOracles.medianRate.selector, _rateFeedID));
  }

  function test_shouldTrigger_withDefaultThreshold_shouldTrigger() public {
    assertEq(breaker.rateChangeThreshold(rateFeedID1), 0);
    updateMedianByPercent(0.7 * 10 ** 24, rateFeedID1);
    assertTrue(breaker.shouldTrigger(rateFeedID1));
  }

  function test_shouldTrigger_whenThresholdIsLargerThanMedian_shouldNotTrigger() public {
    updateMedianByPercent(0.7 * 10 ** 24, rateFeedID1);

    rateChangeThresholds[0] = 0.8 * 10 ** 24;
    rateFeedIDs[0] = rateFeedID1;
    breaker.setRateChangeThresholds(rateFeedIDs, rateChangeThresholds);
    assertEq(breaker.rateChangeThreshold(rateFeedID1), rateChangeThresholds[0]);

    assertFalse(breaker.shouldTrigger(rateFeedID1));
  }

  function test_shouldTrigger_whithDefaultThreshold_ShouldNotTrigger() public {
    assertEq(breaker.rateChangeThreshold(rateFeedID3), 0);

    updateMedianByPercent(1.1 * 10 ** 24, rateFeedID3);

    assertFalse(breaker.shouldTrigger(rateFeedID3));
  }

  function test_shouldTrigger_whenThresholdIsSmallerThanMedian_ShouldTrigger() public {
    updateMedianByPercent(1.1 * 10 ** 24, rateFeedID3);
    rateChangeThresholds[0] = 0.01 * 10 ** 24;
    rateFeedIDs[0] = rateFeedID3;
    breaker.setRateChangeThresholds(rateFeedIDs, rateChangeThresholds);
    assertEq(breaker.rateChangeThreshold(rateFeedID3), rateChangeThresholds[0]);

    assertTrue(breaker.shouldTrigger(rateFeedID3));
  }
}
