// SPDX-License-Identifier: GPL-3.0-or-later
// solhint-disable func-name-mixedcase, var-name-mixedcase, state-visibility
// solhint-disable const-name-snakecase, max-states-count, contract-name-camelcase
pragma solidity ^0.8.26;

import { Test } from "mento-std/Test.sol";

import { MockBreaker } from "test/utils/mocks/MockBreaker.sol";
import { MockSortedOracles } from "test/utils/mocks/MockSortedOracles.sol";

import { IBreakerBox } from "contracts/interfaces/IBreakerBox.sol";
import { ISortedOracles } from "contracts/interfaces/ISortedOracles.sol";

contract BreakerBoxTest is Test {
  address rateFeedID1;
  address rateFeedID2;
  address rateFeedID3;
  address notDeployer;

  MockBreaker mockBreaker1;
  MockBreaker mockBreaker2;
  MockBreaker mockBreaker3;
  MockBreaker mockBreaker4;
  IBreakerBox breakerBox;
  MockSortedOracles sortedOracles;

  event BreakerAdded(address indexed breaker);
  event BreakerRemoved(address indexed breaker);
  event BreakerTripped(address indexed breaker, address indexed rateFeedID);
  event ResetSuccessful(address indexed rateFeedID, address indexed breaker);
  event ResetAttemptCriteriaFail(address indexed rateFeedID, address indexed breaker);
  event ResetAttemptNotCool(address indexed rateFeedID, address indexed breaker);
  event RateFeedAdded(address indexed rateFeedID);
  event RateFeedDependenciesSet(address indexed rateFeedID, address[] indexed dependencies);
  event RateFeedRemoved(address indexed rateFeedID);
  event SortedOraclesUpdated(address indexed newSortedOracles);
  event BreakerStatusUpdated(address breaker, address rateFeedID, bool status);
  event TradingModeUpdated(address indexed rateFeedID, uint256 tradingMode);

  function setUp() public {
    rateFeedID1 = makeAddr("rateFeedID1");
    rateFeedID2 = makeAddr("rateFeedID2");
    rateFeedID3 = makeAddr("rateFeedID3");
    notDeployer = makeAddr("notDeployer");

    address[] memory testRateFeedIDs = new address[](2);
    testRateFeedIDs[0] = rateFeedID1;
    testRateFeedIDs[1] = rateFeedID2;

    mockBreaker1 = new MockBreaker(0, false, false);
    mockBreaker2 = new MockBreaker(0, false, false);
    mockBreaker3 = new MockBreaker(0, false, false);
    mockBreaker4 = new MockBreaker(0, false, false);
    sortedOracles = new MockSortedOracles();

    sortedOracles.addOracle(rateFeedID1, makeAddr("oracleClient1"));
    sortedOracles.addOracle(rateFeedID2, makeAddr("oracleClient1"));

    breakerBox = IBreakerBox(deployCode("BreakerBox", abi.encode(testRateFeedIDs, sortedOracles)));
    breakerBox.addBreaker(address(mockBreaker1), 1);
  }

  function isRateFeed(address rateFeedID) public view returns (bool rateFeedIDFound) {
    address[] memory allRateFeedIDs = breakerBox.getRateFeeds();
    for (uint256 i = 0; i < allRateFeedIDs.length; i++) {
      if (allRateFeedIDs[i] == rateFeedID) {
        rateFeedIDFound = true;
        break;
      }
    }
  }

  function setUpBreaker(MockBreaker breaker, uint8 tradingMode, uint256 cooldown, bool reset, bool trigger) public {
    breaker.setCooldown(cooldown);
    breaker.setReset(reset);
    breaker.setTrigger(trigger);
    breakerBox.addBreaker(address(breaker), tradingMode);
  }

  function toggleAndAssertBreaker(address breaker, address rateFeedID, bool status) public {
    vm.expectEmit(true, true, true, true);
    emit BreakerStatusUpdated(breaker, rateFeedID, status);
    breakerBox.toggleBreaker(breaker, rateFeedID, status);
    assertEq(breakerBox.isBreakerEnabled(breaker, rateFeedID), status);
  }
}

contract BreakerBoxTest_constructorAndSetters is BreakerBoxTest {
  /* ---------- Constructor ---------- */

  function test_constructor_shouldSetOwner() public view {
    assertEq(breakerBox.owner(), address(this));
  }

  function test_constructor_shouldSetInitialBreaker() public view {
    assertEq(uint256(breakerBox.breakerTradingMode(address(mockBreaker1))), 1);
    assertTrue(breakerBox.isBreaker(address(mockBreaker1)));
  }

  function test_constructor_shouldSetSortedOracles() public view {
    assertEq(address(breakerBox.sortedOracles()), address(sortedOracles));
  }

  function test_constructor_shouldAddRateFeedIdsWithDefaultMode() public view {
    assertTrue(breakerBox.rateFeedStatus(rateFeedID1));
    assertEq(uint256(breakerBox.getRateFeedTradingMode(rateFeedID1)), 0);

    assertTrue(breakerBox.rateFeedStatus(rateFeedID2));
    assertEq(uint256(breakerBox.getRateFeedTradingMode(rateFeedID2)), 0);
  }

  /* ---------- Breakers ---------- */

  function test_addBreaker_whenNotOwner_shouldRevert() public {
    vm.expectRevert("Ownable: caller is not the owner");
    vm.prank(notDeployer);
    breakerBox.addBreaker(address(mockBreaker1), 2);
  }

  function test_addBreaker_whenAddingDuplicateBreaker_shouldRevert() public {
    vm.expectRevert("This breaker has already been added");
    breakerBox.addBreaker(address(mockBreaker1), 2);
  }

  function test_addBreaker_whenTradingModeIsDefault_shouldRevert() public {
    vm.expectRevert("The default trading mode can not have a breaker");
    breakerBox.addBreaker(address(mockBreaker4), 0);
  }

  function test_addBreaker_whenValidBreaker_shouldUpdateAndEmit() public {
    vm.expectEmit(true, false, false, false);
    emit BreakerAdded(address(mockBreaker2));

    breakerBox.addBreaker(address(mockBreaker2), 2);

    assertEq(uint256(breakerBox.breakerTradingMode(address(mockBreaker2))), 2);
    assertTrue(breakerBox.isBreaker(address(mockBreaker2)));
  }

  function test_removeBreaker_whenNotOwner_shouldRevert() public {
    vm.expectRevert("Ownable: caller is not the owner");
    vm.prank(notDeployer);
    breakerBox.removeBreaker(address(mockBreaker1));
  }

  function test_removeBreaker_whenBreakerHasntBeenAdded_shouldRevert() public {
    vm.expectRevert("Breaker has not been added");
    breakerBox.removeBreaker(address(mockBreaker2));
  }

  function test_removeBreaker_whenMultipleBreakers_shouldUpdateArray() public {
    breakerBox.addBreaker(address(mockBreaker2), 2);
    breakerBox.addBreaker(address(mockBreaker3), 3);
    breakerBox.addBreaker(address(mockBreaker4), 4);

    address[] memory allBreakers = breakerBox.getBreakers();
    assertEq(allBreakers.length, 4);
    assertEq(allBreakers[0], address(mockBreaker1));
    assertEq(allBreakers[1], address(mockBreaker2));
    assertEq(allBreakers[2], address(mockBreaker3));
    assertEq(allBreakers[3], address(mockBreaker4));

    breakerBox.removeBreaker(address(mockBreaker2));
    allBreakers = breakerBox.getBreakers();
    assertEq(allBreakers.length, 3);
    assertEq(allBreakers[0], address(mockBreaker1));
    assertEq(allBreakers[1], address(mockBreaker4));
    assertEq(allBreakers[2], address(mockBreaker3));
  }

  function test_removeBreaker_whenBreakerExists_shouldUpdateStorageAndEmit() public {
    vm.warp(1672527600); // 2023-01-01 00:00:00
    setUpBreaker(mockBreaker2, 2, 10, false, true);
    breakerBox.toggleBreaker(address(mockBreaker2), rateFeedID1, true);
    IBreakerBox.BreakerStatus memory status;

    vm.prank(address(sortedOracles));
    breakerBox.checkAndSetBreakers(rateFeedID1);

    assertTrue(breakerBox.isBreaker(address(mockBreaker2)));
    assertTrue(breakerBox.isBreakerEnabled(address(mockBreaker2), rateFeedID1));
    assertEq(uint256(breakerBox.breakerTradingMode(address(mockBreaker2))), 2);

    status = breakerBox.rateFeedBreakerStatus(rateFeedID1, address(mockBreaker2));
    assertEq(status.tradingMode, 2);
    assertEq(status.lastUpdatedTime, 1672527600);
    assertTrue(status.enabled);

    vm.expectEmit(true, false, false, false);
    emit BreakerRemoved(address(mockBreaker2));
    breakerBox.removeBreaker(address(mockBreaker2));

    assertFalse(breakerBox.isBreaker(address(mockBreaker2)));
    assertFalse(breakerBox.isBreakerEnabled(address(mockBreaker2), rateFeedID1));
    assertEq(uint256(breakerBox.breakerTradingMode(address(mockBreaker2))), 0);

    status = breakerBox.rateFeedBreakerStatus(rateFeedID1, address(mockBreaker2));
    assertEq(status.tradingMode, 0);
    assertEq(status.lastUpdatedTime, 0);
    assertFalse(status.enabled);
  }

  function test_toggleBreaker_whenNotOwner_shouldRevert() public {
    vm.expectRevert("Ownable: caller is not the owner");
    vm.prank(notDeployer);
    breakerBox.toggleBreaker(address(mockBreaker1), rateFeedID1, true);
  }

  function test_toggleBreaker_whenRateFeedIsNotRegistered_shouldRevert() public {
    vm.expectRevert("Rate feed ID has not been added");
    breakerBox.toggleBreaker(address(mockBreaker1), rateFeedID3, false);
  }

  function test_toggleBreaker_whenBreakerIsNotRegistered_shouldRevert() public {
    vm.expectRevert("This breaker has not been added to the BreakerBox");
    breakerBox.toggleBreaker(address(mockBreaker3), rateFeedID1, true);
  }

  function test_toggleBreaker_whenSenderIsOwner_shouldToggleAndEmit() public {
    toggleAndAssertBreaker(address(mockBreaker1), rateFeedID1, true);
    toggleAndAssertBreaker(address(mockBreaker1), rateFeedID1, false);
  }

  function test_toggleBreaker_whenBreakerIsRemoved_shouldUpdateTradingModeAndBreakerStatus() public {
    uint8 breaker3TradingMode = 1;
    uint8 breaker4TradingMode = 2;
    setUpBreaker(mockBreaker3, breaker3TradingMode, 10, false, true);
    setUpBreaker(mockBreaker4, breaker4TradingMode, 10, false, true);
    toggleAndAssertBreaker(address(mockBreaker3), rateFeedID1, true);
    toggleAndAssertBreaker(address(mockBreaker4), rateFeedID1, true);

    vm.prank(address(sortedOracles));
    breakerBox.checkAndSetBreakers(rateFeedID1);

    assertEq(uint256(breakerBox.getRateFeedTradingMode(rateFeedID1)), breaker3TradingMode | breaker4TradingMode);

    toggleAndAssertBreaker(address(mockBreaker4), rateFeedID1, false);
    assertEq(uint256(breakerBox.getRateFeedTradingMode(rateFeedID1)), breaker3TradingMode);
    assertFalse(breakerBox.isBreakerEnabled(address(mockBreaker4), rateFeedID1));

    IBreakerBox.BreakerStatus memory status = breakerBox.rateFeedBreakerStatus(rateFeedID1, address(mockBreaker4));
    assertEq(status.tradingMode, 0);
    assertEq(status.lastUpdatedTime, 0);
    assertFalse(status.enabled);
  }

  function test_toggleBreaker_whenBreakerIsAdded_shouldCheckAndSetBreakers() public {
    uint8 breaker3TradingMode = 1;
    uint8 breaker4TradingMode = 2;
    setUpBreaker(mockBreaker3, breaker3TradingMode, 10, false, true);
    setUpBreaker(mockBreaker4, breaker4TradingMode, 10, false, true);
    assertEq(uint256(breakerBox.getRateFeedTradingMode(rateFeedID1)), 0);

    breakerBox.toggleBreaker(address(mockBreaker3), rateFeedID1, true);
    assertEq(uint256(breakerBox.getRateFeedTradingMode(rateFeedID1)), breaker3TradingMode);
    breakerBox.toggleBreaker(address(mockBreaker4), rateFeedID1, true);
    assertEq(uint256(breakerBox.getRateFeedTradingMode(rateFeedID1)), breaker3TradingMode | breaker4TradingMode);
  }

  /* ---------- Rate Feed IDs ---------- */

  function test_addRateFeed_whenNotOwner_shouldRevert() public {
    vm.prank(notDeployer);
    vm.expectRevert("Ownable: caller is not the owner");
    breakerBox.addRateFeed(rateFeedID3);
  }

  function test_addRateFeed_whenAlreadyAdded_shouldRevert() public {
    vm.expectRevert("Rate feed ID has already been added");
    breakerBox.addRateFeed(rateFeedID1);
  }

  function test_addRateFeed_whenRateFeedDoesNotExistInOracleList_shouldRevert() public {
    vm.expectRevert("Rate feed ID does not exist as it has 0 oracles");
    breakerBox.addRateFeed(rateFeedID3);
  }

  function test_addRateFeed_whenRateFeedExistsInOracleList_shouldSetDefaultModeAndEmit() public {
    sortedOracles.addOracle(rateFeedID3, makeAddr("oracleAddress"));

    assertFalse(isRateFeed(rateFeedID3));
    assertFalse(breakerBox.rateFeedStatus(rateFeedID3));

    vm.expectEmit(true, true, true, true);
    emit RateFeedAdded(rateFeedID3);
    breakerBox.addRateFeed(rateFeedID3);

    uint256 tradingModeAfter = breakerBox.getRateFeedTradingMode(rateFeedID3);
    assertEq(tradingModeAfter, 0);
    assertTrue(isRateFeed(rateFeedID3));
    assertTrue(breakerBox.rateFeedStatus(rateFeedID3));
  }

  function test_setRateFeedDependencies_whenNotOwner_shouldRevert() public {
    address[] memory testRateFeedIDs = new address[](2);
    testRateFeedIDs[0] = rateFeedID2;
    testRateFeedIDs[1] = rateFeedID3;

    vm.prank(notDeployer);
    vm.expectRevert("Ownable: caller is not the owner");
    breakerBox.setRateFeedDependencies(rateFeedID1, testRateFeedIDs);
  }

  function test_setRateFeedDependencies_whenRateFeedHasNotBeenAdded_shouldRevert() public {
    address[] memory testRateFeedIDs = new address[](2);
    testRateFeedIDs[0] = rateFeedID2;
    testRateFeedIDs[1] = rateFeedID3;

    vm.expectRevert("Rate feed ID has not been added");
    breakerBox.setRateFeedDependencies(makeAddr("notRateFeed"), testRateFeedIDs);
  }

  function test_setRateFeedDependencies_whenRateFeedExists_shouldUpdateDependenciesAndEmit() public {
    address[] memory testRateFeedIDs = new address[](2);
    testRateFeedIDs[0] = rateFeedID2;
    testRateFeedIDs[1] = rateFeedID3;

    vm.expectEmit(true, true, true, true);
    emit RateFeedDependenciesSet(rateFeedID1, testRateFeedIDs);
    breakerBox.setRateFeedDependencies(rateFeedID1, testRateFeedIDs);
    assertEq(breakerBox.rateFeedDependencies(rateFeedID1, 0), rateFeedID2);
    assertEq(breakerBox.rateFeedDependencies(rateFeedID1, 1), rateFeedID3);
  }

  function test_removeRateFeed_whenNotOwner_shouldRevert() public {
    vm.prank(notDeployer);
    vm.expectRevert("Ownable: caller is not the owner");
    breakerBox.removeRateFeed(rateFeedID1);
  }

  function test_removeRateFeed_whenRateFeedHasNotBeenAdded_shouldRevert() public {
    vm.expectRevert("Rate feed ID has not been added");
    breakerBox.removeRateFeed(rateFeedID3);
  }

  function test_removeRateFeed_whenRateFeedExists_shouldRemoveRateFeedFromArray() public {
    assertTrue(isRateFeed(rateFeedID1));
    breakerBox.removeRateFeed(rateFeedID1);
    assertFalse(isRateFeed(rateFeedID1));
  }

  function test_removeRateFeed_whenRateFeedExists_shouldRemoveFromRateFeedDependencies() public {
    address[] memory testRateFeedIDs = new address[](2);
    testRateFeedIDs[0] = rateFeedID2;
    testRateFeedIDs[1] = rateFeedID3;

    breakerBox.setRateFeedDependencies(rateFeedID1, testRateFeedIDs);
    assertEq(breakerBox.rateFeedDependencies(rateFeedID1, 0), rateFeedID2);
    assertEq(breakerBox.rateFeedDependencies(rateFeedID1, 1), rateFeedID3);

    assertTrue(isRateFeed(rateFeedID1));
    breakerBox.removeRateFeed(rateFeedID1);
    assertFalse(isRateFeed(rateFeedID1));
    vm.expectRevert();
    breakerBox.rateFeedDependencies(rateFeedID1, 0);
    vm.expectRevert();
    breakerBox.rateFeedDependencies(rateFeedID1, 1);
  }

  function test_removeRateFeed_whenRateFeedExists_shouldResetTradingModeInfoAndEmit() public {
    toggleAndAssertBreaker(address(mockBreaker1), rateFeedID1, true);
    breakerBox.setRateFeedTradingMode(rateFeedID1, 1);
    IBreakerBox.BreakerStatus memory status;

    uint256 tradingModeBefore = breakerBox.getRateFeedTradingMode(rateFeedID1);
    assertEq(tradingModeBefore, 1);
    assertTrue(isRateFeed(rateFeedID1));
    assertTrue(breakerBox.rateFeedStatus(rateFeedID1));
    status = breakerBox.rateFeedBreakerStatus(rateFeedID1, address(mockBreaker1));
    assertTrue(status.enabled);

    vm.expectEmit(true, true, true, true);
    emit RateFeedRemoved(rateFeedID1);
    breakerBox.removeRateFeed(rateFeedID1);

    uint256 tradingModeAfter = breakerBox.rateFeedTradingMode(rateFeedID1);
    assertEq(tradingModeAfter, 0);
    assertFalse(isRateFeed(rateFeedID1));
    assertFalse(breakerBox.rateFeedStatus(rateFeedID1));
    status = breakerBox.rateFeedBreakerStatus(rateFeedID1, address(mockBreaker1));
    assertFalse(status.enabled);
  }

  function test_setRateFeedTradingMode_whenNotOwner_shouldRevert() public {
    vm.prank(notDeployer);
    vm.expectRevert("Ownable: caller is not the owner");
    breakerBox.setRateFeedTradingMode(rateFeedID1, 9);
  }

  function test_setRateFeedTradingMode_whenRateFeedHasNotBeenAdded_ShouldRevert() public {
    vm.expectRevert("Rate feed ID has not been added");
    breakerBox.setRateFeedTradingMode(rateFeedID3, 1);
  }

  function test_setRateFeedTradingMode_whenRateFeedExists_shouldUpdateAndEmit() public {
    uint256 tradingModeBefore = breakerBox.getRateFeedTradingMode(rateFeedID1);
    assertEq(tradingModeBefore, 0);

    vm.expectEmit(true, true, true, true);
    emit TradingModeUpdated(rateFeedID1, 1);
    breakerBox.setRateFeedTradingMode(rateFeedID1, 1);

    uint256 tradingModeAfter = breakerBox.getRateFeedTradingMode(rateFeedID1);
    assertEq(tradingModeAfter, 1);
  }

  function test_getRateFeedTradingMode_whenRateFeedHasNotBeenAdded_shouldRevert() public {
    vm.expectRevert("Rate feed ID has not been added");
    breakerBox.getRateFeedTradingMode(makeAddr("notRateFeed"));
  }

  function test_getRateFeedTradingMode_whenNoDependencies_shouldReturnRateFeedsTradingMode() public {
    breakerBox.setRateFeedTradingMode(rateFeedID1, 8);
    uint256 tradingMode = breakerBox.getRateFeedTradingMode(rateFeedID1);
    assertEq(tradingMode, 8);
  }

  function test_getRateFeedTradingMode_whenMultipleDependencies_shouldAggregateTradingModes() public {
    sortedOracles.addOracle(rateFeedID3, makeAddr("oracleClient"));
    breakerBox.addRateFeed(rateFeedID3);
    address[] memory testRateFeedIDs = new address[](2);
    testRateFeedIDs[0] = rateFeedID2;
    testRateFeedIDs[1] = rateFeedID3;
    breakerBox.setRateFeedDependencies(rateFeedID1, testRateFeedIDs);

    breakerBox.setRateFeedTradingMode(rateFeedID1, 0);
    breakerBox.setRateFeedTradingMode(rateFeedID2, 1);
    breakerBox.setRateFeedTradingMode(rateFeedID3, 2);
    assertEq(uint256(breakerBox.getRateFeedTradingMode(rateFeedID1)), (0 | 1 | 2));

    breakerBox.setRateFeedTradingMode(rateFeedID1, 1);
    breakerBox.setRateFeedTradingMode(rateFeedID2, 1);
    breakerBox.setRateFeedTradingMode(rateFeedID3, 1);
    assertEq(uint256(breakerBox.getRateFeedTradingMode(rateFeedID1)), (1 | 1 | 1));
  }

  /* ---------- Sorted Oracles ---------- */

  function test_setSortedOracles_whenNotOwner_shouldRevert() public {
    vm.prank(notDeployer);
    vm.expectRevert("Ownable: caller is not the owner");
    breakerBox.setSortedOracles(ISortedOracles(address(0)));
  }

  function test_setSortedOracles_whenAddressIsZero_shouldRevert() public {
    vm.expectRevert("SortedOracles address must be set");
    breakerBox.setSortedOracles(ISortedOracles(address(0)));
  }

  function test_setSortedOracles_whenValidOracle_shouldUpdateAndEmit() public {
    address newSortedOracles = makeAddr("newSortedOracles");
    vm.expectEmit(true, true, true, true);
    emit SortedOraclesUpdated(newSortedOracles);

    breakerBox.setSortedOracles(ISortedOracles(newSortedOracles));

    assertEq(address(breakerBox.sortedOracles()), newSortedOracles);
  }
}

contract BreakerBoxTest_checkAndSetBreakers is BreakerBoxTest {
  function test_checkAndSetBreakers_whenCallerIsNotSortedOraclesOrBreakerBox_shouldRevert() public {
    vm.expectRevert("Caller must be the SortedOracles contract");
    breakerBox.checkAndSetBreakers(rateFeedID1);
  }

  function test_checkAndSetBreakers_whenRateFeedIsNotInDefaultModeAndCooldownNotPassed_shouldEmitNotCool() public {
    setUpBreaker(mockBreaker3, 3, 3600, false, true);
    toggleAndAssertBreaker(address(mockBreaker3), rateFeedID1, true);
    vm.prank(address(sortedOracles));
    breakerBox.checkAndSetBreakers(rateFeedID1);

    skip(3599);

    vm.expectCall(address(mockBreaker3), abi.encodeWithSelector(mockBreaker3.getCooldown.selector));
    vm.expectEmit(true, true, true, true);
    emit ResetAttemptNotCool(rateFeedID1, address(mockBreaker3));

    vm.prank(address(sortedOracles));
    breakerBox.checkAndSetBreakers(rateFeedID1);
    assertEq(uint256(breakerBox.getRateFeedTradingMode(rateFeedID1)), 3);
  }

  function test_checkAndSetBreakers_whenRateFeedIsNotInDefaultModeAndCantReset_shouldEmitCriteriaFail() public {
    setUpBreaker(mockBreaker3, 3, 3600, false, true);
    toggleAndAssertBreaker(address(mockBreaker3), rateFeedID1, true);
    vm.prank(address(sortedOracles));
    breakerBox.checkAndSetBreakers(rateFeedID1);

    skip(3600);
    vm.expectCall(address(mockBreaker3), abi.encodeWithSelector(mockBreaker3.getCooldown.selector));
    vm.expectCall(address(mockBreaker3), abi.encodeWithSelector(mockBreaker3.shouldReset.selector, rateFeedID1));
    vm.expectEmit(true, true, true, true);
    emit ResetAttemptCriteriaFail(rateFeedID1, address(mockBreaker3));

    vm.prank(address(sortedOracles));
    breakerBox.checkAndSetBreakers(rateFeedID1);
    assertEq(uint256(breakerBox.getRateFeedTradingMode(rateFeedID1)), 3);
  }

  function test_checkAndSetBreakers_whenRateFeedIsNotInDefaultModeAndCanReset_shouldResetMode() public {
    setUpBreaker(mockBreaker3, 3, 3600, true, true);
    toggleAndAssertBreaker(address(mockBreaker3), rateFeedID1, true);
    vm.prank(address(sortedOracles));
    breakerBox.checkAndSetBreakers(rateFeedID1);

    skip(3600);

    vm.expectCall(address(mockBreaker3), abi.encodeWithSelector(mockBreaker3.getCooldown.selector));
    vm.expectCall(address(mockBreaker3), abi.encodeWithSelector(mockBreaker3.shouldReset.selector, rateFeedID1));
    vm.expectEmit(true, true, true, true);
    emit ResetSuccessful(rateFeedID1, address(mockBreaker3));

    vm.prank(address(sortedOracles));
    breakerBox.checkAndSetBreakers(rateFeedID1);
    assertEq(uint256(breakerBox.getRateFeedTradingMode(rateFeedID1)), 0);
  }

  function test_checkAndSetBreakers_whenRateFeedIsNotInDefaultModeAndNoBreakerCooldown_shouldReturnCorrectModeAndEmit()
    public
  {
    setUpBreaker(mockBreaker3, 3, 0, true, true);
    toggleAndAssertBreaker(address(mockBreaker3), rateFeedID1, true);

    vm.prank(address(sortedOracles));
    breakerBox.checkAndSetBreakers(rateFeedID1);

    skip(3600);

    vm.expectCall(address(mockBreaker3), abi.encodeWithSelector(mockBreaker3.getCooldown.selector));
    vm.expectEmit(true, true, true, true);
    emit ResetAttemptNotCool(rateFeedID1, address(mockBreaker3));

    vm.prank(address(sortedOracles));
    breakerBox.checkAndSetBreakers(rateFeedID1);
    assertEq(uint256(breakerBox.getRateFeedTradingMode(rateFeedID1)), 3);
  }

  function test_checkAndSetBreakers_whenNoBreakersAreTripped_shouldReturnDefaultMode() public {
    setUpBreaker(mockBreaker3, 3, 3600, false, false);
    setUpBreaker(mockBreaker4, 1, 3600, false, false);
    toggleAndAssertBreaker(address(mockBreaker3), rateFeedID1, true);
    toggleAndAssertBreaker(address(mockBreaker4), rateFeedID1, true);

    assertEq(breakerBox.getRateFeedTradingMode(rateFeedID1), 0);

    vm.expectCall(
      address(mockBreaker3),
      abi.encodeWithSelector(mockBreaker3.shouldTrigger.selector, address(rateFeedID1))
    );
    vm.expectCall(
      address(mockBreaker4),
      abi.encodeWithSelector(mockBreaker4.shouldTrigger.selector, address(rateFeedID1))
    );
    vm.prank(address(sortedOracles));
    breakerBox.checkAndSetBreakers(rateFeedID1);

    assertEq(uint256(breakerBox.getRateFeedTradingMode(rateFeedID1)), 0);
  }

  function test_checkAndSetBreakers_whenABreakerIsTripped_shouldSetModeAndEmit() public {
    vm.warp(1672527600); // 2023-01-01 00:00:00
    setUpBreaker(mockBreaker3, 3, 3600, false, false);
    toggleAndAssertBreaker(address(mockBreaker3), rateFeedID1, true);
    IBreakerBox.BreakerStatus memory status;

    assertEq(breakerBox.getRateFeedTradingMode(rateFeedID1), 0);
    status = breakerBox.rateFeedBreakerStatus(rateFeedID1, address(mockBreaker3));
    assertEq(status.tradingMode, 0);

    mockBreaker3.setTrigger(true);
    vm.expectCall(
      address(mockBreaker3),
      abi.encodeWithSelector(mockBreaker3.shouldTrigger.selector, address(rateFeedID1))
    );
    vm.expectEmit(true, true, true, true);
    emit BreakerTripped(address(mockBreaker3), rateFeedID1);
    vm.prank(address(sortedOracles));
    breakerBox.checkAndSetBreakers(rateFeedID1);

    assertEq(breakerBox.getRateFeedTradingMode(rateFeedID1), 3);
    status = breakerBox.rateFeedBreakerStatus(rateFeedID1, address(mockBreaker3));
    assertEq(status.tradingMode, 3);
    assertEq(status.lastUpdatedTime, 1672527600);
    assertTrue(status.enabled);
  }

  function test_checkAndSetBreakers_whenABreakerIsNotEnabled_shouldNotTrigger() public {
    setUpBreaker(mockBreaker3, 3, 3600, false, true);
    IBreakerBox.BreakerStatus memory status;

    assertTrue(breakerBox.isBreaker(address(mockBreaker3)));
    assertEq(breakerBox.getRateFeedTradingMode(rateFeedID1), 0);
    status = breakerBox.rateFeedBreakerStatus(rateFeedID1, address(mockBreaker3));
    assertFalse(status.enabled);

    vm.prank(address(sortedOracles));
    breakerBox.checkAndSetBreakers(rateFeedID1);
    assertEq(uint256(breakerBox.getRateFeedTradingMode(rateFeedID1)), 0);
  }

  function test_checkAndSetBreakers_whenCooldownOneSecond_shouldSetStatusCorrectly() public {
    vm.warp(1672527600); // 2023-01-01 00:00:00
    setUpBreaker(mockBreaker3, 2, 4, false, true);
    toggleAndAssertBreaker(address(mockBreaker3), rateFeedID1, true);
    vm.prank(address(sortedOracles));
    breakerBox.checkAndSetBreakers(rateFeedID1);

    uint8 tradingMode;
    IBreakerBox.BreakerStatus memory status;

    status = breakerBox.rateFeedBreakerStatus(rateFeedID1, address(mockBreaker3));
    tradingMode = breakerBox.getRateFeedTradingMode(rateFeedID1);
    assertEq(status.tradingMode, 2);
    assertEq(status.lastUpdatedTime, 1672527600);
    assertTrue(status.enabled);
    assertEq(tradingMode, 2);

    vm.warp(1672527605); // 2023-01-01 00:00:05
    mockBreaker3.setTrigger(false);
    mockBreaker3.setReset(true);

    vm.prank(address(sortedOracles));
    breakerBox.checkAndSetBreakers(rateFeedID1);

    status = breakerBox.rateFeedBreakerStatus(rateFeedID1, address(mockBreaker3));
    tradingMode = breakerBox.getRateFeedTradingMode(rateFeedID1);
    assertEq(status.tradingMode, 0);
    assertEq(status.lastUpdatedTime, 1672527605);
    assertTrue(status.enabled);
    assertEq(tradingMode, 0);

    vm.warp(1672527610); // 2023-01-01 00:00:10
    mockBreaker3.setTrigger(true);
    mockBreaker3.setReset(false);

    vm.prank(address(sortedOracles));
    breakerBox.checkAndSetBreakers(rateFeedID1);

    status = breakerBox.rateFeedBreakerStatus(rateFeedID1, address(mockBreaker3));
    tradingMode = breakerBox.getRateFeedTradingMode(rateFeedID1);
    assertEq(status.tradingMode, 2);
    assertEq(status.lastUpdatedTime, 1672527610);
    assertTrue(status.enabled);
    assertEq(tradingMode, 2);
  }

  function test_checkAndSetBreakers_whenCooldownTenSeconds_shouldSetStatusCorrectly() public {
    vm.warp(1672527600); // 2023-01-01 00:00:00
    setUpBreaker(mockBreaker3, 2, 9, false, true);
    toggleAndAssertBreaker(address(mockBreaker3), rateFeedID1, true);
    uint8 tradingMode;
    IBreakerBox.BreakerStatus memory status;

    vm.prank(address(sortedOracles));
    breakerBox.checkAndSetBreakers(rateFeedID1);

    status = breakerBox.rateFeedBreakerStatus(rateFeedID1, address(mockBreaker3));
    tradingMode = breakerBox.getRateFeedTradingMode(rateFeedID1);
    assertEq(status.tradingMode, 2);
    assertEq(status.lastUpdatedTime, 1672527600);
    assertTrue(status.enabled);
    assertEq(tradingMode, 2);

    vm.warp(1672527610); // 2023-01-01 00:00:10
    mockBreaker3.setTrigger(false);
    mockBreaker3.setReset(true);
    vm.prank(address(sortedOracles));
    breakerBox.checkAndSetBreakers(rateFeedID1);

    status = breakerBox.rateFeedBreakerStatus(rateFeedID1, address(mockBreaker3));
    tradingMode = breakerBox.getRateFeedTradingMode(rateFeedID1);
    assertEq(status.tradingMode, 0);
    assertEq(status.lastUpdatedTime, 1672527610);
    assertTrue(status.enabled);
    assertEq(tradingMode, 0);

    vm.warp(1672527620); // 2023-01-01 00:00:20
    mockBreaker3.setTrigger(true);
    mockBreaker3.setReset(false);

    vm.prank(address(sortedOracles));
    breakerBox.checkAndSetBreakers(rateFeedID1);

    status = breakerBox.rateFeedBreakerStatus(rateFeedID1, address(mockBreaker3));
    tradingMode = breakerBox.getRateFeedTradingMode(rateFeedID1);
    assertEq(status.tradingMode, 2);
    assertEq(status.lastUpdatedTime, 1672527620);
    assertTrue(status.enabled);
    assertEq(tradingMode, 2);
  }

  function test_checkAndSetBreakers_whenMultipleBreakersAreEnabled_shouldCalculateTradingModeCorrectly() public {
    uint8 tradingModeBreaker3 = 1;
    uint8 tradingModeBreaker4 = 2;
    setUpBreaker(mockBreaker3, tradingModeBreaker3, 60, false, false);
    setUpBreaker(mockBreaker4, tradingModeBreaker4, 60, false, false);
    breakerBox.toggleBreaker(address(mockBreaker3), rateFeedID1, true);
    breakerBox.toggleBreaker(address(mockBreaker4), rateFeedID1, true);

    uint256 tradingModeBefore = breakerBox.getRateFeedTradingMode(rateFeedID1);
    assertEq(tradingModeBefore, 0);

    mockBreaker3.setTrigger(true);
    mockBreaker4.setTrigger(true);

    vm.prank(address(sortedOracles));
    breakerBox.checkAndSetBreakers(rateFeedID1);

    uint256 tradingModeAfter = breakerBox.getRateFeedTradingMode(rateFeedID1);
    assertEq(tradingModeAfter, tradingModeBreaker3 | tradingModeBreaker4);

    mockBreaker3.setTrigger(false);
    mockBreaker3.setReset(true);
    skip(60);

    vm.prank(address(sortedOracles));
    breakerBox.checkAndSetBreakers(rateFeedID1);
    uint256 tradingModeAfter2 = breakerBox.getRateFeedTradingMode(rateFeedID1);
    assertEq(tradingModeAfter2, tradingModeBreaker4);
  }
}
