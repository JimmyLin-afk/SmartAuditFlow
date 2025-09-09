// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;
// solhint-disable func-name-mixedcase, contract-name-camelcase

import { GovernanceTest } from "../GovernanceTest.sol";
import { LibBrokenLine } from "contracts/governance/locking/libs/LibBrokenLine.sol";

contract LibBrokenLine_Test is GovernanceTest {
  LibBrokenLine.BrokenLine public brokenLine;

  function assertLineEq(LibBrokenLine.Line memory a, LibBrokenLine.Line memory b) internal pure {
    assertEq(a.start, b.start);
    assertEq(a.bias, b.bias);
    assertEq(a.slope, b.slope);
    assertEq(a.cliff, b.cliff);
  }

  function blockNumber() internal view returns (uint32) {
    return uint32(block.number);
  }

  function test_addOneLine_whenSlopeZero_shouldRevert() public {
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(1, 100, 0, 3);
    vm.expectRevert("Slope == 0, unacceptable value for slope");
    LibBrokenLine.addOneLine(brokenLine, 0, line, blockNumber());
  }

  function test_addOneLine_whenSlopeLargerThanBias_shouldRevert() public {
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(1, 100, 101, 3);
    vm.expectRevert("Slope > bias, unacceptable value for slope");
    LibBrokenLine.addOneLine(brokenLine, 0, line, blockNumber());
  }

  function test_addOneLine_whenLineWithIdAlreadyAdded_shouldRevert() public {
    uint256 id = 0;
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(1, 100, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, id, line, blockNumber());
    vm.expectRevert("Line with given id is already exist");
    LibBrokenLine.addOneLine(brokenLine, id, line, blockNumber());
  }

  function test_addOneLine_whenValidCall_shouldSaveSnapShot() public {
    // history should be empty
    assertEq(brokenLine.history.length, 0);

    uint256 id = 0;
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory firstLine = LibBrokenLine.Line(1, 100, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, id, firstLine, blockNumber());

    // verify point in history was saved correctly
    assertEq(brokenLine.history.length, 1);
    LibBrokenLine.Point memory point = brokenLine.history[0];
    assertEq(point.blockNumber, blockNumber());
    assertEq(point.bias, 100);
    assertEq(point.slope, 0);
    assertEq(point.epoch, 1);
  }

  function test_addOneLine_whenOnlyLine_shouldUpdateStruct() public {
    uint256 id = 0;

    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory firstLine = LibBrokenLine.Line(1, 100, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, id, firstLine, blockNumber());

    // verify line attributes were updated correctly
    LibBrokenLine.Line memory initialLine = brokenLine.initial;
    assertEq(initialLine.start, 1);
    assertEq(initialLine.bias, 100);
    assertEq(initialLine.slope, 0);
    assertLineEq(brokenLine.initiatedLines[id], firstLine);

    // verify slope changes were set correctly
    assertEq(brokenLine.slopeChanges[1], 0);
    assertEq(brokenLine.slopeChanges[3], 10); // after cliff ended slope starts at -10 per week
    assertEq(brokenLine.slopeChanges[13], -10); // once line ends, slope changes back to 0
  }

  function test_addOneLine_whenNoCliff_shouldSetSlope() public {
    uint256 id = 0;
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory firstLine = LibBrokenLine.Line(1, 100, 10, 0);
    LibBrokenLine.addOneLine(brokenLine, id, firstLine, blockNumber());

    // verify line attributes were updated correctly
    LibBrokenLine.Line memory initialLine = brokenLine.initial;
    assertEq(initialLine.start, 1);
    assertEq(initialLine.bias, 100);
    assertEq(initialLine.slope, 10);
    assertLineEq(brokenLine.initiatedLines[id], firstLine);

    // verify slope changes were set correctly
    assertEq(brokenLine.slopeChanges[10], -10); // once line ends, slope changes back to 0
  }

  function test_addOneLine_whenMultipleLines_shouldAggregateLines() public {
    // add first Line
    uint256 id1 = 0;
    // Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory firstLine = LibBrokenLine.Line(1, 100, 10, 2);
    LibBrokenLine.addOneLine(brokenLine, id1, firstLine, blockNumber());

    // verify line attributes were updated correctly
    LibBrokenLine.Line memory initialLine = brokenLine.initial;
    assertEq(initialLine.start, 1);
    assertEq(initialLine.bias, 100);
    assertEq(initialLine.slope, 0);
    assertLineEq(brokenLine.initiatedLines[id1], firstLine);

    // add second Line
    uint256 id2 = 1;
    // Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory secondLine = LibBrokenLine.Line(1, 200, 20, 2);
    LibBrokenLine.addOneLine(brokenLine, id2, secondLine, blockNumber());

    // verify line attributes were updated correctly
    initialLine = brokenLine.initial;
    assertEq(initialLine.start, 1);
    assertEq(initialLine.bias, 100 + 200);
    assertEq(initialLine.slope, 0);
    assertLineEq(brokenLine.initiatedLines[id2], secondLine);

    // verify slope changes were set correctly
    assertEq(brokenLine.slopeChanges[2], 10 + 20); // after both cliffs end slope starts at -30 per week
    assertEq(brokenLine.slopeChanges[12], -10 - 20); // once lines end, slope changes back to 0

    // jump to week 3
    LibBrokenLine.update(brokenLine, 3);

    // verify line attributes were updated correctly
    initialLine = brokenLine.initial;
    assertEq(initialLine.start, 3);
    assertEq(initialLine.bias, 100 + 200);
    assertEq(initialLine.slope, 10 + 20);
  }

  function test_addOneLine_whenBiasModSlopeIsNotZero_shouldSetAdditionalSlopeChanges() public {
    // add first line
    uint256 id1 = 0;
    // Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory firstLine = LibBrokenLine.Line(1, 102, 10, 2);
    LibBrokenLine.addOneLine(brokenLine, id1, firstLine, blockNumber());

    // verify line attributes were updated correctly
    LibBrokenLine.Line memory initialLine = brokenLine.initial;
    assertEq(initialLine.start, 1);
    assertEq(initialLine.bias, 102);
    assertEq(initialLine.slope, 0);
    assertLineEq(brokenLine.initiatedLines[id1], firstLine);

    // verify slope changes were set correctly
    assertEq(brokenLine.slopeChanges[2], 10); // after cliff ends slope starts at -10 per week
    assertEq(brokenLine.slopeChanges[12], -8); // Since 102%10 = 2, slope changes to -2 per week
    assertEq(brokenLine.slopeChanges[13], -2); // once line ends, slope changes back to 0
  }

  function test_actualValue_whenToTimeInPastAndInCliffPeriod_shouldReturnBias() public {
    uint256 initialBlock = 100;
    vm.roll(initialBlock);
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(1, 100, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, 0, line, blockNumber());

    vm.roll(initialBlock + 50);
    LibBrokenLine.update(brokenLine, 20);

    uint96 bias = LibBrokenLine.actualValue(brokenLine, 1, uint32(initialBlock));

    assertEq(bias, 100);
  }

  function test_actualValue_whenToTimeInPastAndInSlopePeriod_shouldReturnCorrectValue() public {
    uint256 initialBlock = 100;
    vm.roll(initialBlock);

    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(1, 100, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, 0, line, blockNumber());

    vm.roll(initialBlock + 50);
    LibBrokenLine.update(brokenLine, 5);
    uint96 actualBias = brokenLine.initial.bias;

    vm.roll(initialBlock + 100);
    LibBrokenLine.update(brokenLine, 20);

    uint96 bias = LibBrokenLine.actualValue(brokenLine, 5, uint32(initialBlock + 50));
    assertEq(bias, actualBias);
  }

  function test_actualValue_whenToTimeInPastAndInReminderPeriod_shouldReturnReminder() public {
    uint256 initialBlock = 100;
    vm.roll(initialBlock);

    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(1, 102, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, 0, line, blockNumber());

    vm.roll(initialBlock + 50);
    LibBrokenLine.update(brokenLine, 14);
    uint96 actualBias = brokenLine.initial.bias;

    vm.roll(initialBlock + 100);
    LibBrokenLine.update(brokenLine, 20);

    uint96 bias = LibBrokenLine.actualValue(brokenLine, 14, uint32(initialBlock + 50));
    assertEq(bias, 2);
    assertEq(bias, actualBias);
  }

  function test_actualValue_whenToTimeInPastAndInFinishedPeriod_shouldReturnZero() public {
    uint256 initialBlock = 100;
    vm.roll(initialBlock);

    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(1, 102, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, 0, line, blockNumber());

    vm.roll(initialBlock + 50);
    LibBrokenLine.update(brokenLine, 16);
    uint96 actualBias = brokenLine.initial.bias;

    vm.roll(initialBlock + 100);
    LibBrokenLine.update(brokenLine, 20);

    uint96 bias = LibBrokenLine.actualValue(brokenLine, 16, uint32(initialBlock + 50));
    assertEq(bias, 0);
    assertEq(bias, actualBias);
  }

  function test_actualValue_whenToTimeIsNowAndInCliffPeriod_shouldReturnBias() public {
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(1, 100, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, 0, line, blockNumber());

    uint96 bias = LibBrokenLine.actualValue(brokenLine, 1, blockNumber());
    uint96 actualBias = brokenLine.initial.bias;
    assertEq(bias, actualBias);
  }

  function test_actualValue_whenToTimeIsBeforeLineStart_shouldReturnZero() public {
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(2, 100, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, 5, line, blockNumber());

    uint96 bias = LibBrokenLine.actualValue(brokenLine, 1, blockNumber() - 1);
    uint96 actualBias = 0;
    assertEq(bias, actualBias);
  }

  function test_actualValue_whenToTimeIsNowAndInSlopePeriod_shouldReturnCorrectValue() public {
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(1, 100, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, 0, line, blockNumber());
    LibBrokenLine.update(brokenLine, 5);

    uint96 bias = LibBrokenLine.actualValue(brokenLine, 5, blockNumber());
    uint96 actualBias = brokenLine.initial.bias;
    assertEq(actualBias, 90);
    assertEq(bias, actualBias);
  }

  function test_actualValue_whenToTimeIsNowAndInReminderWeek_shouldReturnReminder() public {
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(1, 102, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, 0, line, blockNumber());
    LibBrokenLine.update(brokenLine, 14);

    uint96 bias = LibBrokenLine.actualValue(brokenLine, 14, blockNumber());
    uint96 actualBias = brokenLine.initial.bias;
    assertEq(actualBias, 102 % 10);
    assertEq(bias, actualBias);
  }

  function test_actualValue_whenToTimeIsNowAndInFinishedPeriod_shouldReturnZero() public {
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(1, 102, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, 0, line, blockNumber());
    LibBrokenLine.update(brokenLine, 15);

    uint96 bias = LibBrokenLine.actualValue(brokenLine, 15, blockNumber());
    uint96 actualBias = brokenLine.initial.bias;
    assertEq(actualBias, 0);
    assertEq(bias, actualBias);
  }

  function test_actualValue_whenToTimeInFutureAndInCliffPeriod_shouldReturnBias() public {
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(1, 100, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, 0, line, blockNumber());

    uint96 bias = LibBrokenLine.actualValue(brokenLine, 3, blockNumber());
    uint96 expectedBias = brokenLine.initial.bias;
    assertEq(expectedBias, 100);
    assertEq(bias, expectedBias);
  }

  function test_actualValue_whenToTimeInFutureAndInSlopePeriod_shouldReturnCorrectValue() public {
    uint32 start = 1;
    uint96 bias = 100;
    uint96 slope = 10;
    uint32 cliff = 3;
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(start, bias, slope, cliff);
    LibBrokenLine.addOneLine(brokenLine, 0, line, blockNumber());

    for (uint256 i = 1; i <= bias / slope; i++) {
      uint32 toTime = start + cliff + uint32(i);
      uint96 expectedBias = bias - slope * uint96(i);

      uint96 actualBias = LibBrokenLine.actualValue(brokenLine, toTime, blockNumber());
      assertEq(actualBias, expectedBias);
    }
  }

  function test_actualValue_whenToTimeInFutureAndInReminderWeek_shouldReturnReminder() public {
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(1, 102, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, 0, line, blockNumber());

    uint96 bias = LibBrokenLine.actualValue(brokenLine, 14, blockNumber());
    uint96 expectedBias = 102 % 10;
    assertEq(bias, expectedBias);
  }

  function test_actualValue_whenToTimeInFutureAndInFinishedPeriod_shouldReturnZero() public {
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(1, 102, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, 0, line, blockNumber());

    uint96 bias = LibBrokenLine.actualValue(brokenLine, 15, blockNumber());

    uint96 expectedBias = 0;
    assertEq(bias, expectedBias);
  }

  function test_update_whenLineUpToDate_shouldNotUpdate() public {
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(1, 100, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, 0, line, blockNumber());

    LibBrokenLine.Line memory initialBefore = brokenLine.initial;

    LibBrokenLine.update(brokenLine, 1);

    LibBrokenLine.Line memory initialAfter = brokenLine.initial;
    assertLineEq(initialBefore, initialAfter);
  }

  function test_update_whenToTimeIsInPast_shouldRevert() public {
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(1, 100, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, 0, line, blockNumber());

    vm.expectRevert("can't update BrokenLine for past time");
    LibBrokenLine.update(brokenLine, 0);
  }

  function test_update_whenNegativeSlope_shouldRevert() public {
    uint96 slope = 10;
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(1, 100, slope, 0);
    LibBrokenLine.addOneLine(brokenLine, 0, line, blockNumber());

    brokenLine.slopeChanges[1] = -(int96(slope) + 1);
    vm.expectRevert("slope < 0, something wrong with slope");
    LibBrokenLine.update(brokenLine, 2);
  }

  function test_update_whenToTimeInCliffPeriod_shouldUpdateOnlyStart() public {
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(1, 100, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, 0, line, blockNumber());

    LibBrokenLine.update(brokenLine, 3);
    assertEq(brokenLine.initial.bias, 100);
    assertEq(brokenLine.initial.slope, 0);
    assertEq(brokenLine.initial.start, 3);
  }

  function test_update_whenToTimeInSlopePeriod_shouldSetAndAppySlope() public {
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(1, 100, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, 0, line, blockNumber());

    LibBrokenLine.update(brokenLine, 5);
    assertEq(brokenLine.initial.bias, 90);
    assertEq(brokenLine.initial.slope, 10);
    assertEq(brokenLine.initial.start, 5);
  }

  function test_update_whenToTimeInReminderPeriod_shouldSetAndAppySlope() public {
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(1, 102, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, 0, line, blockNumber());

    LibBrokenLine.update(brokenLine, 14);
    assertEq(brokenLine.initial.bias, 2);
    assertEq(brokenLine.initial.slope, 2);
    assertEq(brokenLine.initial.start, 14);
  }

  function test_update_whenToTimeInEndPeriod_shouldSetAndAppySlope() public {
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(1, 102, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, 0, line, blockNumber());

    LibBrokenLine.update(brokenLine, 15);
    assertEq(brokenLine.initial.bias, 0);
    assertEq(brokenLine.initial.slope, 0);
    assertEq(brokenLine.initial.start, 15);
  }

  function test_update_whenMultipleLines_shouldSetAndAppySlope() public {
    //Line(start, bias, slope, cliff)
    uint32 otherStart = 1;
    uint96 otherBias = 200;
    uint96 otherSlope = 10;
    uint32 otherCliff = 0;
    LibBrokenLine.Line memory line = LibBrokenLine.Line(otherStart, otherBias, otherSlope, otherCliff);
    LibBrokenLine.addOneLine(brokenLine, 0, line, blockNumber());

    uint32 testeeStart = 1;
    uint96 testeeBias = 102;
    uint96 testeeSlope = 10;
    uint32 testeeCliff = 3;
    LibBrokenLine.Line memory testee = LibBrokenLine.Line(testeeStart, testeeBias, testeeSlope, testeeCliff);
    LibBrokenLine.addOneLine(brokenLine, 1, testee, blockNumber());

    uint32 index;
    uint96 newBias = otherBias + testeeBias;
    for (uint32 i = 2; i <= testeeStart + testeeCliff + testeeBias / testeeSlope; i++) {
      newBias = newBias - otherSlope;
      if (i > testeeStart + testeeCliff) {
        //testee cliff period has ended
        newBias = newBias - testeeSlope;
      }
      LibBrokenLine.update(brokenLine, i);

      assertEq(brokenLine.initial.bias, newBias);
      index = i;
    }

    //testee reminder period of bias % slope
    newBias = newBias - (testeeBias % testeeSlope) - otherSlope;
    LibBrokenLine.update(brokenLine, index + 1);
    assertEq(brokenLine.initial.bias, newBias);
  }

  function test_remove_whenLineDoesNotExist_shouldRevert() public {
    vm.expectRevert("Removing Line, which not exists");
    LibBrokenLine.remove(brokenLine, 0, 1, blockNumber());
  }

  function test_remove_whenValidCall_shouldSaveSnapshot() public {
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(1, 100, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, 0, line, blockNumber());

    // history should have one entry from adding line
    assertEq(brokenLine.history.length, 1);

    LibBrokenLine.remove(brokenLine, 0, 1, blockNumber());

    // verify point in history was saved correctly
    assertEq(brokenLine.history.length, 2);
    LibBrokenLine.Point memory point = brokenLine.history[1];
    assertEq(point.blockNumber, blockNumber());
    assertEq(point.bias, 0);
    assertEq(point.slope, 0);
    assertEq(point.epoch, 1);
  }

  function test_remove_whenLineIsOutdated_shouldReturnZero() public {
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory line = LibBrokenLine.Line(1, 100, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, 0, line, blockNumber());
    (uint96 bias, uint96 slope, uint32 cliff) = LibBrokenLine.remove(brokenLine, 0, 14, blockNumber());
    assertEq(bias, 0);
    assertEq(slope, 0);
    assertEq(cliff, 0);
  }

  function test_remove_whenInCLiffPeriod_shouldUpdateLineCorrectly() public {
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory firstLine = LibBrokenLine.Line(1, 50, 5, 3);
    LibBrokenLine.addOneLine(brokenLine, 0, firstLine, blockNumber());
    LibBrokenLine.Line memory secondLine = LibBrokenLine.Line(1, 102, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, 1, secondLine, blockNumber());

    int96 slopeChangeStart = brokenLine.slopeChanges[3];
    assertEq(slopeChangeStart, 15);
    int96 slopeChangeEnd = brokenLine.slopeChanges[13];
    assertEq(slopeChangeEnd, -13);
    int96 slopeChangeReminder = brokenLine.slopeChanges[14];
    assertEq(slopeChangeReminder, -2);

    (uint96 bias, uint96 slope, uint32 cliff) = LibBrokenLine.remove(brokenLine, 1, 3, blockNumber());
    //curren state of second Line
    assertEq(bias, 102);
    assertEq(slope, 10);
    assertEq(cliff, 1);

    //brokenLine state
    assertEq(brokenLine.initiatedLines[1].bias, 0);
    LibBrokenLine.Line memory initialLine = brokenLine.initial;
    assertEq(initialLine.bias, 50);
    assertEq(initialLine.slope, 0);

    //brokenLine slopeChanges
    assertEq(brokenLine.slopeChanges[3], slopeChangeStart - 10);
    assertEq(brokenLine.slopeChanges[13], slopeChangeEnd + 8);
    assertEq(brokenLine.slopeChanges[14], slopeChangeReminder + 2);
  }

  function test_remove_whenInSlopePeriod_shouldUpdateLineCorrectly() public {
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory firstLine = LibBrokenLine.Line(1, 50, 5, 3);
    LibBrokenLine.addOneLine(brokenLine, 0, firstLine, blockNumber());
    LibBrokenLine.Line memory secondLine = LibBrokenLine.Line(1, 102, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, 1, secondLine, blockNumber());

    int96 slopeChangeStart = brokenLine.slopeChanges[3];
    assertEq(slopeChangeStart, 15);
    int96 slopeChangeEnd = brokenLine.slopeChanges[13];
    assertEq(slopeChangeEnd, -13);
    int96 slopeChangeReminder = brokenLine.slopeChanges[14];
    assertEq(slopeChangeReminder, -2);

    (uint96 bias, uint96 slope, uint32 cliff) = LibBrokenLine.remove(brokenLine, 1, 5, blockNumber());
    //curren state of second Line
    assertEq(bias, 92);
    assertEq(slope, 10);
    assertEq(cliff, 0);

    //brokenLine state
    assertEq(brokenLine.initiatedLines[1].bias, 0);
    LibBrokenLine.Line memory initialLine = brokenLine.initial;
    assertEq(initialLine.bias, 45);
    assertEq(initialLine.slope, 5);

    //brokenLine slopeChanges
    assertEq(brokenLine.slopeChanges[13], slopeChangeEnd + 8);
    assertEq(brokenLine.slopeChanges[14], slopeChangeReminder + 2);
  }

  function test_remove_whenInReminderPeriod_shouldUpdateLineCorrectly() public {
    //Line(start, bias, slope, cliff)
    LibBrokenLine.Line memory firstLine = LibBrokenLine.Line(1, 50, 5, 3);
    LibBrokenLine.addOneLine(brokenLine, 0, firstLine, blockNumber());
    LibBrokenLine.Line memory secondLine = LibBrokenLine.Line(1, 102, 10, 3);
    LibBrokenLine.addOneLine(brokenLine, 1, secondLine, blockNumber());

    int96 slopeChangeStart = brokenLine.slopeChanges[3];
    assertEq(slopeChangeStart, 15);
    int96 slopeChangeEnd = brokenLine.slopeChanges[13];
    assertEq(slopeChangeEnd, -13);
    int96 slopeChangeReminder = brokenLine.slopeChanges[14];
    assertEq(slopeChangeReminder, -2);

    (uint96 bias, uint96 slope, uint32 cliff) = LibBrokenLine.remove(brokenLine, 1, 14, blockNumber());
    //curren state of second Line
    assertEq(bias, 2);
    assertEq(slope, 2);
    assertEq(cliff, 0);

    //brokenLine state
    assertEq(brokenLine.initiatedLines[1].bias, 0);
    LibBrokenLine.Line memory initialLine = brokenLine.initial;
    assertEq(initialLine.bias, 0);
    assertEq(initialLine.slope, 0);

    //brokenLine slopeChanges
    assertEq(brokenLine.slopeChanges[14], slopeChangeReminder + 2);
  }
}
