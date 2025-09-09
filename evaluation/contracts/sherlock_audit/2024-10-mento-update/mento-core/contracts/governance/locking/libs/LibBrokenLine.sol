// SPDX-License-Identifier: MIT
// solhint-disable gas-custom-errors
pragma solidity 0.8.18;

import "./LibIntMapping.sol";

/**
 * Line describes a linear function, how the user's voice decreases from point (start, bias) with speed slope
 * BrokenLine - a curve that describes the curve of the change in the sum of votes of several users
 * This curve starts with a line (Line) and then, at any time, the slope can be changed.
 * All slope changes are stored in slopeChanges. The slope can always be reduced only, it cannot increase,
 * because users can only run out of lockup periods.
 **/

library LibBrokenLine {
  using LibIntMapping for mapping(uint256 => int96);
  /**
   * @dev Line describes a function
   * start - week number when line starts
   * bias - initial y value of the function
   * slope - absolute value of the slope applied during slope period
   * cliff - period where slope is 0 and bias is not changing
   */
  struct Line {
    uint32 start;
    uint96 bias;
    uint96 slope;
    uint32 cliff;
  }
  /**
   * @dev Point describes a point in time
   * blockNumber - block number when the point was saved
   * bias - y value of the line at snapshot time
   * slope - current slope at snapshot time
   * epoch - week number when the point was saved
   */
  struct Point {
    uint32 blockNumber;
    uint96 bias;
    uint96 slope;
    uint32 epoch;
  }
  /**
   * @dev BrokenLine struct is used to track and document a function
   * slopeChanges - a map of changes in slope at a certain time
   * initiatedLines - a map of initiated (successfully added) Lines
   * history - a list of snapshots of the function
   * initial - the line representing the aggregated lines from the initiatedLines mapping
   */
  struct BrokenLine {
    mapping(uint256 => int96) slopeChanges; //change of slope applies to the next time point
    mapping(uint256 => Line) initiatedLines; //initiated (successfully added) Lines
    Point[] history;
    Line initial;
  }

  /**
   * @notice Adds a line to a given BrokenLine struct
   * @dev Add Line, save data in LineData. Run update BrokenLine, require:
   *      1. slope != 0, slope <= bias
   *      2. line not exists
   * @param brokenLine the BrokenLine struct to add the line to
   * @param id the id of the line to add
   * @param line the line to add
   */
  function _addOneLine(BrokenLine storage brokenLine, uint256 id, Line memory line) internal {
    require(line.slope != 0, "Slope == 0, unacceptable value for slope");
    require(line.slope <= line.bias, "Slope > bias, unacceptable value for slope");
    require(brokenLine.initiatedLines[id].bias == 0, "Line with given id is already exist");
    brokenLine.initiatedLines[id] = line;

    update(brokenLine, line.start);
    brokenLine.initial.bias = brokenLine.initial.bias + (line.bias);
    //save bias for history in line.start minus one
    uint32 lineStartMinusOne = line.start - 1;
    //period is time without tail
    uint32 period = uint32(line.bias / (line.slope));

    if (line.cliff == 0) {
      //no cliff, need to increase brokenLine.initial.slope write now
      brokenLine.initial.slope = brokenLine.initial.slope + (line.slope);
      //no cliff, save slope in history in time minus one
      brokenLine.slopeChanges.addToItem(lineStartMinusOne, safeInt(line.slope));
    } else {
      //cliffEnd finish in lineStart minus one plus cliff
      uint32 cliffEnd = lineStartMinusOne + (line.cliff);
      //save slope in history in cliffEnd
      brokenLine.slopeChanges.addToItem(cliffEnd, safeInt(line.slope));
      period = period + (line.cliff);
    }

    int96 mod = safeInt(line.bias % (line.slope));
    uint32 endPeriod = line.start + (period);
    uint32 endPeriodMinus1 = endPeriod - 1;
    brokenLine.slopeChanges.subFromItem(endPeriodMinus1, safeInt(line.slope) - (mod));
    brokenLine.slopeChanges.subFromItem(endPeriod, mod);
  }

  /**
   * @notice Adds a line to a given BrokenLine struct and saves a snapshot
   * @param brokenLine the BrokenLine struct to add the line to
   * @param id the id of the line to add
   * @param line the line to add
   * @param blockNumber the block number when the line is added
   */
  function addOneLine(BrokenLine storage brokenLine, uint256 id, Line memory line, uint32 blockNumber) internal {
    _addOneLine(brokenLine, id, line);
    saveSnapshot(brokenLine, line.start, blockNumber);
  }

  /**
   * @notice Removes a line from a given BrokenLine struct
   * @param brokenLine the BrokenLine struct to remove the line from
   * @param id the id of the line to remove
   * @param toTime current week number
   * @return bias - the current y value of the removed line
   * @return slope - the current absolute slope value of the removed line
   * @return cliff - the remaining cliff period of the removed line
   */
  function _remove(
    BrokenLine storage brokenLine,
    uint256 id,
    uint32 toTime
  ) internal returns (uint96 bias, uint96 slope, uint32 cliff) {
    Line memory line = brokenLine.initiatedLines[id];
    require(line.bias != 0, "Removing Line, which not exists");

    update(brokenLine, toTime);
    //check time Line is over
    bias = line.bias;
    slope = line.slope;
    cliff = 0;
    //for information: bias / (slope) - this`s period while slope works
    uint32 finishTime = line.start + (uint32(bias / (slope))) + (line.cliff);
    if (toTime > finishTime) {
      bias = 0;
      slope = 0;
      return (bias, slope, cliff);
    }
    uint32 finishTimeMinusOne = finishTime - 1;
    uint32 toTimeMinusOne = toTime - 1;
    int96 mod = safeInt(bias % slope);
    uint32 cliffEnd = line.start + (line.cliff) - 1;
    if (toTime <= cliffEnd) {
      //cliff works
      cliff = cliffEnd - (toTime) + 1;
      //in cliff finish time compensate change slope by oldLine.slope
      brokenLine.slopeChanges.subFromItem(cliffEnd, safeInt(slope));
      //in new Line finish point use oldLine.slope
      brokenLine.slopeChanges.addToItem(finishTimeMinusOne, safeInt(slope) - (mod));
    } else if (toTime <= finishTimeMinusOne) {
      //slope works
      //now compensate change slope by oldLine.slope
      brokenLine.initial.slope = brokenLine.initial.slope - (slope);
      //in new Line finish point use oldLine.slope
      brokenLine.slopeChanges.addToItem(finishTimeMinusOne, safeInt(slope) - (mod));
      bias = (uint96(finishTime - (toTime)) * slope) + (uint96(mod));
      //save slope for history
      brokenLine.slopeChanges.subFromItem(toTimeMinusOne, safeInt(slope));
    } else {
      //tail works
      //now compensate change slope by tail
      brokenLine.initial.slope = brokenLine.initial.slope - (uint96(mod));
      bias = uint96(mod);
      slope = bias;
      //save slope for history
      brokenLine.slopeChanges.subFromItem(toTimeMinusOne, safeInt(slope));
    }
    brokenLine.slopeChanges.addToItem(finishTime, mod);
    brokenLine.initial.bias = brokenLine.initial.bias - (bias);
    brokenLine.initiatedLines[id].bias = 0;
  }

  /**
   * @notice Removes a line from a given BrokenLine struct and saves a snapshot
   * @param brokenLine the BrokenLine struct to remove the line from
   * @param id the id of the line to remove
   * @param toTime current week number
   * @param blockNumber the block number when the line is removed
   * @return bias - the current y value of the removed line
   * @return slope - the current absolute slope value of the removed line
   * @return cliff - the remaining cliff period of the removed line
   */
  function remove(
    BrokenLine storage brokenLine,
    uint256 id,
    uint32 toTime,
    uint32 blockNumber
  ) internal returns (uint96 bias, uint96 slope, uint32 cliff) {
    (bias, slope, cliff) = _remove(brokenLine, id, toTime);
    saveSnapshot(brokenLine, toTime, blockNumber);
  }

  /**
   * @notice Updates a given BrokenLine until a given week number
   * @param brokenLine the BrokenLine struct to update
   * @param toTime the week number to update the BrokenLine to
   */
  function update(BrokenLine storage brokenLine, uint32 toTime) internal {
    uint32 time = brokenLine.initial.start;
    if (time == toTime) {
      return;
    }
    uint96 slope = brokenLine.initial.slope;
    uint96 bias = brokenLine.initial.bias;
    if (bias != 0) {
      require(toTime > time, "can't update BrokenLine for past time");
      while (time < toTime) {
        bias = bias - (slope);

        int96 newSlope = safeInt(slope) + (brokenLine.slopeChanges[time]);
        require(newSlope >= 0, "slope < 0, something wrong with slope");
        slope = uint96(newSlope);

        time = time + 1;
      }
    }
    brokenLine.initial.start = toTime;
    brokenLine.initial.bias = bias;
    brokenLine.initial.slope = slope;
  }

  /**
   * @notice Returns y value of a given BrokenLine at a given week and block number
   * @param brokenLine the BrokenLine struct to get the value from
   * @param toTime the week number to get the value at
   * @param toBlock the block number to get the value at
   * @return the y value of the BrokenLine at the given week and block number
   */
  function actualValue(BrokenLine storage brokenLine, uint32 toTime, uint32 toBlock) internal view returns (uint96) {
    uint32 fromTime = brokenLine.initial.start;
    if (fromTime == toTime) {
      if (brokenLine.history[brokenLine.history.length - 1].blockNumber < toBlock) {
        return (brokenLine.initial.bias);
      } else {
        return actualValueBack(brokenLine, toTime, toBlock);
      }
    }
    if (toTime > fromTime) {
      return actualValueForward(brokenLine, fromTime, toTime, brokenLine.initial.bias, brokenLine.initial.slope);
    }
    return actualValueBack(brokenLine, toTime, toBlock);
  }

  /**
   * @notice Calculates and returns the y value of a given BrokenLine for given a week
   * @dev calculates the y value at toTime by applying the slope to the bias starting at the fromTime.
   * Per week the slope is updated by the slopeChanges map until the toTime is reached.
   * @param brokenLine the BrokenLine struct to calculate the value for
   * @param fromTime the week number to start the calculation from
   * @param toTime the week number to calculate the value at
   * @param bias the y value at the fromTime
   * @param slope the absolute slope value at the fromTime
   * @return the y value of the BrokenLine at the given toTime week
   */
  function actualValueForward(
    BrokenLine storage brokenLine,
    uint32 fromTime,
    uint32 toTime,
    uint96 bias,
    uint96 slope
  ) internal view returns (uint96) {
    if ((bias == 0)) {
      return (bias);
    }
    uint32 time = fromTime;

    while (time < toTime) {
      bias = bias - (slope);

      int96 newSlope = safeInt(slope) + (brokenLine.slopeChanges[time]);
      require(newSlope >= 0, "slope < 0, something wrong with slope");
      slope = uint96(newSlope);

      time = time + 1;
    }
    return bias;
  }

  /**
   * @notice Finds the closest snapshot to a given block number and calculates the y value based on that.
   * @param brokenLine the BrokenLine struct to calculate the value for
   * @param toTime the week number to get the y value at
   * @param toBlock the block number to get the y value at
   * @return the y value of the BrokenLine at the given toTime week
   */
  function actualValueBack(
    BrokenLine storage brokenLine,
    uint32 toTime,
    uint32 toBlock
  ) internal view returns (uint96) {
    (uint96 bias, uint96 slope, uint32 fromTime) = binarySearch(brokenLine.history, toBlock);
    return actualValueForward(brokenLine, fromTime, toTime, bias, slope);
  }

  /**
   * @notice Safely casts a uint96 to an int96
   * @param value the uint96 to cast
   * @return result the int96 represesntation of the given uint96 value
   */
  function safeInt(uint96 value) internal pure returns (int96 result) {
    require(value < 2 ** 95, "int cast error");
    result = int96(value);
  }

  /**
   * @notice Saves a point in the history of a given BrokenLine
   * @param brokenLine The BrokenLine that will be snapshotted
   * @param epoch The week number of the snapshot
   * @param blockNumber The block number of the snapshot
   */
  function saveSnapshot(BrokenLine storage brokenLine, uint32 epoch, uint32 blockNumber) internal {
    brokenLine.history.push(
      Point({ blockNumber: blockNumber, bias: brokenLine.initial.bias, slope: brokenLine.initial.slope, epoch: epoch })
    );
  }

  /**
   * @notice Search for a block number in Points array
   * @dev Performs a binary search on an arrary of Points to find the
   *      element with the closest block number less than or equal to toBlock
   * @param history The array of snapshots of Points
   * @param toBlock The block number to search for
   * @return bias The y value of the point
   * @return slope The slope of the point
   * @return epoch The week number of the point
   */
  function binarySearch(Point[] storage history, uint32 toBlock) internal view returns (uint96, uint96, uint32) {
    uint256 len = history.length;
    if (len == 0 || history[0].blockNumber > toBlock) {
      return (0, 0, 0);
    }
    uint256 min = 0;
    uint256 max = len - 1;

    for (uint256 i = 0; i < 128; i++) {
      if (min >= max) {
        break;
      }
      uint256 mid = (min + max + 1) / 2;
      if (history[mid].blockNumber <= toBlock) {
        min = mid;
      } else {
        max = mid - 1;
      }
    }
    return (history[min].bias, history[min].slope, history[min].epoch);
  }
}
