// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;
pragma experimental ABIEncoderV2;

import { ITradingLimits } from "contracts/interfaces/ITradingLimits.sol";

/**
 * @title TradingLimits
 * @author Mento Team
 * @notice This library provides data structs and utility functions for
 * defining and verifying trading limits on the netflow of an asset.
 * There are three limits that can be enabled:
 * - L0: A timewindow based limit, verifies that:
 *       -1 * limit0 <= netflow0 <= limit0,
 *       for a netflow0 that resets every timespan0 seconds.
 * - L1: A timewindow based limit, verifies that:
 *       -1 * limit1 <= netflow1 <= limit1,
 *       for a netflow1 that resets every timespan1 second.
 * - LG: A global (or lifetime) limit that ensures that:
 *       -1 * limitGlobal <= netflowGlobal <= limitGlobal,
 *       for a netflowGlobal that doesn't reset until the
 *       limit is disabled.
 * @dev All contained functions are pure or view and marked internal to
 * be inlined on consuming contracts at compile time for gas efficiency.
 * Both State and Config structs are designed to be packed in one
 * storage slot each.
 * In order to pack both the state and config into one slot each,
 * some assumptions are made:
 * 1. limit{0,1,Global} and netflow{0,1,Global} are recorded with
 *    ZERO decimals precision to fit in an int48.
 *    Any subunit delta in netflow will be rounded up to one unit.
 * 2. netflow{0,1,Global} have to fit in int48, thus have to fit in the range:
 *    -140_737_488_355_328 to 140_737_488_355_328, which can cover most
 *    tokens of interest, but will break down for tokens which trade
 *    in large unit values.
 * 3. timespan{0,1} and lastUpdated{0,1} have to fit in int32 therefore
 *    the timestamps will overflow sometime in the year 2102.
 *
 * The library ensures that netflow0 and netflow1 are reset during
 * the update phase, but does not control how the full State gets
 * updated if the Config changes, this is left to the library consumer.
 */
library TradingLimits {
  uint8 private constant L0 = 1; // 0b001 Limit0
  uint8 private constant L1 = 2; // 0b010 Limit1
  uint8 private constant LG = 4; // 0b100 LimitGlobal
  int48 private constant MAX_INT48 = type(int48).max;

  /**
   * @notice Validate a trading limit configuration.
   * @dev Reverts if the configuration is malformed.
   * @param self the Config struct to check.
   */
  function validate(ITradingLimits.Config memory self) internal pure {
    require(self.flags & L1 == 0 || self.flags & L0 != 0, "L1 without L0 not allowed");
    require(self.flags & L0 == 0 || self.timestep0 > 0, "timestep0 can't be zero if active");
    require(self.flags & L1 == 0 || self.timestep1 > 0, "timestep1 can't be zero if active");
    require(self.flags & L0 == 0 || self.limit0 > 0, "limit0 can't be zero if active");
    require(self.flags & L1 == 0 || self.limit1 > 0, "limit1 can't be zero if active");
    require(self.flags & LG == 0 || self.limitGlobal > 0, "limitGlobal can't be zero if active");
    require(self.flags & (L0 | L1) != 3 || self.limit0 < self.limit1, "limit1 must be greater than limit0");
    require(self.flags & (L1 | LG) != 6 || self.limit1 < self.limitGlobal, "limitGlobal must be greater than limit1");
    require(self.flags & (L0 | LG) != 5 || self.limit0 < self.limitGlobal, "limitGlobal must be greater than limit0");
  }

  /**
   * @notice Verify a trading limit State with a provided Config.
   * @dev Reverts if the limits are exceeded.
   * @param self the trading limit State to check.
   * @param config the trading limit Config to check against.
   */
  function verify(ITradingLimits.State memory self, ITradingLimits.Config memory config) internal pure {
    if ((config.flags & L0) > 0 && (-1 * config.limit0 > self.netflow0 || self.netflow0 > config.limit0)) {
      revert("L0 Exceeded");
    }
    if ((config.flags & L1) > 0 && (-1 * config.limit1 > self.netflow1 || self.netflow1 > config.limit1)) {
      revert("L1 Exceeded");
    }
    if (
      (config.flags & LG) > 0 &&
      (-1 * config.limitGlobal > self.netflowGlobal || self.netflowGlobal > config.limitGlobal)
    ) {
      revert("LG Exceeded");
    }
  }

  /**
   * @notice Reset an existing state with a new config.
   * It keps netflows of enabled limits and resets when disabled.
   * It resets all timestamp checkpoints to reset time-window limits
   * on next swap.
   * @param self the trading limit state to reset.
   * @param config the updated config to reset against.
   * @return the reset state.
   */
  function reset(
    ITradingLimits.State memory self,
    ITradingLimits.Config memory config
  ) internal pure returns (ITradingLimits.State memory) {
    // Ensure the next swap will reset the trading limits windows.
    self.lastUpdated0 = 0;
    self.lastUpdated1 = 0;
    if (config.flags & L0 == 0) {
      self.netflow0 = 0;
    }
    if (config.flags & L1 == 0) {
      self.netflow1 = 0;
    }
    if (config.flags & LG == 0) {
      self.netflowGlobal = 0;
    }
    return self;
  }

  /**
   * @notice  Updates a trading limit State in the context of a Config with the deltaFlow provided.
   * @dev Reverts if the values provided cause overflows.
   * @param self the trading limit State to update.
   * @param config the trading limit Config for the provided State.
   * @param _deltaFlow the delta flow to add to the netflow.
   * @param decimals the number of decimals the _deltaFlow is denominated in.
   * @return State the updated state.
   */
  function update(
    ITradingLimits.State memory self,
    ITradingLimits.Config memory config,
    int256 _deltaFlow,
    uint8 decimals
  ) internal view returns (ITradingLimits.State memory) {
    int256 _deltaFlowUnits = _deltaFlow / int256((10 ** uint256(decimals)));
    require(_deltaFlowUnits <= MAX_INT48, "dFlow too large");

    int48 deltaFlowUnits = int48(_deltaFlowUnits);
    if (deltaFlowUnits == 0) {
      deltaFlowUnits = _deltaFlow > 0 ? int48(1) : int48(-1);
    }

    if (config.flags & L0 > 0) {
      if (block.timestamp > self.lastUpdated0 + config.timestep0) {
        self.netflow0 = 0;
        self.lastUpdated0 = uint32(block.timestamp);
      }
      self.netflow0 = safeINT48Add(self.netflow0, deltaFlowUnits);

      if (config.flags & L1 > 0) {
        if (block.timestamp > self.lastUpdated1 + config.timestep1) {
          self.netflow1 = 0;
          self.lastUpdated1 = uint32(block.timestamp);
        }
        self.netflow1 = safeINT48Add(self.netflow1, deltaFlowUnits);
      }
    }
    if (config.flags & LG > 0) {
      self.netflowGlobal = safeINT48Add(self.netflowGlobal, deltaFlowUnits);
    }

    return self;
  }

  /**
   * @notice Safe add two int48s.
   * @dev Reverts if addition causes over/underflow.
   * @param a number to add.
   * @param b number to add.
   * @return int48 result of addition.
   */
  function safeINT48Add(int48 a, int48 b) internal pure returns (int48) {
    int256 c = int256(a) + int256(b);
    require(c >= -1 * MAX_INT48 && c <= MAX_INT48, "int48 addition overflow");
    return int48(c);
  }
}
