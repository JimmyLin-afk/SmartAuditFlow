// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

import { console } from "forge-std/console.sol";
import { StdAssertions } from "forge-std/StdAssertions.sol";
import { Vm } from "forge-std/Vm.sol";
import { VM_ADDRESS } from "mento-std/Constants.sol";
import { FixidityLib } from "celo/contracts/common/FixidityLib.sol";

import { IMedianDeltaBreaker } from "contracts/interfaces/IMedianDeltaBreaker.sol";
import { IValueDeltaBreaker } from "contracts/interfaces/IValueDeltaBreaker.sol";

import { ExchangeForkTest } from "../ExchangeForkTest.sol";
import { Actions } from "../actions/all.sol";
import { TokenHelpers } from "../helpers/TokenHelpers.sol";
import { OracleHelpers } from "../helpers/OracleHelpers.sol";
import { SwapHelpers } from "../helpers/SwapHelpers.sol";
import { TradingLimitHelpers } from "../helpers/TradingLimitHelpers.sol";
import { LogHelpers } from "../helpers/LogHelpers.sol";

contract CircuitBreakerAssertions is StdAssertions, Actions {
  using FixidityLib for FixidityLib.Fraction;
  using OracleHelpers for *;
  using SwapHelpers for *;
  using TokenHelpers for *;
  using TradingLimitHelpers for *;
  using LogHelpers for *;

  Vm private vm = Vm(VM_ADDRESS);
  ExchangeForkTest private ctx = ExchangeForkTest(address(this));

  uint256 private constant fixed1 = 1e24;

  function assert_breakerBreaks(address rateFeedId, address breaker, uint256 breakerIndex) public {
    // XXX: There is currently no straightforward way to determine what type of a breaker
    // we are dealing with, so we will use the deployment setup that we currently chose,
    // where the medianDeltaBreaker gets deployed first and the valueDeltaBreaker second.
    bool isMedianDeltaBreaker = breakerIndex == 0;
    bool isValueDeltaBreaker = breakerIndex == 1;
    if (isMedianDeltaBreaker) {
      assert_medianDeltaBreakerBreaks_onIncrease(rateFeedId, breaker);
      assert_medianDeltaBreakerBreaks_onDecrease(rateFeedId, breaker);
    } else if (isValueDeltaBreaker) {
      assert_valueDeltaBreakerBreaks_onIncrease(rateFeedId, breaker);
      assert_valueDeltaBreakerBreaks_onDecrease(rateFeedId, breaker);
    } else {
      revert("Unknown trading mode, can't infer breaker type");
    }
  }

  function assert_medianDeltaBreakerBreaks_onIncrease(address rateFeedId, address _breaker) public {
    uint256 currentMedian = ensureRateActive(rateFeedId); // ensure trading mode is 0

    // trigger breaker by setting new median to ema - (threshold + 0.001% buffer)
    uint256 currentEMA = IMedianDeltaBreaker(_breaker).medianRatesEMA(rateFeedId);
    uint256 rateChangeThreshold = ctx.getBreakerRateChangeThreshold(rateFeedId, _breaker);
    uint256 thresholdBuffer = FixidityLib.newFixedFraction(1, 1000).unwrap(); // small buffer because of rounding errors
    uint256 maxPercent = fixed1 + rateChangeThreshold + thresholdBuffer;
    uint256 newMedian = (currentEMA * maxPercent) / fixed1;

    console.log("Current Median: ", currentMedian);
    console.log("Current EMA: ", currentEMA);
    console.log("New Median: ", newMedian);
    assert_breakerBreaks_withNewMedian(rateFeedId, newMedian, 3);
  }

  function assert_medianDeltaBreakerBreaks_onDecrease(address rateFeedId, address _breaker) public {
    uint256 currentMedian = ensureRateActive(rateFeedId); // ensure trading mode is 0

    // trigger breaker by setting new median to ema + (threshold + 0.001% buffer)
    uint256 currentEMA = IMedianDeltaBreaker(_breaker).medianRatesEMA(rateFeedId);
    uint256 rateChangeThreshold = ctx.getBreakerRateChangeThreshold(rateFeedId, _breaker);
    uint256 thresholdBuffer = FixidityLib.newFixedFraction(1, 1000).unwrap(); // small buffer because of rounding errors
    uint256 maxPercent = fixed1 - (rateChangeThreshold + thresholdBuffer);
    uint256 newMedian = (currentEMA * maxPercent) / fixed1;

    console.log("Current Median: ", currentMedian);
    console.log("Current EMA: ", currentEMA);
    console.log("New Median: ", newMedian);
    assert_breakerBreaks_withNewMedian(rateFeedId, newMedian, 3);
  }

  function assert_valueDeltaBreakerBreaks_onIncrease(address rateFeedId, address _breaker) public {
    uint256 currentMedian = ensureRateActive(rateFeedId); // ensure trading mode is 0

    // trigger breaker by setting new median to reference value + threshold + 1
    uint256 rateChangeThreshold = ctx.getBreakerRateChangeThreshold(rateFeedId, _breaker);
    uint256 referenceValue = ctx.getValueDeltaBreakerReferenceValue(rateFeedId, _breaker);
    uint256 maxPercent = fixed1 + rateChangeThreshold;
    uint256 newMedian = (referenceValue * maxPercent) / fixed1;
    newMedian = newMedian + 1;

    console.log("Current Median: ", currentMedian);
    console.log("Reference Value: ", referenceValue);
    console.log("New Median: ", newMedian);
    assert_breakerBreaks_withNewMedian(rateFeedId, newMedian, 3);
  }

  function assert_valueDeltaBreakerBreaks_onDecrease(address rateFeedId, address _breaker) public {
    uint256 currentMedian = ensureRateActive(rateFeedId); // ensure trading mode is 0

    // trigger breaker by setting new median to reference value - threshold - 1
    uint256 rateChangeThreshold = ctx.getBreakerRateChangeThreshold(rateFeedId, _breaker);
    uint256 referenceValue = ctx.getValueDeltaBreakerReferenceValue(rateFeedId, _breaker);
    uint256 maxPercent = fixed1 - rateChangeThreshold;
    uint256 newMedian = (referenceValue * maxPercent) / fixed1;
    newMedian = newMedian - 1;

    console.log("Current Median: ", currentMedian);
    console.log("Reference Value: ", referenceValue);
    console.log("New Median: ", newMedian);
    assert_breakerBreaks_withNewMedian(rateFeedId, newMedian, 3);
  }

  function assert_breakerRecovers(address rateFeedId, address breaker, uint256 breakerIndex) public {
    // XXX: There is currently no straightforward way to determine what type of a breaker
    // we are dealing with, so we will use the deployment setup that we currently chose,
    // where the medianDeltaBreaker gets deployed first and the valueDeltaBreaker second.
    bool isMedianDeltaBreaker = breakerIndex == 0;
    bool isValueDeltaBreaker = breakerIndex == 1;
    if (isMedianDeltaBreaker) {
      assert_medianDeltaBreakerRecovers(rateFeedId, breaker);
    } else if (isValueDeltaBreaker) {
      assert_valueDeltaBreakerRecovers(rateFeedId, breaker);
    } else {
      revert("Unknown trading mode, can't infer breaker type");
    }
  }

  function assert_medianDeltaBreakerRecovers(address rateFeedId, address _breaker) internal {
    uint256 currentMedian = ensureRateActive(rateFeedId); // ensure trading mode is 0
    IMedianDeltaBreaker breaker = IMedianDeltaBreaker(_breaker);

    // trigger breaker by setting new median to ema + threshold + 0.001%
    uint256 currentEMA = breaker.medianRatesEMA(rateFeedId);
    uint256 rateChangeThreshold = ctx.getBreakerRateChangeThreshold(rateFeedId, _breaker);
    uint256 thresholdBuffer = FixidityLib.newFixedFraction(1, 1000).unwrap();
    uint256 maxPercent = fixed1 + rateChangeThreshold + thresholdBuffer;
    uint256 newMedian = (currentEMA * maxPercent) / fixed1;

    console.log("Current Median: ", currentMedian);
    console.log("New Median: ", newMedian);
    assert_breakerBreaks_withNewMedian(rateFeedId, newMedian, 3);

    // wait for cool down and reset by setting new median to ema
    uint256 cooldown = breaker.getCooldown(rateFeedId);
    if (cooldown == 0) {
      changePrank(ctx.breakerBox().owner());
      ctx.breakerBox().setRateFeedTradingMode(rateFeedId, 0);
    } else {
      skip(cooldown);
      currentEMA = breaker.medianRatesEMA(rateFeedId);
      assert_breakerRecovers_withNewMedian(rateFeedId, currentEMA);
    }
  }

  function assert_valueDeltaBreakerRecovers(address rateFeedId, address _breaker) internal {
    uint256 currentMedian = ensureRateActive(rateFeedId); // ensure trading mode is 0
    IValueDeltaBreaker breaker = IValueDeltaBreaker(_breaker);

    // trigger breaker by setting new median to reference value + threshold + 1
    uint256 rateChangeThreshold = ctx.getBreakerRateChangeThreshold(rateFeedId, _breaker);
    uint256 referenceValue = ctx.getValueDeltaBreakerReferenceValue(rateFeedId, _breaker);
    uint256 maxPercent = fixed1 + rateChangeThreshold;
    uint256 newMedian = (referenceValue * maxPercent) / fixed1;
    newMedian = newMedian + 1;

    console.log("Current Median: ", currentMedian);
    console.log("Reference Value: ", referenceValue);
    console.log("New Median: ", newMedian);
    assert_breakerBreaks_withNewMedian(rateFeedId, newMedian, 3);

    // wait for cool down and reset by setting new median to refernece value
    uint256 cooldown = breaker.getCooldown(rateFeedId);
    if (cooldown == 0) {
      changePrank(ctx.breakerBox().owner());
      ctx.breakerBox().setRateFeedTradingMode(rateFeedId, 0);
    } else {
      skip(cooldown);
      assert_breakerRecovers_withNewMedian(rateFeedId, referenceValue);
    }
  }

  function assert_breakerBreaks_withNewMedian(
    address rateFeedId,
    uint256 newMedian,
    uint256 expectedTradingMode
  ) public {
    uint256 tradingMode = ctx.breakerBox().getRateFeedTradingMode(rateFeedId);
    require(tradingMode == 0, "breaker should be recovered");

    ctx.updateOracleMedianRate(rateFeedId, newMedian);
    tradingMode = ctx.breakerBox().getRateFeedTradingMode(ctx.rateFeedId());
    require(tradingMode == expectedTradingMode, "trading more is different from expected");
  }

  function assert_breakerRecovers_withNewMedian(address rateFeedId, uint256 newMedian) public {
    uint256 tradingMode = ctx.breakerBox().getRateFeedTradingMode(rateFeedId);
    require(tradingMode != 0, "breaker should be triggered");

    ctx.updateOracleMedianRate(rateFeedId, newMedian);
    tradingMode = ctx.breakerBox().getRateFeedTradingMode(rateFeedId);
    require(tradingMode == 0, "breaker should be recovered");
  }
}
