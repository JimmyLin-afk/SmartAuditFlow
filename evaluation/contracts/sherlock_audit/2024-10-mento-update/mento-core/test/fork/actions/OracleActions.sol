// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

import { console } from "forge-std/console.sol";
import { StdCheats } from "forge-std/StdCheats.sol";
import { Vm } from "forge-std/Vm.sol";
import { VM_ADDRESS } from "mento-std/Constants.sol";

import { ExchangeForkTest } from "../ExchangeForkTest.sol";
import { TokenHelpers } from "../helpers/TokenHelpers.sol";
import { OracleHelpers } from "../helpers/OracleHelpers.sol";
import { SwapHelpers } from "../helpers/SwapHelpers.sol";

import { IBiPoolManager } from "contracts/interfaces/IBiPoolManager.sol";
import { IBreakerBox } from "contracts/interfaces/IBreakerBox.sol";
import { IValueDeltaBreaker } from "contracts/interfaces/IValueDeltaBreaker.sol";

contract OracleActions is StdCheats {
  using OracleHelpers for *;
  using SwapHelpers for *;
  using TokenHelpers for *;

  Vm private vm = Vm(VM_ADDRESS);
  ExchangeForkTest private ctx = ExchangeForkTest(address(this));

  function updateOracleMedianRate(uint256 newMedian) public {
    updateOracleMedianRate(ctx.rateFeedId(), newMedian);
  }

  function updateOracleMedianRate(address rateFeedId, uint256 newMedian) public {
    address[] memory oracles = ctx.sortedOracles().getOracles(rateFeedId);
    require(oracles.length > 0, "No oracles for rateFeedId");
    console.log(unicode"🔮 Updating oracles to new median: ", newMedian);
    for (uint256 i = 0; i < oracles.length; i++) {
      skip(5);
      address oracle = oracles[i];
      address lesserKey;
      address greaterKey;
      (address[] memory keys, uint256[] memory values, ) = ctx.sortedOracles().getRates(rateFeedId);
      for (uint256 j = 0; j < keys.length; j++) {
        if (keys[j] == oracle) continue;
        if (values[j] < newMedian) lesserKey = keys[j];
        if (values[j] >= newMedian) greaterKey = keys[j];
      }

      vm.startPrank(oracle);
      ctx.sortedOracles().report(rateFeedId, newMedian, lesserKey, greaterKey);
      vm.stopPrank();
    }
  }

  function addReportsIfNeeded() public {
    IBiPoolManager.PoolExchange memory pool = ctx.exchangeProvider().getPoolExchange(ctx.exchangeId());
    (bool timePassed, bool enoughReports, bool medianReportRecent, bool isReportExpired, ) = ctx.shouldUpdateBuckets();
    if (timePassed && (!medianReportRecent || isReportExpired || !enoughReports)) {
      (uint256 newMedian, ) = ctx.sortedOracles().medianRate(pool.config.referenceRateFeedID);
      (timePassed, enoughReports, medianReportRecent, isReportExpired, ) = ctx.shouldUpdateBuckets();
      updateOracleMedianRate((newMedian * 1_000_001) / 1_000_000);
      return;
    }
  }

  function ensureRateActive() public returns (uint256 newMedian) {
    return ensureRateActive(ctx.rateFeedId());
  }

  function ensureRateActive(address rateFeedId) public returns (uint256 newMedian) {
    // Always do a small update in order to make sure the breakers are warm.
    (uint256 currentRate, ) = ctx.sortedOracles().medianRate(rateFeedId);
    newMedian = currentRate + (currentRate / 100_000_000); // a small increase
    updateOracleMedianRate(rateFeedId, newMedian);
    uint8 tradingMode = ctx.breakerBox().getRateFeedTradingMode(rateFeedId);
    uint256 attempts = 0;
    while (tradingMode != 0 && attempts < 10) {
      console.log("attempt #%d", attempts);
      attempts++;
      // while the breaker is active, we wait for the cooldown and try to update the median
      console.log(block.timestamp, "Waiting for cooldown to pass");
      console.log("RateFeedID:", rateFeedId);
      address[] memory _breakers = ctx.breakerBox().getBreakers();
      uint256 cooldown = 0;
      uint256 breakerIndex;
      for (uint256 i = 0; i < _breakers.length; i++) {
        if (ctx.breakerBox().isBreakerEnabled(_breakers[i], rateFeedId)) {
          IBreakerBox.BreakerStatus memory status = ctx.breakerBox().rateFeedBreakerStatus(rateFeedId, _breakers[i]);
          if (status.tradingMode != 0) {
            breakerIndex = i;
            cooldown = IValueDeltaBreaker(_breakers[i]).getCooldown(rateFeedId);
            break;
          }
        }
      }
      skip(cooldown);
      newMedian = ctx.newMedianToResetBreaker(rateFeedId, breakerIndex);
      ctx.updateOracleMedianRate(rateFeedId, newMedian);
      if (cooldown == 0) {
        console.log("Manual recovery required for breaker %s", _breakers[breakerIndex]);
        vm.startPrank(ctx.breakerBox().owner());
        ctx.breakerBox().setRateFeedTradingMode(rateFeedId, 0);
        vm.stopPrank();
      }
      tradingMode = ctx.breakerBox().getRateFeedTradingMode(rateFeedId);
    }
  }
}
