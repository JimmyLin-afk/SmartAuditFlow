// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

import { console } from "forge-std/console.sol";
import { ExchangeForkTest } from "../ExchangeForkTest.sol";
import { SwapHelpers } from "./SwapHelpers.sol";
import { TokenHelpers } from "./TokenHelpers.sol";
import { OracleHelpers } from "./OracleHelpers.sol";
import { TradingLimitHelpers } from "./TradingLimitHelpers.sol";

import { IBiPoolManager } from "contracts/interfaces/IBiPoolManager.sol";
import { ITradingLimits } from "contracts/interfaces/ITradingLimits.sol";
import { L0, L1, LG } from "./misc.sol";

library LogHelpers {
  using SwapHelpers for *;
  using OracleHelpers for *;
  using TokenHelpers for *;
  using TradingLimitHelpers for *;

  function logHeader(ExchangeForkTest ctx) internal view {
    console.log("========================================");
    console.log(unicode"🔦 Testing pair:", ctx.ticker());
    console.log("========================================");
  }

  function logPool(ExchangeForkTest ctx) internal view {
    IBiPoolManager.PoolExchange memory exchange = ctx.getPool();

    (bool timePassed, bool enoughReports, bool medianReportRecent, bool isReportExpired, ) = ctx.shouldUpdateBuckets();
    console.log(unicode"🎱 Pool: %s", ctx.ticker());
    console.log(
      "\t timePassed: %s | enoughReports: %s",
      timePassed ? "true" : "false",
      enoughReports ? "true" : "false"
    );
    console.log(
      "\t medianReportRecent: %s | !isReportExpired: %s",
      medianReportRecent ? "true" : "false",
      !isReportExpired ? "true" : "false"
    );
    console.log(
      "\t exchange.bucket0: %d | exchange.bucket1: %d",
      exchange.bucket0.toUnits(ctx.asset(0)),
      exchange.bucket1.toUnits(ctx.asset(1))
    );
    console.log("\t exchange.lastBucketUpdate: %d", exchange.lastBucketUpdate);
  }

  function logLimits(ExchangeForkTest ctx, address target) internal view {
    ITradingLimits.State memory state = ctx.refreshedTradingLimitsState(target);
    ITradingLimits.Config memory config = ctx.tradingLimitsConfig(target);
    console.log("TradingLimits[%s]:", target.symbol());
    if (config.flags & L0 > 0) {
      console.log(
        "\tL0: %s%d/%d",
        state.netflow0 < 0 ? "-" : "",
        uint256(int256(state.netflow0 < 0 ? state.netflow0 * -1 : state.netflow0)),
        uint256(int256(config.limit0))
      );
    }
    if (config.flags & L1 > 0) {
      console.log(
        "\tL1: %s%d/%d",
        state.netflow1 < 0 ? "-" : "",
        uint256(int256(state.netflow1 < 0 ? state.netflow1 * -1 : state.netflow1)),
        uint256(int256(config.limit1))
      );
    }
    if (config.flags & LG > 0) {
      console.log(
        "\tLG: %s%d/%d",
        state.netflowGlobal < 0 ? "-" : "",
        uint256(int256(state.netflowGlobal < 0 ? state.netflowGlobal * -1 : state.netflowGlobal)),
        uint256(int256(config.limitGlobal))
      );
    }
  }
}
