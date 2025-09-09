// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

import { console } from "forge-std/console.sol";
import { StdCheats } from "forge-std/StdCheats.sol";
import { ExchangeForkTest } from "../ExchangeForkTest.sol";
import { Vm } from "forge-std/Vm.sol";
import { VM_ADDRESS } from "mento-std/Constants.sol";
import { ExchangeForkTest } from "../ExchangeForkTest.sol";

import { IERC20 } from "contracts/interfaces/IERC20.sol";
import { FixidityLib } from "celo/contracts/common/FixidityLib.sol";
import { ITradingLimits } from "contracts/interfaces/ITradingLimits.sol";

import { TokenHelpers } from "../helpers/TokenHelpers.sol";
import { OracleHelpers } from "../helpers/OracleHelpers.sol";
import { SwapHelpers } from "../helpers/SwapHelpers.sol";
import { TradingLimitHelpers } from "../helpers/TradingLimitHelpers.sol";
import { LogHelpers } from "../helpers/LogHelpers.sol";

contract SwapActions is StdCheats {
  Vm private vm = Vm(VM_ADDRESS);
  ExchangeForkTest private ctx = ExchangeForkTest(address(this));

  using FixidityLib for FixidityLib.Fraction;
  using OracleHelpers for *;
  using SwapHelpers for *;
  using TokenHelpers for *;
  using TradingLimitHelpers for *;
  using LogHelpers for *;

  uint8 private constant L0 = 1; // 0b001 Limit0
  uint8 private constant L1 = 2; // 0b010 Limit1
  uint8 private constant LG = 4; // 0b100 LimitGlobal

  function swapIn(address from, address to, uint256 sellAmount) public returns (uint256 amountOut) {
    ctx.mint(from, ctx.trader(), sellAmount, true);
    vm.startPrank(ctx.trader());
    IERC20(from).approve(address(ctx.broker()), sellAmount);
    vm.stopPrank();

    ctx.addReportsIfNeeded();
    uint256 minAmountOut = ctx.broker().getAmountOut(
      address(ctx.exchangeProvider()),
      ctx.exchangeId(),
      from,
      to,
      sellAmount
    );

    amountOut = brokerSwapIn(from, to, sellAmount, minAmountOut);
  }

  function brokerSwapIn(
    address from,
    address to,
    uint256 sellAmount,
    uint256 minAmountOut
  ) public returns (uint256 amountOut) {
    console.log(
      unicode"🤝 swapIn(%s, amountIn: %d, minAmountOut: %d)",
      string(abi.encodePacked(from.symbol(), "->", to.symbol())),
      sellAmount.toUnits(from),
      minAmountOut.toUnits(to)
    );
    vm.startPrank(ctx.trader());
    amountOut = ctx.broker().swapIn(ctx.exchangeProviderAddr(), ctx.exchangeId(), from, to, sellAmount, minAmountOut);
    vm.stopPrank();
  }

  function swapOut(address from, address to, uint256 buyAmount) public returns (uint256) {
    ctx.addReportsIfNeeded();
    uint256 maxAmountIn = ctx.getAmountIn(from, to, buyAmount);

    ctx.mint(from, ctx.trader(), maxAmountIn, true);
    vm.startPrank(ctx.trader());
    IERC20(from).approve(address(ctx.broker()), maxAmountIn);
    return brokerSwapOut(from, to, buyAmount, maxAmountIn);
  }

  function brokerSwapOut(
    address from,
    address to,
    uint256 buyAmount,
    uint256 maxAmountIn
  ) public returns (uint256 amountIn) {
    console.log(
      string(
        abi.encodePacked(unicode"🤝 swapOut(", from.symbol(), "->", to.symbol(), ",amountOut: %d, maxAmountIn: %d)")
      ),
      buyAmount.toUnits(to),
      maxAmountIn.toUnits(from)
    );
    vm.startPrank(ctx.trader());
    amountIn = ctx.broker().swapOut(ctx.exchangeProviderAddr(), ctx.exchangeId(), from, to, buyAmount, maxAmountIn);
    vm.stopPrank();
  }

  function swapUntilL0_onInflow(address from, address to) internal {
    /*
     * L0[from] -> to
     * This function will do valid swaps until just before L0 is hit
     * during inflow on `from`, therfore we check the positive end
     * of the limit because `from` flows into the reserve.
     */

    ITradingLimits.Config memory limitConfig = ctx.tradingLimitsConfig(from);
    console.log(unicode"🏷️ [%d] Swap until L0=%d on inflow", block.timestamp, uint256(int256(limitConfig.limit0)));
    uint256 maxPossible;
    uint256 maxPossibleUntilLimit;
    do {
      ctx.logLimits(from);
      int48 maxPossibleUntilLimitUnits = ctx.maxInflow(from);
      console.log("\tmaxPossibleUntilLimitUnits: %d", maxPossibleUntilLimitUnits);
      require(maxPossibleUntilLimitUnits >= 0, "max possible trade amount is negative");
      maxPossibleUntilLimit = uint256(int256(maxPossibleUntilLimitUnits)).toSubunits(from);
      console.log("\tmaxPossibleUntilLimit: %d", maxPossibleUntilLimit);
      maxPossible = ctx.maxSwapIn(maxPossibleUntilLimit, from, to);
      console.log("\tmaxPossible: %d", maxPossible);

      if (maxPossible > 0) {
        ctx.swapIn(from, to, maxPossible);
      }
    } while (maxPossible > 0 && maxPossibleUntilLimit > maxPossible);
    ctx.logLimits(from);
  }

  function swapUntilL1_onInflow(address from, address to) internal {
    /*
     * L1[from] -> to
     * This function will do valid swaps until just before L1 is hit
     * during inflow on `from`, therfore we check the positive end
     * of the limit because `from` flows into the reserve.
     */
    ITradingLimits.Config memory limitConfig = ctx.tradingLimitsConfig(from);
    ITradingLimits.State memory limitState = ctx.refreshedTradingLimitsState(from);
    console.log(unicode"🏷️ [%d] Swap until L1=%d on inflow", block.timestamp, uint256(int256(limitConfig.limit1)));
    int48 maxPerSwap = limitConfig.limit0;
    while (limitState.netflow1 + maxPerSwap <= limitConfig.limit1) {
      skip(limitConfig.timestep0 + 1);
      ctx.ensureRateActive(); // needed because otherwise constantSum might revert if the median is stale due to the skip

      swapUntilL0_onInflow(from, to);
      limitConfig = ctx.tradingLimitsConfig(from);
      limitState = ctx.refreshedTradingLimitsState(from);
      if (limitState.netflowGlobal == limitConfig.limitGlobal) {
        console.log(unicode"🚨 LG reached during L1 inflow");
        break;
      }
    }
    skip(limitConfig.timestep0 + 1);
    ctx.ensureRateActive();
  }

  function swapUntilLG_onInflow(address from, address to) internal {
    /*
     * L1[from] -> to
     * This function will do valid swaps until just before LG is hit
     * during inflow on `from`, therfore we check the positive end
     * of the limit because `from` flows into the reserve.
     */
    ITradingLimits.Config memory limitConfig = ctx.tradingLimitsConfig(from);
    ITradingLimits.State memory limitState = ctx.refreshedTradingLimitsState(from);
    console.log(unicode"🏷️ [%d] Swap until LG=%d on inflow", block.timestamp, uint256(int256(limitConfig.limitGlobal)));

    if (limitConfig.isLimitEnabled(L1)) {
      int48 maxPerSwap = limitConfig.limit0;
      uint256 it;
      while (limitState.netflowGlobal + maxPerSwap <= limitConfig.limitGlobal) {
        skip(limitConfig.timestep1 + 1);
        swapUntilL1_onInflow(from, to);
        limitConfig = ctx.tradingLimitsConfig(from);
        limitState = ctx.tradingLimitsState(from);
        it++;
        require(it < 50, "infinite loop");
      }
      skip(limitConfig.timestep1 + 1);
    } else if (limitConfig.isLimitEnabled(L0)) {
      int48 maxPerSwap = limitConfig.limit0;
      uint256 it;
      while (limitState.netflowGlobal + maxPerSwap <= limitConfig.limitGlobal) {
        skip(limitConfig.timestep0 + 1);
        swapUntilL0_onInflow(from, to);
        limitConfig = ctx.tradingLimitsConfig(from);
        limitState = ctx.tradingLimitsState(from);
        it++;
        require(it < 50, "infinite loop");
      }
      skip(limitConfig.timestep0 + 1);
    }
  }

  function swapUntilL0_onOutflow(address from, address to) public {
    /*
     * from -> L0[to]
     * This function will do valid swaps until just before L0 is hit
     * during outflow on `to`, therfore we check the negative end
     * of the limit because `to` flows out of the reserve.
     */

    ITradingLimits.Config memory limitConfig = ctx.tradingLimitsConfig(to);
    console.log(unicode"🏷️ [%d] Swap until L0=%d on outflow", block.timestamp, uint256(int256(limitConfig.limit0)));
    uint256 maxPossible;
    uint256 maxPossibleUntilLimit;
    do {
      ctx.logLimits(to);
      int48 maxPossibleUntilLimitUnits = ctx.maxOutflow(to);
      console.log("\tmaxPossibleUnits: %d", maxPossibleUntilLimitUnits);
      require(maxPossibleUntilLimitUnits >= 0, "max possible trade amount is negative");
      maxPossibleUntilLimit = uint256(maxPossibleUntilLimitUnits.toSubunits(to));
      console.log("\tmaxPossibleUnits: %d", maxPossibleUntilLimit);
      maxPossible = ctx.maxSwapOut(maxPossibleUntilLimit, to);
      console.log("\tmaxPossibleActual: %d", maxPossible);

      if (maxPossible > 0) {
        ctx.swapOut(from, to, maxPossible);
      }
    } while (maxPossible > 0 && maxPossibleUntilLimit > maxPossible);
    ctx.logLimits(to);
  }

  function swapUntilL1_onOutflow(address from, address to) public {
    /*
     * from -> L1[to]
     * This function will do valid swaps until just before L1 is hit
     * during outflow on `to`, therfore we check the negative end
     * of the limit because `to` flows out of the reserve.
     */
    ITradingLimits.Config memory limitConfig = ctx.tradingLimitsConfig(to);
    ITradingLimits.State memory limitState = ctx.refreshedTradingLimitsState(to);

    console.log(unicode"🏷️ [%d] Swap until L1=%d on outflow", block.timestamp, uint48(limitConfig.limit1));
    int48 maxPerSwap = limitConfig.limit0;
    uint256 it;
    while (limitState.netflow1 - maxPerSwap >= -1 * limitConfig.limit1) {
      skip(limitConfig.timestep0 + 1);
      // Check that there's still outflow to trade as sometimes we hit LG while
      // still having a bit of L1 left, which causes an infinite loop.
      if (ctx.maxOutflow(to) == 0) {
        break;
      }
      swapUntilL0_onOutflow(from, to);
      limitConfig = ctx.tradingLimitsConfig(to);
      limitState = ctx.tradingLimitsState(to);
      it++;
      require(it < 10, "infinite loop");
    }
    skip(limitConfig.timestep0 + 1);
  }

  function swapUntilLG_onOutflow(address from, address to) public {
    /*
     * from -> LG[to]
     * This function will do valid swaps until just before LG is hit
     * during outflow on `to`, therefore we check the negative end
     * of the limit because `to` flows out of the reserve.
     */
    ITradingLimits.Config memory limitConfig = ctx.tradingLimitsConfig(to);
    ITradingLimits.State memory limitState = ctx.refreshedTradingLimitsState(to);
    console.log(unicode"🏷️ [%d] Swap until LG=%d on outflow", block.timestamp, uint48(limitConfig.limitGlobal));

    if (limitConfig.isLimitEnabled(L1)) {
      int48 maxPerSwap = limitConfig.limit0;
      while (limitState.netflowGlobal - maxPerSwap >= -1 * limitConfig.limitGlobal) {
        skip(limitConfig.timestep1 + 1);
        swapUntilL1_onOutflow(from, to);
        limitConfig = ctx.tradingLimitsConfig(to);
        // Trigger an update to reset netflows
        limitState = ctx.tradingLimitsState(to);
      }
      skip(limitConfig.timestep1 + 1);
    } else if (limitConfig.isLimitEnabled(L0)) {
      int48 maxPerSwap = limitConfig.limit0;
      while (limitState.netflowGlobal - maxPerSwap >= -1 * limitConfig.limitGlobal) {
        skip(limitConfig.timestep0 + 1);
        swapUntilL0_onOutflow(from, to);
        limitConfig = ctx.tradingLimitsConfig(to);
        // Trigger an update to reset netflows
        limitState = ctx.tradingLimitsState(to);
      }
      skip(limitConfig.timestep0 + 1);
    }
  }
}
