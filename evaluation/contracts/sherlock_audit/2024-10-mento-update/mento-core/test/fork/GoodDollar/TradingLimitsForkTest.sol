// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

// Libraries
import { console } from "forge-std/console.sol";
import { L0, L1, LG } from "../helpers/misc.sol";
import { TokenHelpers } from "../helpers/TokenHelpers.sol";
import { TradingLimitHelpers } from "../helpers/TradingLimitHelpers.sol";

// Interfaces
import { ITradingLimits } from "contracts/interfaces/ITradingLimits.sol";

// Contracts
import { GoodDollarBaseForkTest } from "./GoodDollarBaseForkTest.sol";

contract GoodDollarTradingLimitsForkTest is GoodDollarBaseForkTest {
  using TradingLimitHelpers for *;
  using TokenHelpers for *;

  constructor(uint256 _chainId) GoodDollarBaseForkTest(_chainId) {}

  function setUp() public override {
    super.setUp();
  }

  function test_tradingLimitsAreConfiguredForReserveToken() public view {
    ITradingLimits.Config memory config = getTradingLimitsConfig(address(reserveToken));
    bool reserveAssetLimitConfigured = config.flags > uint8(0);
    require(reserveAssetLimitConfigured, "Limit not configured");
  }

  function test_tradingLimitsAreEnforced_reserveTokenOutflowLimit0() public {
    _swapUntilReserveTokenLimit0_onOutflow();
    _swapGoodDollarForReserveToken(bytes(L0.revertReason()));
  }

  function test_tradingLimitsAreEnforced_reserveTokenOutflowLimit1() public {
    _swapUntilReserveTokenLimit1_onOutflow();
    _swapGoodDollarForReserveToken(bytes(L1.revertReason()));
  }

  function test_tradingLimitsAreEnforced_reserveTokenOutflowLimitGlobal() public {
    _swapUntilReserveTokenGlobalLimit_onOutflow();
    _swapGoodDollarForReserveToken(bytes(LG.revertReason()));
  }

  function test_tradingLimitsAreEnforced_reserveTokenInflowLimit0() public {
    _swapUntilReserveTokenLimit0_onInflow();
    _swapReserveTokenForGoodDollar(bytes(L0.revertReason()));
  }

  function test_tradingLimitsAreEnforced_reserveTokenInflowLimit1() public {
    _swapUntilReserveTokenLimit1_onInflow();
    _swapReserveTokenForGoodDollar(bytes(L1.revertReason()));
  }

  function test_tradingLimitsAreEnforced_reserveTokenInflowGlobalLimit() public {
    _swapUntilReserveTokenGlobalLimit_onInflow();
    _swapReserveTokenForGoodDollar(bytes(LG.revertReason()));
  }

  /**
   * @notice Swaps G$ for cUSD with the maximum amount allowed per swap
   * @param revertReason An optional revert reason to expect, if swap should revert.
   * @dev Pass an empty string when not expecting a revert.
   */
  function _swapGoodDollarForReserveToken(bytes memory revertReason) internal {
    ITradingLimits.Config memory config = getTradingLimitsConfig(address(reserveToken));

    // Get the max amount we can swap in a single transaction before we hit L0
    uint256 maxPerSwapInWei = uint256(uint48(config.limit0)) * 1e18;
    uint256 inflowRequiredForAmountOut = goodDollarExchangeProvider.getAmountIn({
      exchangeId: exchangeId,
      tokenIn: address(goodDollarToken),
      tokenOut: address(reserveToken),
      amountOut: maxPerSwapInWei
    });

    mintGoodDollar(inflowRequiredForAmountOut, trader);

    vm.startPrank(trader);
    goodDollarToken.approve(address(broker), inflowRequiredForAmountOut);

    // If a revertReason was provided, expect a revert with that reason
    if (revertReason.length > 0) {
      vm.expectRevert(revertReason);
    }
    broker.swapOut({
      exchangeProvider: address(goodDollarExchangeProvider),
      exchangeId: exchangeId,
      tokenIn: address(goodDollarToken),
      tokenOut: address(reserveToken),
      amountOut: maxPerSwapInWei,
      amountInMax: type(uint256).max
    });
    vm.stopPrank();
  }

  function _swapUntilReserveTokenLimit0_onOutflow() internal {
    _swapGoodDollarForReserveToken({ revertReason: "" });
  }

  function _swapUntilReserveTokenLimit1_onOutflow() internal {
    // Get the trading limits config and state for the reserve token
    ITradingLimits.Config memory config = getTradingLimitsConfig(address(reserveToken));
    ITradingLimits.State memory state = getRefreshedTradingLimitsState(address(reserveToken));
    console.log(unicode"🏷️ [%d] Swap until L1=%d on outflow", block.timestamp, uint48(config.limit1));

    // Get the max amount we can swap in a single transaction before we hit L0
    int48 maxPerSwap = config.limit0;

    // Swap until right before we would hit the L1 limit.
    // We swap in `maxPerSwap` increments and timewarp
    // by `timestep0 + 1` seconds so we avoid hitting L0.
    uint256 i;
    while (state.netflow1 - maxPerSwap >= -1 * config.limit1) {
      skip(config.timestep0 + 1);
      // Check that there's still outflow to trade as sometimes we hit LG while
      // still having a bit of L1 left, which causes an infinite loop.
      if (maxOutflow(address(reserveToken)) == 0) {
        break;
      }

      _swapUntilReserveTokenLimit0_onOutflow();

      config = getTradingLimitsConfig(address(reserveToken));
      state = getTradingLimitsState(address(reserveToken));

      i++;
      require(i <= 10, "possible infinite loopL more than 10 iterations");
    }
    skip(config.timestep0 + 1);
  }

  function _swapUntilReserveTokenGlobalLimit_onOutflow() internal {
    // Get the trading limits config and state for the reserve token
    ITradingLimits.Config memory config = getTradingLimitsConfig(address(reserveToken));
    ITradingLimits.State memory state = getRefreshedTradingLimitsState(address(reserveToken));
    console.log(unicode"🏷️ [%d] Swap until LG=%d on outflow", block.timestamp, uint48(config.limitGlobal));

    int48 maxPerSwap = config.limit0;
    uint256 i;
    while (state.netflowGlobal - maxPerSwap >= config.limitGlobal * -1) {
      skip(config.timestep1 + 1);
      _swapUntilReserveTokenLimit1_onOutflow();
      state = getRefreshedTradingLimitsState(address(reserveToken));
      i++;
      require(i <= 50, "possible infinite loop: more than 50 iterations");
    }
    skip(config.timestep1 + 1);
  }

  /**
   * @notice Swaps cUSD for G$ with the maximum amount allowed per swap
   * @param revertReason An optional revert reason to expect, if swap should revert.
   * @dev Pass an empty string when not expecting a revert.
   */
  function _swapReserveTokenForGoodDollar(bytes memory revertReason) internal {
    ITradingLimits.Config memory config = getTradingLimitsConfig(address(reserveToken));

    // Get the max amount we can swap in a single transaction before we hit L0
    uint256 maxPerSwapInWei = uint256(uint48(config.limit0)) * 1e18;
    deal({ token: address(reserveToken), to: trader, give: maxPerSwapInWei });

    vm.startPrank(trader);
    reserveToken.approve(address(broker), maxPerSwapInWei);

    // If a revertReason was provided, expect a revert with that reason
    if (revertReason.length > 0) {
      vm.expectRevert(revertReason);
    }
    broker.swapIn({
      exchangeProvider: address(goodDollarExchangeProvider),
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(goodDollarToken),
      amountIn: maxPerSwapInWei,
      amountOutMin: 0
    });
    vm.stopPrank();
  }

  function _swapUntilReserveTokenLimit0_onInflow() internal {
    _swapReserveTokenForGoodDollar({ revertReason: "" });
  }

  function _swapUntilReserveTokenLimit1_onInflow() internal {
    // Get the trading limits config and state for the reserve token
    ITradingLimits.Config memory config = getTradingLimitsConfig(address(reserveToken));
    ITradingLimits.State memory state = getRefreshedTradingLimitsState(address(reserveToken));
    console.log(unicode"🏷️ [%d] Swap until L1=%d on inflow", block.timestamp, uint48(config.limit1));

    // Get the max amount we can swap in a single transaction before we hit L0
    int48 maxPerSwap = config.limit0;

    // Swap until right before we would hit the L1 limit.
    // We swap in `maxPerSwap` increments and timewarp
    // by `timestep0 + 1` seconds so we avoid hitting L0.
    while (state.netflow1 + maxPerSwap <= config.limit1) {
      skip(config.timestep0 + 1);
      _swapUntilReserveTokenLimit0_onInflow();
      config = getTradingLimitsConfig(address(reserveToken));
      state = getTradingLimitsState(address(reserveToken));

      if (state.netflowGlobal == config.limitGlobal) {
        console.log(unicode"🚨 LG reached during L1 inflow");
        break;
      }
    }
    skip(config.timestep0 + 1);
  }

  function _swapUntilReserveTokenGlobalLimit_onInflow() internal {
    // Get the trading limits config and state for the reserve token
    ITradingLimits.Config memory config = getTradingLimitsConfig(address(reserveToken));
    ITradingLimits.State memory state = getRefreshedTradingLimitsState(address(reserveToken));
    console.log(unicode"🏷️ [%d] Swap until LG=%d on inflow", block.timestamp, uint48(config.limitGlobal));

    int48 maxPerSwap = config.limit0;
    uint256 i;
    while (state.netflowGlobal + maxPerSwap <= config.limitGlobal) {
      skip(config.timestep1 + 1);
      _swapUntilReserveTokenLimit1_onInflow();
      state = getRefreshedTradingLimitsState(address(reserveToken));
      i++;
      require(i <= 50, "possible infinite loop: more than 50 iterations");
    }
    skip(config.timestep1 + 1);
  }
}
