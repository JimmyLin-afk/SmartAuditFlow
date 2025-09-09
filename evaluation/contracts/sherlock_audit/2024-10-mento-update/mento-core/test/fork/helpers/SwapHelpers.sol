// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

import { ExchangeForkTest } from "../ExchangeForkTest.sol";

import { IBiPoolManager } from "contracts/interfaces/IBiPoolManager.sol";
import { IERC20 } from "contracts/interfaces/IERC20.sol";

import { OracleHelpers } from "./OracleHelpers.sol";

library SwapHelpers {
  using OracleHelpers for ExchangeForkTest;

  function ticker(ExchangeForkTest ctx) internal view returns (string memory) {
    return string(abi.encodePacked(IERC20(ctx.asset(0)).symbol(), "/", IERC20(ctx.asset(1)).symbol()));
  }

  function maxSwapIn(ExchangeForkTest ctx, uint256 desired, address from, address to) internal view returns (uint256) {
    IBiPoolManager.PoolExchange memory pool = ctx.getPool();
    uint256 leftInBucket = (pool.asset0 == to ? pool.bucket0 : pool.bucket1) - 1;
    (, , , , bool shouldUpdate) = ctx.shouldUpdateBuckets();
    if (shouldUpdate) {
      (uint256 bucket0, uint256 bucket1) = ctx.getUpdatedBuckets();
      leftInBucket = (pool.asset0 == to ? bucket0 : bucket1) - 1;
    }
    leftInBucket = leftInBucket / ctx.exchangeProvider().tokenPrecisionMultipliers(to);
    uint256 maxPossible = getAmountIn(ctx, from, to, leftInBucket);
    return maxPossible > desired ? desired : maxPossible;
  }

  function maxSwapOut(ExchangeForkTest ctx, uint256 desired, address to) internal view returns (uint256 max) {
    IBiPoolManager.PoolExchange memory pool = ctx.getPool();
    uint256 leftInBucket = (pool.asset0 == to ? pool.bucket0 : pool.bucket1);
    (, , , , bool shouldUpdate) = ctx.shouldUpdateBuckets();
    if (shouldUpdate) {
      (uint256 bucket0, uint256 bucket1) = ctx.getUpdatedBuckets();
      leftInBucket = (pool.asset0 == to ? bucket0 : bucket1) - 1;
    }

    leftInBucket = leftInBucket / ctx.exchangeProvider().tokenPrecisionMultipliers(to);
    return leftInBucket > desired ? desired : leftInBucket;
  }

  function getAmountOut(
    ExchangeForkTest ctx,
    address from,
    address to,
    uint256 sellAmount
  ) internal view returns (uint256) {
    return ctx.broker().getAmountOut(ctx.exchangeProviderAddr(), ctx.exchangeId(), from, to, sellAmount);
  }

  function getAmountIn(
    ExchangeForkTest ctx,
    address from,
    address to,
    uint256 buyAmount
  ) internal view returns (uint256) {
    return ctx.broker().getAmountIn(ctx.exchangeProviderAddr(), ctx.exchangeId(), from, to, buyAmount);
  }
}
