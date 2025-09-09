// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

// Libraries
import { TokenHelpers } from "../helpers/TokenHelpers.sol";
import { TradingLimitHelpers } from "../helpers/TradingLimitHelpers.sol";

// Interfaces
import { IBancorExchangeProvider } from "contracts/interfaces/IBancorExchangeProvider.sol";

// Contracts
import { GoodDollarBaseForkTest } from "./GoodDollarBaseForkTest.sol";

contract GoodDollarSwapForkTest is GoodDollarBaseForkTest {
  using TradingLimitHelpers for *;
  using TokenHelpers for *;

  constructor(uint256 _chainId) GoodDollarBaseForkTest(_chainId) {}

  function setUp() public override {
    super.setUp();
  }

  function test_swapIn_reserveTokenToGoodDollar() public {
    uint256 amountIn = 1000 * 1e18;

    uint256 reserveBalanceBefore = reserveToken.balanceOf(address(goodDollarReserve));
    uint256 priceBefore = IBancorExchangeProvider(address(goodDollarExchangeProvider)).currentPrice(exchangeId);

    // Calculate the expected amount of G$ to receive for `amountIn` cUSD
    uint256 expectedAmountOut = broker.getAmountOut(
      address(goodDollarExchangeProvider),
      exchangeId,
      address(reserveToken),
      address(goodDollarToken),
      amountIn
    );

    // Give trader required amount of cUSD to swap
    deal({ token: address(reserveToken), to: trader, give: amountIn });

    vm.startPrank(trader);
    // Trader approves the broker to spend their cUSD
    reserveToken.approve({ spender: address(broker), amount: amountIn });

    // Broker swaps `amountIn` of trader's cUSD for G$
    broker.swapIn({
      exchangeProvider: address(goodDollarExchangeProvider),
      exchangeId: exchangeId,
      tokenIn: address(reserveToken),
      tokenOut: address(goodDollarToken),
      amountIn: amountIn,
      amountOutMin: expectedAmountOut
    });
    vm.stopPrank();

    uint256 priceAfter = IBancorExchangeProvider(address(goodDollarExchangeProvider)).currentPrice(exchangeId);
    uint256 reserveBalanceAfter = reserveToken.balanceOf(address(goodDollarReserve));

    assertEq(expectedAmountOut, goodDollarToken.balanceOf(trader));
    assertEq(reserveBalanceBefore + amountIn, reserveBalanceAfter);
    assertTrue(priceBefore < priceAfter);
  }

  function test_swapIn_goodDollarToReserveToken() public {
    uint256 amountIn = 1000 * 1e18;

    uint256 reserveBalanceBefore = reserveToken.balanceOf(address(goodDollarReserve));
    uint256 priceBefore = IBancorExchangeProvider(address(goodDollarExchangeProvider)).currentPrice(exchangeId);
    uint256 expectedAmountOut = broker.getAmountOut(
      address(goodDollarExchangeProvider),
      exchangeId,
      address(goodDollarToken),
      address(reserveToken),
      amountIn
    );

    mintGoodDollar(amountIn, trader);

    vm.startPrank(trader);
    goodDollarToken.approve(address(broker), amountIn);
    broker.swapIn(
      address(goodDollarExchangeProvider),
      exchangeId,
      address(goodDollarToken),
      address(reserveToken),
      amountIn,
      expectedAmountOut
    );
    uint256 priceAfter = IBancorExchangeProvider(address(goodDollarExchangeProvider)).currentPrice(exchangeId);
    uint256 reserveBalanceAfter = reserveToken.balanceOf(address(goodDollarReserve));

    assertEq(expectedAmountOut, reserveToken.balanceOf(trader));
    assertEq(reserveBalanceBefore - expectedAmountOut, reserveBalanceAfter);
    assertTrue(priceAfter < priceBefore);
  }

  function test_swapOut_reserveTokenToGoodDollar() public {
    uint256 amountOut = 1000 * 1e18;
    uint256 reserveBalanceBefore = reserveToken.balanceOf(address(goodDollarReserve));
    uint256 priceBefore = IBancorExchangeProvider(address(goodDollarExchangeProvider)).currentPrice(exchangeId);
    uint256 expectedAmountIn = broker.getAmountIn(
      address(goodDollarExchangeProvider),
      exchangeId,
      address(reserveToken),
      address(goodDollarToken),
      amountOut
    );

    deal(address(reserveToken), trader, expectedAmountIn);

    vm.startPrank(trader);
    reserveToken.approve(address(broker), expectedAmountIn);
    broker.swapOut(
      address(goodDollarExchangeProvider),
      exchangeId,
      address(reserveToken),
      address(goodDollarToken),
      amountOut,
      expectedAmountIn
    );
    uint256 priceAfter = IBancorExchangeProvider(address(goodDollarExchangeProvider)).currentPrice(exchangeId);
    uint256 reserveBalanceAfter = reserveToken.balanceOf(address(goodDollarReserve));

    assertEq(amountOut, goodDollarToken.balanceOf(trader));
    assertEq(reserveBalanceBefore + expectedAmountIn, reserveBalanceAfter);
    assertTrue(priceBefore < priceAfter);
  }

  function test_swapOut_goodDollarToReserveToken() public {
    uint256 amountOut = 1000 * 1e18;

    uint256 reserveBalanceBefore = reserveToken.balanceOf(address(goodDollarReserve));
    uint256 priceBefore = IBancorExchangeProvider(address(goodDollarExchangeProvider)).currentPrice(exchangeId);
    uint256 expectedAmountIn = broker.getAmountIn(
      address(goodDollarExchangeProvider),
      exchangeId,
      address(goodDollarToken),
      address(reserveToken),
      amountOut
    );

    mintGoodDollar(expectedAmountIn, trader);

    vm.startPrank(trader);
    goodDollarToken.approve(address(broker), expectedAmountIn);
    broker.swapOut(
      address(goodDollarExchangeProvider),
      exchangeId,
      address(goodDollarToken),
      address(reserveToken),
      amountOut,
      expectedAmountIn
    );
    uint256 priceAfter = IBancorExchangeProvider(address(goodDollarExchangeProvider)).currentPrice(exchangeId);
    uint256 reserveBalanceAfter = reserveToken.balanceOf(address(goodDollarReserve));

    assertEq(amountOut, reserveToken.balanceOf(trader));
    assertEq(reserveBalanceBefore - amountOut, reserveBalanceAfter);
    assertTrue(priceAfter < priceBefore);
  }
}
