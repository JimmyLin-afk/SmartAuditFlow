// SPDX-License-Identifier: GPL-3.0-or-later
// solhint-disable func-name-mixedcase, var-name-mixedcase, state-visibility, const-name-snakecase, max-states-count
pragma solidity ^0.8;

import { ProtocolTest } from "./ProtocolTest.sol";
import { IERC20 } from "contracts/interfaces/IERC20.sol";

import { IExchangeProvider } from "contracts/interfaces/IExchangeProvider.sol";
import { IBiPoolManager } from "contracts/interfaces/IBiPoolManager.sol";

import { FixidityLib } from "celo/contracts/common/FixidityLib.sol";

// forge test --match-contract BrokerIntegration -vvv
contract BrokerIntegrationTest is ProtocolTest {
  using FixidityLib for FixidityLib.Fraction;

  address trader;

  function setUp() public override {
    super.setUp();

    trader = makeAddr("trader");

    deal(address(cUSDToken), trader, 10 ** 22, true); // Mint 10k to trader
    deal(address(cEURToken), trader, 10 ** 22, true); // Mint 10k to trader

    deal(address(celoToken), trader, 1000 * 10 ** 18, true); // Gift 10k to trader

    deal(address(celoToken), address(reserve), 10 ** (6 + 18), true); // Gift 1Mil Celo to reserve
    deal(address(usdcToken), address(reserve), 10 ** (6 + 6), true); // Gift 1Mil USDC to reserve
  }

  /**
   * @notice Test helper function to do swap in
   */
  function doSwapIn(
    bytes32 poolId,
    uint256 amountIn,
    address tokenIn,
    address tokenOut
  ) public returns (uint256 expectedOut, uint256 actualOut) {
    // Get exchange provider from broker
    address[] memory exchangeProviders = broker.getExchangeProviders();
    assertEq(exchangeProviders.length, 1);

    expectedOut = broker.getAmountOut(exchangeProviders[0], poolId, tokenIn, tokenOut, amountIn);

    changePrank(trader);
    IERC20(tokenIn).approve(address(broker), amountIn);

    // Execute swap
    actualOut = broker.swapIn(address(exchangeProviders[0]), poolId, tokenIn, tokenOut, amountIn, 0);
  }

  function test_swap_whenBucketTriggerConditionsAreMet_shouldTriggerBucketUpdate() public {
    IBiPoolManager.PoolExchange memory pool = biPoolManager.getPoolExchange(pair_cUSD_bridgedUSDC_ID);

    // FF by bucket update frequency time
    vm.warp(pool.config.referenceRateResetFrequency + pool.lastBucketUpdate);

    // Median report recent == true
    (uint256 numerator, uint256 denominator) = sortedOracles.medianRate(pool.config.referenceRateFeedID);
    setMedianRate(pool.config.referenceRateFeedID, FixidityLib.newFixedFraction(numerator, denominator).unwrap());

    changePrank(trader);
    IERC20(address(cUSDToken)).approve(address(broker), cUSDToken.totalSupply());

    vm.expectEmit(false, false, false, false);
    emit BucketsUpdated(pair_cUSD_bridgedUSDC_ID, 0, 0);

    broker.swapIn(address(biPoolManager), pair_cUSD_bridgedUSDC_ID, address(cUSDToken), address(usdcToken), 1, 0);
  }

  function test_getExchangeProviders_shouldReturnProviderWithCorrectExchanges() public view {
    address[] memory exchangeProviders = broker.getExchangeProviders();
    assertEq(exchangeProviders.length, 1);

    IExchangeProvider.Exchange[] memory exchanges = IExchangeProvider(exchangeProviders[0]).getExchanges();
    assertEq(exchanges.length, 6);

    IExchangeProvider.Exchange memory exchange;
    for (uint256 i = 0; i < exchanges.length; i++) {
      exchange = exchanges[i];
      assert(
        exchange.assets[0] == address(cUSDToken) ||
          exchange.assets[0] == address(cEURToken) ||
          exchange.assets[0] == address(eXOFToken)
      );
      assert(
        exchange.assets[1] == address(cEURToken) ||
          exchange.assets[1] == address(celoToken) ||
          exchange.assets[1] == address(usdcToken) ||
          exchange.assets[1] == address(eurocToken)
      );
    }
  }

  function test_swapIn_cUSDToBridgedUSDC() public {
    uint256 amountIn = 1000 * 10 ** 18; // 1k (18 decimals)
    IERC20 tokenIn = IERC20(address(cUSDToken));
    IERC20 tokenOut = IERC20(address(usdcToken));
    bytes32 poolId = pair_cUSD_bridgedUSDC_ID;

    // Get amounts before swap
    uint256 traderTokenInBefore = tokenIn.balanceOf(trader);
    uint256 traderTokenOutBefore = tokenOut.balanceOf(trader);
    uint256 reserveCollateralBalanceBefore = tokenOut.balanceOf(address(reserve));
    uint256 StableAssetSupplyBefore = tokenIn.totalSupply();

    // Execute swap
    (uint256 expectedOut, uint256 actualOut) = doSwapIn(poolId, amountIn, address(tokenIn), address(tokenOut));
    assertEq(actualOut, 995 * 10 ** 6); // 0.9995k (6 decimals)

    // Get amounts after swap
    uint256 traderTokenInAfter = tokenIn.balanceOf(trader);
    uint256 traderTokenOutAfter = tokenOut.balanceOf(trader);
    uint256 reserveCollateralBalanceAfter = tokenOut.balanceOf(address(reserve));
    uint256 StableAssetSupplyAfter = tokenIn.totalSupply();

    // getAmountOut == swapOut
    assertEq(expectedOut, actualOut);
    // Trader token in balance decreased
    assertEq(traderTokenInBefore - traderTokenInAfter, amountIn);
    // Trader token out balance increased
    assertEq(traderTokenOutBefore + expectedOut, traderTokenOutAfter);
    // Reserve collateral asset balance decreased
    assertEq(reserveCollateralBalanceBefore - expectedOut, reserveCollateralBalanceAfter);
    // Stable asset supply decrease from burn
    assertEq(StableAssetSupplyBefore - amountIn, StableAssetSupplyAfter);
  }

  function test_swapIn_cEURToBridgedUSDC() public {
    uint256 amountIn = 1000 * 10 ** 18; // 1k
    IERC20 tokenIn = IERC20(address(cEURToken));
    IERC20 tokenOut = IERC20(address(usdcToken));
    bytes32 poolId = pair_cEUR_bridgedUSDC_ID;

    // Get amounts before swap
    uint256 traderTokenInBefore = tokenIn.balanceOf(trader);
    uint256 traderTokenOutBefore = tokenOut.balanceOf(trader);
    uint256 reserveCollateralBalanceBefore = tokenOut.balanceOf(address(reserve));
    uint256 stableAssetSupplyBefore = tokenIn.totalSupply();

    // Execute swap
    (uint256 expectedOut, uint256 actualOut) = doSwapIn(poolId, amountIn, address(tokenIn), address(tokenOut));

    // Get amounts after swap
    uint256 traderTokenInAfter = tokenIn.balanceOf(trader);
    uint256 traderTokenOutAfter = tokenOut.balanceOf(trader);
    uint256 reserveCollateralBalanceAfter = tokenOut.balanceOf(address(reserve));
    uint256 stableAssetSupplyAfter = tokenIn.totalSupply();

    // getAmountOut == swapOut
    assertEq(expectedOut, actualOut);
    // Trader token in balance decreased
    assertEq(traderTokenInBefore - traderTokenInAfter, amountIn);
    // Trader token out balance increased
    assertEq(traderTokenOutBefore + expectedOut, traderTokenOutAfter);
    // Reserve collateral asset balance decreased
    assertEq(reserveCollateralBalanceBefore - expectedOut, reserveCollateralBalanceAfter);
    // Stable asset supply decrease from burn
    assertEq(stableAssetSupplyBefore - amountIn, stableAssetSupplyAfter);
  }

  function test_swapIn_cEURTocUSD() public {
    uint256 amountIn = 1000 * 10 ** 18; // 1k
    IERC20 tokenIn = IERC20(address(cEURToken));
    IERC20 tokenOut = IERC20(address(cUSDToken));
    bytes32 poolId = pair_cUSD_cEUR_ID;

    // Get amounts before swap
    uint256 traderTokenInBefore = tokenIn.balanceOf(trader);
    uint256 traderTokenOutBefore = tokenOut.balanceOf(trader);

    uint256 tokenInSupplyBefore = tokenIn.totalSupply();
    uint256 tokenOutSupplyBefore = tokenOut.totalSupply();

    // Execute swap
    (uint256 expectedOut, uint256 actualOut) = doSwapIn(poolId, amountIn, address(tokenIn), address(tokenOut));

    // Get amounts after swap
    uint256 traderTokenInAfter = tokenIn.balanceOf(trader);
    uint256 traderTokenOutAfter = tokenOut.balanceOf(trader);

    uint256 tokenInSupplyAfter = tokenIn.totalSupply();
    uint256 tokenOutSupplyAfter = tokenOut.totalSupply();

    // getAmountOut == swapOut
    assertEq(expectedOut, actualOut);
    // Trader token in balance decreased
    assertEq(traderTokenInBefore - traderTokenInAfter, amountIn);
    // Trader token out balance increased
    assertEq(traderTokenOutBefore + expectedOut, traderTokenOutAfter);
    // Token out stable asset supply increase from mint
    assertEq(tokenOutSupplyBefore + expectedOut, tokenOutSupplyAfter);
    // Token in stable asset supply decrease from burn
    assertEq(tokenInSupplyBefore - amountIn, tokenInSupplyAfter);
  }

  function test_swapIn_cUSDTocEUR() public {
    uint256 amountIn = 1000 * 10 ** 18; // 1k
    IERC20 tokenIn = IERC20(address(cUSDToken));
    IERC20 tokenOut = IERC20(address(cEURToken));
    bytes32 poolId = pair_cUSD_cEUR_ID;

    // Get amounts before swap
    uint256 traderTokenInBefore = tokenIn.balanceOf(trader);
    uint256 traderTokenOutBefore = tokenOut.balanceOf(trader);

    uint256 tokenInSupplyBefore = tokenIn.totalSupply();
    uint256 tokenOutSupplyBefore = tokenOut.totalSupply();

    // Execute swap
    (uint256 expectedOut, uint256 actualOut) = doSwapIn(poolId, amountIn, address(tokenIn), address(tokenOut));

    // Get amounts after swap
    uint256 traderTokenInAfter = tokenIn.balanceOf(trader);
    uint256 traderTokenOutAfter = tokenOut.balanceOf(trader);

    uint256 tokenInSupplyAfter = tokenIn.totalSupply();
    uint256 tokenOutSupplyAfter = tokenOut.totalSupply();

    // getAmountOut == swapOut
    assertEq(expectedOut, actualOut);
    // Trader token in balance decreased
    assertEq(traderTokenInBefore - traderTokenInAfter, amountIn);
    // Trader token out balance increased
    assertEq(traderTokenOutBefore + expectedOut, traderTokenOutAfter);
    // Token out stable asset supply increase from mint
    assertEq(tokenOutSupplyBefore + expectedOut, tokenOutSupplyAfter);
    // Token in stable asset supply decrease from burn
    assertEq(tokenInSupplyBefore - amountIn, tokenInSupplyAfter);
  }

  function test_swapIn_CELOTocEUR() public {
    uint256 amountIn = 1000 * 10 ** 18; // 1k
    IERC20 tokenIn = IERC20(address(celoToken));
    IERC20 tokenOut = IERC20(address(cEURToken));
    bytes32 poolId = pair_cEUR_CELO_ID;

    // Get amounts before swap
    uint256 traderTokenInBefore = tokenIn.balanceOf(trader);
    uint256 traderTokenOutBefore = tokenOut.balanceOf(trader);

    uint256 reserveCollateralBalanceBefore = tokenIn.balanceOf(address(reserve));
    uint256 StableAssetSupplyBefore = tokenOut.totalSupply();

    // Execute swap
    (uint256 expectedOut, uint256 actualOut) = doSwapIn(poolId, amountIn, address(tokenIn), address(tokenOut));

    // Get amounts after swap
    uint256 traderTokenInAfter = tokenIn.balanceOf(trader);
    uint256 traderTokenOutAfter = tokenOut.balanceOf(trader);

    uint256 reserveCollateralBalanceAfter = tokenIn.balanceOf(address(reserve));
    uint256 StableAssetSupplyAfter = tokenOut.totalSupply();

    // getAmountOut == swapOut
    assertEq(expectedOut, actualOut);
    // Trader token in balance decreased
    assertEq(traderTokenInBefore - traderTokenInAfter, amountIn);
    // Trader token out balance increased
    assertEq(traderTokenOutBefore + expectedOut, traderTokenOutAfter);
    // Reserve collateral asset balance increase
    assertEq(reserveCollateralBalanceBefore + amountIn, reserveCollateralBalanceAfter);
    // Stable asset supply increase from mint
    assertEq(StableAssetSupplyBefore + expectedOut, StableAssetSupplyAfter);
  }

  function test_swapIn_CELOTocUSD() public {
    uint256 amountIn = 1000 * 10 ** 18; // 1k
    IERC20 tokenIn = IERC20(address(celoToken));
    IERC20 tokenOut = IERC20(address(cUSDToken));
    bytes32 poolId = pair_cUSD_CELO_ID;

    // Get amounts before swap
    uint256 traderTokenInBefore = tokenIn.balanceOf(trader);
    uint256 traderTokenOutBefore = tokenOut.balanceOf(trader);

    uint256 reserveCollateralBalanceBefore = tokenIn.balanceOf(address(reserve));
    uint256 StableAssetSupplyBefore = tokenOut.totalSupply();

    // Execute swap
    (uint256 expectedOut, uint256 actualOut) = doSwapIn(poolId, amountIn, address(tokenIn), address(tokenOut));

    // Get amounts after swap
    uint256 traderTokenInAfter = tokenIn.balanceOf(trader);
    uint256 traderTokenOutAfter = tokenOut.balanceOf(trader);

    uint256 reserveCollateralBalanceAfter = tokenIn.balanceOf(address(reserve));
    uint256 StableAssetSupplyAfter = tokenOut.totalSupply();

    // getAmountOut == swapOut
    assertEq(expectedOut, actualOut);
    // Trader token in balance decreased
    assertEq(traderTokenInBefore - traderTokenInAfter, amountIn);
    // Trader token out balance increased
    assertEq(traderTokenOutBefore + expectedOut, traderTokenOutAfter);
    // Reserve collateral asset balance increase
    assertEq(reserveCollateralBalanceBefore + amountIn, reserveCollateralBalanceAfter);
    // Stable asset supply increase from mint
    assertEq(StableAssetSupplyBefore + expectedOut, StableAssetSupplyAfter);
  }
}
