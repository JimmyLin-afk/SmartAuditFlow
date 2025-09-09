// SPDX-License-Identifier: GPL-3.0-or-later
// solhint-disable func-name-mixedcase, var-name-mixedcase, state-visibility, const-name-snakecase, max-states-count
pragma solidity ^0.8;

import { ProtocolTest } from "./ProtocolTest.sol";

import { IERC20 } from "contracts/interfaces/IERC20.sol";
import { IBiPoolManager } from "contracts/interfaces/IBiPoolManager.sol";

contract ConstantSumIntegrationTest is ProtocolTest {
  address trader = makeAddr("trader");

  function setUp() public override {
    super.setUp();

    deal(address(cUSDToken), trader, 10 ** 22, true); // Mint 10k to trader
    deal(address(usdcToken), address(reserve), 10 ** (6 + 6), true); // Gift 1Mil USDC to reserve
  }

  /**
   * @notice Test helper function to do swap in
   */
  function doSwapIn(
    bytes32 poolId,
    uint256 amountIn,
    address tokenIn,
    address tokenOut,
    bool shouldRevert
  ) public returns (uint256 expectedOut, uint256 actualOut) {
    // Get exchange provider from broker
    address[] memory exchangeProviders = broker.getExchangeProviders();
    assertEq(exchangeProviders.length, 1);
    if (shouldRevert) {
      vm.expectRevert("no valid median");
    }
    expectedOut = broker.getAmountOut(exchangeProviders[0], poolId, tokenIn, tokenOut, amountIn);

    vm.prank(trader);
    IERC20(tokenIn).approve(address(broker), amountIn);

    // Execute swap
    if (shouldRevert) {
      vm.expectRevert("no valid median");
    }
    vm.prank(trader);
    actualOut = broker.swapIn(address(exchangeProviders[0]), poolId, tokenIn, tokenOut, amountIn, 0);
  }

  function test_swap_whenConstantSum_pricesShouldStayTheSameInBetweenBucketUpdates() public {
    uint256 amountIn = 5000 * 10 ** 18; // 5k cUSD
    IERC20 tokenIn = IERC20(address(cUSDToken));
    IERC20 tokenOut = IERC20(address(usdcToken));
    bytes32 poolId = pair_cUSD_bridgedUSDC_ID;

    // Buckets before
    IBiPoolManager.PoolExchange memory exchangeBefore = biPoolManager.getPoolExchange(poolId);
    assertEq(exchangeBefore.bucket0, 1e24);
    assertEq(exchangeBefore.bucket1, 1e24);

    // Execute swap cUSD -> USDC
    (, uint256 actualOut) = doSwapIn(poolId, amountIn, address(tokenIn), address(tokenOut), false);
    assertEq(actualOut, 5000 * 0.995 * 10 ** 6); //  4975(6 decimals)

    IBiPoolManager.PoolExchange memory exchangeAfter1 = biPoolManager.getPoolExchange(poolId);
    assertEq(exchangeBefore.bucket0, exchangeAfter1.bucket0);
    assertEq(exchangeBefore.bucket1, exchangeAfter1.bucket1);

    // Execute second swap cUSD -> USDC
    (, uint256 actualOut2) = doSwapIn(poolId, amountIn, address(tokenIn), address(tokenOut), false);
    assertEq(actualOut, actualOut2);

    IBiPoolManager.PoolExchange memory exchangeAfter2 = biPoolManager.getPoolExchange(poolId);
    assertEq(exchangeBefore.bucket0, exchangeAfter2.bucket0);
    assertEq(exchangeBefore.bucket1, exchangeAfter2.bucket1);

    // Execute swap USDC -> cUSD
    amountIn = 5000 * 10 ** 6; // 5k USDC
    (, uint256 actualOut3) = doSwapIn(poolId, amountIn, address(tokenOut), address(tokenIn), false);
    assertEq(actualOut3, 5000 * 0.995 * 10 ** 18); //  4975(18 decimals)

    IBiPoolManager.PoolExchange memory exchangeAfter3 = biPoolManager.getPoolExchange(poolId);
    assertEq(exchangeBefore.bucket0, exchangeAfter3.bucket0);
    assertEq(exchangeBefore.bucket1, exchangeAfter3.bucket1);
  }

  function test_swap_whenConstantSum_pricesShouldChangeWhenBucketRatioChanges() public {
    uint256 amountIn = 5000 * 10 ** 18; // 5k cUSD
    IERC20 tokenIn = IERC20(address(cUSDToken));
    IERC20 tokenOut = IERC20(address(usdcToken));
    bytes32 poolId = pair_cUSD_bridgedUSDC_ID;
    IBiPoolManager.PoolExchange memory exchangeBefore = biPoolManager.getPoolExchange(poolId);

    assertEq(exchangeBefore.bucket0, 1e24); // stable pool reset size
    assertEq(exchangeBefore.bucket1, exchangeBefore.bucket0); // current median is 1

    // Execute swap cUSD -> USDC
    (, uint256 actualOut) = doSwapIn(poolId, amountIn, address(tokenIn), address(tokenOut), false);
    assertEq(actualOut, 5000 * 0.995 * 10 ** 6); //  4975(6 decimals)

    vm.warp(block.timestamp + exchangeBefore.config.referenceRateResetFrequency); // time travel enable bucket update
    setMedianRate(cUSD_bridgedUSDC_referenceRateFeedID, 1e24 * 1.1); // new valid Median that wont trigger breaker 0.13

    // Execute swap cUSD -> USDC
    (, uint256 actualOut2) = doSwapIn(poolId, amountIn, address(tokenIn), address(tokenOut), false);
    assertTrue(actualOut > actualOut2); //  4522(6 decimals)

    IBiPoolManager.PoolExchange memory exchangeAfter = biPoolManager.getPoolExchange(poolId);
    assertEq(exchangeAfter.bucket0, 1e24); // stable pool reset size
    assertEq(exchangeAfter.bucket1, 909090909090909090909090); // current median is 1.1
  }

  function test_swap_whenConstantSumAndOldestReportExpired_shouldRevert() public {
    uint256 amountIn = 5000 * 10 ** 18; // 5k cUSD
    IERC20 tokenIn = IERC20(address(cUSDToken));
    IERC20 tokenOut = IERC20(address(usdcToken));
    bytes32 poolId = pair_cUSD_bridgedUSDC_ID;

    // Oldest report is not expired
    uint256 tokenExpiry = sortedOracles.getTokenReportExpirySeconds(cUSD_bridgedUSDC_referenceRateFeedID);
    (bool expired, ) = sortedOracles.isOldestReportExpired(cUSD_bridgedUSDC_referenceRateFeedID);
    assertEq(false, expired);

    // Expire report
    vm.warp(block.timestamp + tokenExpiry);
    (expired, ) = sortedOracles.isOldestReportExpired(cUSD_bridgedUSDC_referenceRateFeedID);
    assertEq(true, expired);

    // Execute swap cUSD -> USDC with shouldRevert true
    doSwapIn(poolId, amountIn, address(tokenIn), address(tokenOut), true);
  }

  function test_swap_whenConstantSumAndMedianExpired_shouldRevert() public {
    uint256 amountIn = 5000 * 10 ** 18; // 5k cUSD
    IERC20 tokenIn = IERC20(address(cUSDToken));
    IERC20 tokenOut = IERC20(address(usdcToken));
    bytes32 poolId = pair_cUSD_bridgedUSDC_ID;
    IBiPoolManager.PoolExchange memory exchange = biPoolManager.getPoolExchange(poolId);

    // Median is recent enough
    bool medianReportRecent = sortedOracles.medianTimestamp(cUSD_bridgedUSDC_referenceRateFeedID) >
      block.timestamp - exchange.config.referenceRateResetFrequency;
    assertEq(true, medianReportRecent);

    // Expire median
    vm.warp(block.timestamp + exchange.config.referenceRateResetFrequency);
    medianReportRecent =
      sortedOracles.medianTimestamp(cUSD_bridgedUSDC_referenceRateFeedID) >
      block.timestamp - exchange.config.referenceRateResetFrequency;
    assertEq(false, medianReportRecent);

    // Execute swap cUSD -> USDC with shouldRevert true
    doSwapIn(poolId, amountIn, address(tokenIn), address(tokenOut), true);
  }

  function test_swap_whenConstantSumAndNotEnoughReports_shouldRevert() public {
    uint256 amountIn = 5000 * 10 ** 18; // 5k cUSD
    IERC20 tokenIn = IERC20(address(cUSDToken));
    IERC20 tokenOut = IERC20(address(usdcToken));
    bytes32 poolId = pair_cUSD_bridgedUSDC_ID;
    IBiPoolManager.PoolExchange memory exchange = biPoolManager.getPoolExchange(poolId);

    // Enough reports
    uint256 numReports = sortedOracles.numRates(cUSD_bridgedUSDC_referenceRateFeedID);
    assertTrue(numReports > exchange.config.minimumReports);

    // remove reports by removing oracles
    while (numReports >= exchange.config.minimumReports) {
      address oracle = sortedOracles.oracles(cUSD_bridgedUSDC_referenceRateFeedID, 0);
      sortedOracles.removeOracle(cUSD_bridgedUSDC_referenceRateFeedID, oracle, 0);
      numReports = sortedOracles.numRates(cUSD_bridgedUSDC_referenceRateFeedID);
    }

    // Execute swap cUSD -> USDC with shouldRevert true
    doSwapIn(poolId, amountIn, address(tokenIn), address(tokenOut), true);
  }
}
