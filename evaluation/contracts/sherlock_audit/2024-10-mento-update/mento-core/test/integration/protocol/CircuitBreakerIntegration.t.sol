// SPDX-License-Identifier: GPL-3.0-or-later
// solhint-disable func-name-mixedcase, var-name-mixedcase, state-visibility, const-name-snakecase, max-states-count
pragma solidity ^0.8;

import { ProtocolTest } from "./ProtocolTest.sol";

import { FixidityLib } from "celo/contracts/common/FixidityLib.sol";

import { IERC20 } from "contracts/interfaces/IERC20.sol";
import { IBreakerBox } from "contracts/interfaces/IBreakerBox.sol";

contract CircuitBreakerIntegration is ProtocolTest {
  using FixidityLib for FixidityLib.Fraction;

  uint256 fixed1 = FixidityLib.fixed1().unwrap();

  address trader = makeAddr("trader");

  function setUp() public override {
    super.setUp();

    deal(address(cUSDToken), trader, 10 ** 22, true); // Mint 10k to trader
    deal(address(cEURToken), trader, 10 ** 22, true); // Mint 10k to trader
    deal(address(celoToken), address(reserve), 1e24); // Gift 1Mil Celo to reserve
    deal(address(usdcToken), address(reserve), 1e24); // Gift 1Mil USDC to reserve
  }

  /**
   * @notice Test helper function to do swap in
   */
  function doSwapIn(bytes32 poolId, address tokenIn, address tokenOut, bool shouldBreak) public {
    uint256 amountIn = 10 ** 18;
    address[] memory exchangeProviders = broker.getExchangeProviders();
    assertEq(exchangeProviders.length, 1);

    vm.prank(trader);
    IERC20(tokenIn).approve(address(broker), amountIn);

    if (shouldBreak) {
      vm.expectRevert("Trading is suspended for this reference rate");
    }
    // Execute swap
    vm.prank(trader);
    broker.swapIn(exchangeProviders[0], poolId, tokenIn, tokenOut, amountIn, 0);
  }

  function test_medianDeltaBreaker__whenMedianExceedsThresholdAndRecovers_shouldHaltAndRecover() public {
    // Trip median delta breaker for cUSD_CELO_referenceRateFeedID threshold: 15%
    setMedianRate(cUSD_CELO_referenceRateFeedID, 5e23 * 1.151);

    // Try swap with shouldBreak true
    doSwapIn(pair_cUSD_CELO_ID, address(cUSDToken), address(celoToken), true);

    // Check trading mode
    uint8 tradingMode = breakerBox.getRateFeedTradingMode(cUSD_CELO_referenceRateFeedID);
    assertEq(uint256(tradingMode), 3); // 3 = trading halted

    // Cool down breaker and set new median that doesnt exceed threshold
    vm.warp(block.timestamp + 5 minutes);
    setMedianRate(cUSD_CELO_referenceRateFeedID, 5e23);

    // Check trading mode
    tradingMode = breakerBox.getRateFeedTradingMode(cUSD_CELO_referenceRateFeedID);
    assertEq(uint256(tradingMode), 0); // 0 = bidirectional trading

    // Try swap with shouldBreak false -> trading is bidirectional again
    doSwapIn(pair_cUSD_CELO_ID, address(cUSDToken), address(celoToken), false);
  }

  function test_medianDeltaBreaker__whenMedianSubceedsThresholdAndRecovers_shouldHaltAndRecover() public {
    uint256 newMedian = 5e23 - 5e23 * 0.151;
    // Trip median delta breaker for cUSD_CELO_referenceRateFeedID threshold: 15%
    setMedianRate(cUSD_CELO_referenceRateFeedID, newMedian);

    // Try swap with shouldBreak true
    doSwapIn(pair_cUSD_CELO_ID, address(cUSDToken), address(celoToken), true);

    // Check trading mode
    uint8 tradingMode = breakerBox.getRateFeedTradingMode(cUSD_CELO_referenceRateFeedID);
    assertEq(uint256(tradingMode), 3); // 3 = trading halted

    // Cool down breaker and set new median that doesnt exceed threshold
    vm.warp(block.timestamp + 5 minutes);
    newMedian = newMedian + (5e23 - 5e23 * 0.151) * 0.14;
    setMedianRate(cUSD_CELO_referenceRateFeedID, newMedian);

    // Check trading mode
    tradingMode = breakerBox.getRateFeedTradingMode(cUSD_CELO_referenceRateFeedID);
    assertEq(uint256(tradingMode), 0); // 0 = bidirectional trading

    // Try swap with shouldBreak false -> trading is bidirectional again
    doSwapIn(pair_cUSD_CELO_ID, address(cUSDToken), address(celoToken), false);
  }

  function test_valueDeltaBreaker_whenMedianExceedsThresholdAndRecovers_shouldHaltAndRecover() public {
    // Trip value delta breaker for cUSD_bridgedUSDC_referenceRateFeedID threshold: 0.1 * 1e24
    setMedianRate(cUSD_bridgedUSDC_referenceRateFeedID, 1e24 + 1e24 * 0.11);

    // Try swap with shouldBreak true
    doSwapIn(pair_cUSD_bridgedUSDC_ID, address(cUSDToken), address(usdcToken), true);

    // Check trading modes & ensure only value delta breaker tripped
    uint8 rateFeedTradingMode = breakerBox.getRateFeedTradingMode(cUSD_bridgedUSDC_referenceRateFeedID);
    IBreakerBox.BreakerStatus memory valueBreakerStatus = breakerBox.rateFeedBreakerStatus(
      cUSD_bridgedUSDC_referenceRateFeedID,
      address(valueDeltaBreaker)
    );
    IBreakerBox.BreakerStatus memory medianBreakerStatus = breakerBox.rateFeedBreakerStatus(
      cUSD_bridgedUSDC_referenceRateFeedID,
      address(medianDeltaBreaker)
    );
    assertEq(uint256(rateFeedTradingMode), 3); // 3 = trading halted
    assertEq(uint256(valueBreakerStatus.tradingMode), 3); // 3 = trading halted
    assertEq(uint256(medianBreakerStatus.tradingMode), 0); // 0 = bidirectional trading

    setMedianRate(cUSD_bridgedUSDC_referenceRateFeedID, 1e24 + 1e24 * 0.11);

    // Cool down breaker and set new median that doesnt exceed threshold
    vm.warp(block.timestamp + 1 seconds);
    setMedianRate(cUSD_bridgedUSDC_referenceRateFeedID, 1e24);

    // Check trading modes
    rateFeedTradingMode = breakerBox.getRateFeedTradingMode(cUSD_bridgedUSDC_referenceRateFeedID);
    valueBreakerStatus = breakerBox.rateFeedBreakerStatus(
      cUSD_bridgedUSDC_referenceRateFeedID,
      address(valueDeltaBreaker)
    );
    medianBreakerStatus = breakerBox.rateFeedBreakerStatus(
      cUSD_bridgedUSDC_referenceRateFeedID,
      address(medianDeltaBreaker)
    );
    assertEq(uint256(rateFeedTradingMode), 0); // 0 = bidirectional trading
    assertEq(uint256(valueBreakerStatus.tradingMode), 0); // 0 = bidirectional trading
    assertEq(uint256(medianBreakerStatus.tradingMode), 0); // 0 = bidirectional trading

    // Try swap with shouldBreak false -> trading is bidirectional again
    doSwapIn(pair_cUSD_bridgedUSDC_ID, address(cUSDToken), address(usdcToken), false);
  }

  function test_valueDeltaBreaker_whenMedianSubceedsThresholdAndRecovers_shouldHaltAndRecover() public {
    // Trip value delta breaker for cUSD_bridgedUSDC_referenceRateFeedID threshold: 0.1 * 1e24
    setMedianRate(cUSD_bridgedUSDC_referenceRateFeedID, 1e24 - 1e24 * 0.11);

    // Try swap with shouldBreak true
    doSwapIn(pair_cUSD_bridgedUSDC_ID, address(cUSDToken), address(usdcToken), true);

    // Check trading modes & ensure only value delta breaker tripped
    uint8 rateFeedTradingMode = breakerBox.getRateFeedTradingMode(cUSD_bridgedUSDC_referenceRateFeedID);
    IBreakerBox.BreakerStatus memory valueBreakerStatus = breakerBox.rateFeedBreakerStatus(
      cUSD_bridgedUSDC_referenceRateFeedID,
      address(valueDeltaBreaker)
    );
    IBreakerBox.BreakerStatus memory medianBreakerStatus = breakerBox.rateFeedBreakerStatus(
      cUSD_bridgedUSDC_referenceRateFeedID,
      address(medianDeltaBreaker)
    );
    assertEq(rateFeedTradingMode, 3); // 3 = trading halted
    assertEq(valueBreakerStatus.tradingMode, 3); // 3 = trading halted
    assertEq(medianBreakerStatus.tradingMode, 0); // 0 = bidirectional trading

    // Cool down breaker and set new median that doesnt exceed threshold
    vm.warp(block.timestamp + 1 seconds);
    setMedianRate(cUSD_bridgedUSDC_referenceRateFeedID, 1e24);

    // Check trading modes
    rateFeedTradingMode = breakerBox.getRateFeedTradingMode(cUSD_bridgedUSDC_referenceRateFeedID);
    valueBreakerStatus = breakerBox.rateFeedBreakerStatus(
      cUSD_bridgedUSDC_referenceRateFeedID,
      address(valueDeltaBreaker)
    );
    medianBreakerStatus = breakerBox.rateFeedBreakerStatus(
      cUSD_bridgedUSDC_referenceRateFeedID,
      address(medianDeltaBreaker)
    );
    assertEq(uint256(rateFeedTradingMode), 0); // 0 = bidirectional trading
    assertEq(uint256(valueBreakerStatus.tradingMode), 0); // 0 = bidirectional trading
    assertEq(uint256(medianBreakerStatus.tradingMode), 0); // 0 = bidirectional trading

    // Try swap with shouldBreak false -> trading is bidirectional again
    doSwapIn(pair_cUSD_bridgedUSDC_ID, address(cUSDToken), address(usdcToken), false);
  }

  function test_medianDeltaBreaker_whenCooledDownButMedianStillExceeds_shouldNotRecover() public {
    // Trip median delta breaker for cUSD_CELO_referenceRateFeedID threshold: 15%
    setMedianRate(cUSD_CELO_referenceRateFeedID, 5e23 * 1.151);

    // Try swap with shouldBreak true
    doSwapIn(pair_cUSD_CELO_ID, address(cUSDToken), address(celoToken), true);

    // Check trading mode
    uint8 tradingMode = breakerBox.getRateFeedTradingMode(cUSD_CELO_referenceRateFeedID);
    assertEq(uint256(tradingMode), 3); // 3 = trading halted

    // Cool down breaker
    vm.warp(block.timestamp + 5 minutes);
    setMedianRate(cUSD_CELO_referenceRateFeedID, 5e23 * 0.95);

    // Check trading mode
    tradingMode = breakerBox.getRateFeedTradingMode(cUSD_CELO_referenceRateFeedID);
    assertEq(uint256(tradingMode), 3); // 3 = bidirectional trading

    // Try swap with shouldBreak true -> median still exceeds threshold
    doSwapIn(pair_cUSD_CELO_ID, address(cUSDToken), address(celoToken), true);
  }

  function test_valueDeltaBreaker_whenCooledDownButMedianStillExceeds_shouldNotRecover() public {
    // Trip value delta breaker for cUSD_bridgedUSDC_referenceRateFeedID threshold: 0.1 * 1e24
    setMedianRate(cUSD_bridgedUSDC_referenceRateFeedID, 1e24 + 1e24 * 0.11);

    // Try swap with shouldBreak true
    doSwapIn(pair_cUSD_bridgedUSDC_ID, address(cUSDToken), address(usdcToken), true);

    // Check trading modes
    uint8 rateFeedTradingMode = breakerBox.getRateFeedTradingMode(cUSD_bridgedUSDC_referenceRateFeedID);
    IBreakerBox.BreakerStatus memory valueBreakerStatus = breakerBox.rateFeedBreakerStatus(
      cUSD_bridgedUSDC_referenceRateFeedID,
      address(valueDeltaBreaker)
    );
    IBreakerBox.BreakerStatus memory medianBreakerStatus = breakerBox.rateFeedBreakerStatus(
      cUSD_bridgedUSDC_referenceRateFeedID,
      address(medianDeltaBreaker)
    );
    assertEq(rateFeedTradingMode, 3); // 3 = trading halted
    assertEq(valueBreakerStatus.tradingMode, 3); // 3 = trading halted
    assertEq(medianBreakerStatus.tradingMode, 0); // 0 = bidirectional trading

    // Cool down breaker
    vm.warp(block.timestamp + 1 seconds);
    setMedianRate(cUSD_bridgedUSDC_referenceRateFeedID, 1e24 + 1e24 * 0.11);

    // Check trading modes
    rateFeedTradingMode = breakerBox.getRateFeedTradingMode(cUSD_bridgedUSDC_referenceRateFeedID);
    valueBreakerStatus = breakerBox.rateFeedBreakerStatus(
      cUSD_bridgedUSDC_referenceRateFeedID,
      address(valueDeltaBreaker)
    );
    medianBreakerStatus = breakerBox.rateFeedBreakerStatus(
      cUSD_bridgedUSDC_referenceRateFeedID,
      address(medianDeltaBreaker)
    );
    assertEq(rateFeedTradingMode, 3); // 3 = trading halted
    assertEq(valueBreakerStatus.tradingMode, 3); // 3 = trading halted
    assertEq(medianBreakerStatus.tradingMode, 0); // 0 = bidirectional trading

    // Try swap with shouldBreak true -> median still exceeds threshold
    doSwapIn(pair_cUSD_bridgedUSDC_ID, address(cUSDToken), address(usdcToken), true);
  }

  function test_valueDeltaBreaker_whenCooldownTimeIsZero_shouldNeverRecover() public {
    // Trip value delta breaker for cUSD_cEUR_referenceRateFeedID  threshold: 0.05 * 1e24 referenceRate = 11e23
    setMedianRate(cUSD_cEUR_referenceRateFeedID, 11e23 + 11e23 * 0.051);

    // Try swap with shouldBreak true
    doSwapIn(pair_cUSD_cEUR_ID, address(cUSDToken), address(cEURToken), true);

    // Check trading modes
    uint8 rateFeedTradingMode = breakerBox.getRateFeedTradingMode(cUSD_cEUR_referenceRateFeedID);
    assertEq(uint256(rateFeedTradingMode), 3); // 3 = trading halted

    // Cool down breaker
    vm.warp(block.timestamp + 5 seconds);
    setMedianRate(cUSD_cEUR_referenceRateFeedID, 11e23 + 11e23 * 0.01);

    // Try swap with shouldBreak true
    doSwapIn(pair_cUSD_cEUR_ID, address(cEURToken), address(cUSDToken), true);

    // Check trading modes
    rateFeedTradingMode = breakerBox.getRateFeedTradingMode(cUSD_cEUR_referenceRateFeedID);
    assertEq(uint256(rateFeedTradingMode), 3); // 3 = trading halted
  }

  function test_medianDeltaBreaker_whenCooldownTimeIsZero_shouldNeverRecover() public {
    // Trip median delta breaker for cEUR_CELO_referenceRateFeedID  threshold: 0.14 * 1e24
    setMedianRate(cEUR_CELO_referenceRateFeedID, 1e24 + 1e24 * 0.141);

    // Try swap with shouldBreak true
    doSwapIn(pair_cEUR_CELO_ID, address(cEURToken), address(celoToken), true);

    // Check trading modes
    uint8 rateFeedTradingMode = breakerBox.getRateFeedTradingMode(cEUR_CELO_referenceRateFeedID);
    assertEq(uint256(rateFeedTradingMode), 3); // 3 = trading halted

    // Cool down breaker
    vm.warp(block.timestamp + 5 seconds);
    setMedianRate(cEUR_CELO_referenceRateFeedID, 1e24 + 1e24 * 0.1);

    // Try swap with shouldBreak true
    doSwapIn(pair_cEUR_CELO_ID, address(cEURToken), address(celoToken), true);

    // Check trading modes
    rateFeedTradingMode = breakerBox.getRateFeedTradingMode(cEUR_CELO_referenceRateFeedID);
    assertEq(uint256(rateFeedTradingMode), 3); // 3 = trading halted
  }

  function test_dependantRateFeeds_whenDependenciesHalt_shouldHalt() public {
    // Trip value delta Breaker for cUSD_bridgedUSDC_referenceRateFeedID threshold: 0.1 * 1e24
    setMedianRate(cUSD_bridgedUSDC_referenceRateFeedID, 1e24 + 1e24 * 0.11);

    // Check that trading modes
    uint8 tradingModecUSDbridgedUSDC = breakerBox.getRateFeedTradingMode(cUSD_bridgedUSDC_referenceRateFeedID);
    uint8 tradingModecEURBridgedUSDC = breakerBox.getRateFeedTradingMode(cEUR_bridgedUSDC_referenceRateFeedID);
    uint8 independentTradingMode = breakerBox.rateFeedTradingMode(cEUR_bridgedUSDC_referenceRateFeedID);
    assertEq(uint256(tradingModecUSDbridgedUSDC), 3); // 3 = trading halted
    assertEq(uint256(independentTradingMode), 0); // 0 = bidirectional trading
    assertEq(uint256(tradingModecEURBridgedUSDC), 3); // 3 = trading halted

    // Try swap with shouldBreak true
    doSwapIn(pair_cEUR_bridgedUSDC_ID, address(cEURToken), address(usdcToken), true);

    // Cool down breaker and set new median that doesnt exceed threshold
    vm.warp(block.timestamp + 1 seconds);
    setMedianRate(cUSD_bridgedUSDC_referenceRateFeedID, 1e24);

    // Check that trading modes are set correctly
    tradingModecUSDbridgedUSDC = breakerBox.getRateFeedTradingMode(cUSD_bridgedUSDC_referenceRateFeedID);
    tradingModecEURBridgedUSDC = breakerBox.getRateFeedTradingMode(cEUR_bridgedUSDC_referenceRateFeedID);
    independentTradingMode = breakerBox.rateFeedTradingMode(cEUR_bridgedUSDC_referenceRateFeedID);
    assertEq(uint256(tradingModecUSDbridgedUSDC), 0); // 0 = bidirectional trading
    assertEq(uint256(independentTradingMode), 0); // 0 = bidirectional trading
    assertEq(uint256(tradingModecEURBridgedUSDC), 0); // 0 = bidirectional trading

    // Try swap with shouldBreak false -> trading is bidirectional again
    doSwapIn(pair_cEUR_bridgedUSDC_ID, address(cEURToken), address(usdcToken), false);
  }
}
