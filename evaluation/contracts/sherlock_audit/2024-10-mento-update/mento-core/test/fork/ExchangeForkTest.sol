// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

import { console } from "forge-std/console.sol";
import { FixidityLib } from "celo/contracts/common/FixidityLib.sol";

import { IExchangeProvider } from "contracts/interfaces/IExchangeProvider.sol";
import { IBiPoolManager } from "contracts/interfaces/IBiPoolManager.sol";
import { IERC20 } from "contracts/interfaces/IERC20.sol";

import { BaseForkTest } from "./BaseForkTest.sol";
import { SwapAssertions } from "./assertions/SwapAssertions.sol";
import { CircuitBreakerAssertions } from "./assertions/CircuitBreakerAssertions.sol";
import { TradingLimitHelpers } from "./helpers/TradingLimitHelpers.sol";
import { TokenHelpers } from "./helpers/TokenHelpers.sol";
import { OracleHelpers } from "./helpers/OracleHelpers.sol";
import { SwapHelpers } from "./helpers/SwapHelpers.sol";
import { LogHelpers } from "./helpers/LogHelpers.sol";
import { L0, L1, LG } from "./helpers/misc.sol";

abstract contract ExchangeForkTest is SwapAssertions, CircuitBreakerAssertions, BaseForkTest {
  using FixidityLib for FixidityLib.Fraction;
  using TradingLimitHelpers for *;
  using OracleHelpers for *;
  using SwapHelpers for *;
  using TokenHelpers for *;
  using LogHelpers for *;

  uint256 public exchangeIndex;
  uint256 public exchangeProviderIndex;
  bytes32 public exchangeId;

  address public exchangeProviderAddr;
  IBiPoolManager public exchangeProvider;

  IExchangeProvider.Exchange public exchange;
  IBiPoolManager.PoolExchange public poolExchange;
  address public rateFeedId;

  ExchangeForkTest private ctx = this;

  constructor(uint256 _chainId, uint256 _exchangeProviderIndex, uint256 _exchangeIndex) BaseForkTest(_chainId) {
    exchangeProviderIndex = _exchangeProviderIndex;
    exchangeIndex = _exchangeIndex;
  }

  function setUp() public override {
    super.setUp();
    loadExchange();

    console.log(
      "%s | ExchangeProvider: %s | ExchangeID: %s",
      this.ticker(),
      exchangeProviderAddr,
      vm.toString(exchangeId)
    );
    for (uint256 i = 0; i < COLLATERAL_ASSETS_COUNT; i++) {
      address collateralAsset = mentoReserve.collateralAssets(i);
      vm.label(collateralAsset, IERC20(collateralAsset).symbol());
      mint(collateralAsset, address(mentoReserve), uint256(25_000_000).toSubunits(collateralAsset), true);
    }
  }

  function loadExchange() internal {
    exchangeProviderAddr = broker.exchangeProviders(exchangeProviderIndex);
    exchangeProvider = IBiPoolManager(exchangeProviderAddr);
    exchange = exchangeProvider.getExchanges()[exchangeIndex];
    vm.label(exchange.assets[0], exchange.assets[0].symbol());
    vm.label(exchange.assets[1], exchange.assets[1].symbol());
    exchangeId = exchange.exchangeId;
    poolExchange = exchangeProvider.getPoolExchange(exchange.exchangeId);
    rateFeedId = poolExchange.config.referenceRateFeedID;
  }

  function asset(uint256 index) public view returns (address) {
    return exchange.assets[index];
  }

  function getExchange() public view returns (IExchangeProvider.Exchange memory) {
    return exchange;
  }

  function getPool() public view returns (IBiPoolManager.PoolExchange memory) {
    return poolExchange;
  }

  function test_swapIn_worksInBothDirections() public {
    // asset0 -> asset1
    assert_swapIn(exchange.assets[0], exchange.assets[1]);
    // asset1 -> asset0
    assert_swapIn(exchange.assets[1], exchange.assets[0]);
  }

  function test_swapOut_worksInBothDirections() public {
    // asset0 -> asset1
    assert_swapOut(exchange.assets[0], exchange.assets[1]);
    // asset1 -> asset0
    assert_swapOut(exchange.assets[1], exchange.assets[0]);
  }

  function test_tradingLimitsAreConfigured() public view {
    bytes32 asset0Bytes32 = bytes32(uint256(uint160(exchange.assets[0])));
    bytes32 limitIdForAsset0 = exchange.exchangeId ^ asset0Bytes32;
    bytes32 asset1Bytes32 = bytes32(uint256(uint160(exchange.assets[1])));
    bytes32 limitIdForAsset1 = exchange.exchangeId ^ asset1Bytes32;

    bool asset0LimitConfigured = ctx.isLimitConfigured(limitIdForAsset0);
    bool asset1LimitConfigured = ctx.isLimitConfigured(limitIdForAsset1);

    require(asset0LimitConfigured || asset1LimitConfigured, "Limit not configured");
  }

  function test_tradingLimitsAreEnforced_0to1_L0() public {
    ctx.logHeader();
    assert_swapOverLimitFails(exchange.assets[0], exchange.assets[1], L0);
  }

  function test_tradingLimitsAreEnforced_0to1_L1() public {
    ctx.logHeader();
    assert_swapOverLimitFails(exchange.assets[0], exchange.assets[1], L1);
  }

  function test_tradingLimitsAreEnforced_0to1_LG() public {
    ctx.logHeader();
    assert_swapOverLimitFails(exchange.assets[0], exchange.assets[1], LG);
  }

  function test_tradingLimitsAreEnforced_1to0_L0() public {
    ctx.logHeader();
    assert_swapOverLimitFails(exchange.assets[1], exchange.assets[0], L0);
  }

  function test_tradingLimitsAreEnforced_1to0_L1() public {
    ctx.logHeader();
    assert_swapOverLimitFails(exchange.assets[1], exchange.assets[0], L1);
  }

  function test_tradingLimitsAreEnforced_1to0_LG() public {
    ctx.logHeader();
    assert_swapOverLimitFails(exchange.assets[1], exchange.assets[0], LG);
  }

  function test_circuitBreaker_rateFeedIsProtected() public view {
    address[] memory breakers = breakerBox.getBreakers();
    ctx.logHeader();
    bool found = false;
    for (uint256 j = 0; j < breakers.length && !found; j++) {
      found = breakerBox.isBreakerEnabled(breakers[j], rateFeedId);
    }
    require(found, "No breaker found for rateFeedId");
  }

  function test_circuitBreaker_breaks() public {
    address[] memory breakers = breakerBox.getBreakers();
    ctx.logHeader();
    for (uint256 j = 0; j < breakers.length; j++) {
      if (breakerBox.isBreakerEnabled(breakers[j], rateFeedId)) {
        assert_breakerBreaks(rateFeedId, breakers[j], j);
      }
    }
  }

  function test_circuitBreaker_recovers() public {
    address[] memory breakers = breakerBox.getBreakers();
    ctx.logHeader();
    for (uint256 j = 0; j < breakers.length; j++) {
      if (breakerBox.isBreakerEnabled(breakers[j], rateFeedId)) {
        assert_breakerRecovers(rateFeedId, breakers[j], j);
      }
    }
  }

  function test_circuitBreaker_haltsTrading() public {
    address[] memory breakers = breakerBox.getBreakers();
    ctx.logHeader();
    for (uint256 j = 0; j < breakers.length; j++) {
      if (breakerBox.isBreakerEnabled(breakers[j], rateFeedId)) {
        assert_breakerBreaks(rateFeedId, breakers[j], j);

        assert_swapInFails(
          exchange.assets[0],
          exchange.assets[1],
          uint256(10000).toSubunits(exchange.assets[0]),
          "Trading is suspended for this reference rate"
        );
        assert_swapInFails(
          exchange.assets[1],
          exchange.assets[0],
          uint256(10000).toSubunits(exchange.assets[1]),
          "Trading is suspended for this reference rate"
        );

        assert_swapOutFails(
          exchange.assets[0],
          exchange.assets[1],
          uint256(1000).toSubunits(exchange.assets[1]),
          "Trading is suspended for this reference rate"
        );
        assert_swapOutFails(
          exchange.assets[1],
          exchange.assets[0],
          uint256(1000).toSubunits(exchange.assets[0]),
          "Trading is suspended for this reference rate"
        );
      }
    }
  }

  function test_rateFeedDependencies_haltsDependantTrading() public {
    uint256 depsCount = rateFeedDependenciesCount[rateFeedId];

    /// @dev If this doesn't revert thare are more dependencies than expected.
    /// In which case the mapping in BaseForkTest should be updated.
    vm.expectRevert();
    breakerBox.rateFeedDependencies(rateFeedId, depsCount);

    if (depsCount == 0) {
      return;
    }

    address[] memory breakers = breakerBox.getBreakers();
    address[] memory dependencies = new address[](depsCount);
    for (uint256 i = 0; i < depsCount; i++) {
      dependencies[i] = breakerBox.rateFeedDependencies(rateFeedId, i);
    }

    ctx.logPool();
    console.log(
      "\t exchangeIndex: %d | rateFeedId: %s | %s dependencies",
      exchangeIndex,
      rateFeedId,
      dependencies.length
    );

    for (uint256 i = 0; i < dependencies.length; i++) {
      console.log("===========================================================");
      console.log("Dependency: %s", dependencies[i]);
      console.log("===========================================================");
      for (uint256 j = 0; j < breakers.length; j++) {
        if (breakerBox.isBreakerEnabled(breakers[j], dependencies[i])) {
          assert_breakerBreaks(dependencies[i], breakers[j], j);

          assert_swapInFails(
            exchange.assets[0],
            exchange.assets[1],
            uint256(1000).toSubunits(exchange.assets[0]),
            "Trading is suspended for this reference rate"
          );

          assert_breakerRecovers(dependencies[i], breakers[j], j);
        }
      }
      console.log("\n");
    }
  }
}
