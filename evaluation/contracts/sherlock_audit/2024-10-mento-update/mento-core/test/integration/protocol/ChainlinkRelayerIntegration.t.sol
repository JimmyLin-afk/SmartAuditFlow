// SPDX-License-Identifier: GPL-3.0-or-later
// solhint-disable func-name-mixedcase, var-name-mixedcase, state-visibility,
// solhint-disable const-name-snakecase, max-states-count, contract-name-camelcase
pragma solidity ^0.8;

import { ProtocolTest } from "./ProtocolTest.sol";
import { MockAggregatorV3 } from "test/utils/mocks/MockAggregatorV3.sol";

import { IOwnable } from "contracts/interfaces/IOwnable.sol";

import "contracts/interfaces/IChainlinkRelayerFactory.sol";
import "contracts/interfaces/ITransparentProxy.sol";
import "contracts/oracles/ChainlinkRelayerFactoryProxyAdmin.sol";
import "contracts/oracles/ChainlinkRelayerFactoryProxy.sol";
import "contracts/oracles/ChainlinkRelayerFactory.sol";
import "contracts/oracles/ChainlinkRelayerV1.sol";
// solhint-disable-next-line max-line-length
import { ITransparentUpgradeableProxy } from "openzeppelin-contracts-next/contracts/proxy/transparent/TransparentUpgradeableProxy.sol";

contract ChainlinkRelayerIntegration is ProtocolTest {
  address owner = makeAddr("owner");

  ChainlinkRelayerFactory relayerFactoryImplementation;
  ChainlinkRelayerFactoryProxyAdmin proxyAdmin;
  ITransparentUpgradeableProxy proxy;
  IChainlinkRelayerFactory relayerFactory;

  function setUp() public virtual override {
    super.setUp();

    proxyAdmin = new ChainlinkRelayerFactoryProxyAdmin();
    relayerFactoryImplementation = new ChainlinkRelayerFactory(true);
    proxy = ITransparentUpgradeableProxy(
      address(
        new ChainlinkRelayerFactoryProxy(
          address(relayerFactoryImplementation),
          address(proxyAdmin),
          abi.encodeWithSignature("initialize(address,address)", address(sortedOracles), owner)
        )
      )
    );
    relayerFactory = IChainlinkRelayerFactory(address(proxy));

    IOwnable(address(proxyAdmin)).transferOwnership(owner);
    IOwnable(address(relayerFactory)).transferOwnership(owner);
  }
}

contract ChainlinkRelayerIntegration_ProxySetup is ChainlinkRelayerIntegration {
  function test_proxyOwnedByAdmin() public view {
    address admin = proxyAdmin.getProxyAdmin(proxy);
    assertEq(admin, address(proxyAdmin));
  }

  function test_adminOwnedByOwner() public view {
    address realOwner = IOwnable(address(proxyAdmin)).owner();
    assertEq(realOwner, owner);
  }

  function test_adminCantCallImplementation() public {
    vm.prank(address(proxyAdmin));
    vm.expectRevert("TransparentUpgradeableProxy: admin cannot fallback to proxy target");
    relayerFactory.sortedOracles();
  }

  function test_nonAdminCantCallProxy() public {
    vm.prank(owner);
    vm.expectRevert();
    proxy.implementation();
  }

  function test_implementationOwnedByOwner() public view {
    address realOwner = IOwnable(address(relayerFactory)).owner();
    assertEq(realOwner, owner);
  }

  function test_implementationSetCorrectly() public view {
    address implementation = proxyAdmin.getProxyImplementation(proxy);
    assertEq(implementation, address(relayerFactoryImplementation));
  }

  function test_implementationNotInitializable() public {
    vm.expectRevert("Initializable: contract is already initialized");
    relayerFactoryImplementation.initialize(address(sortedOracles), address(this));
  }
}

contract ChainlinkRelayerIntegration_ReportAfterRedeploy is ChainlinkRelayerIntegration {
  // Fictional rate feed ID
  address rateFeedId = address(bytes20(keccak256(("cUSD/FOO"))));

  MockAggregatorV3 chainlinkAggregator0;
  MockAggregatorV3 chainlinkAggregator1;

  function setUp() public override {
    super.setUp();

    chainlinkAggregator0 = new MockAggregatorV3(8);
    chainlinkAggregator1 = new MockAggregatorV3(8);
  }

  function test_reportAfterRedeploy() public {
    IChainlinkRelayer.ChainlinkAggregator[] memory aggregatorList0 = new IChainlinkRelayer.ChainlinkAggregator[](1);
    aggregatorList0[0] = IChainlinkRelayer.ChainlinkAggregator(address(chainlinkAggregator0), false);
    IChainlinkRelayer.ChainlinkAggregator[] memory aggregatorList1 = new IChainlinkRelayer.ChainlinkAggregator[](1);
    aggregatorList1[0] = IChainlinkRelayer.ChainlinkAggregator(address(chainlinkAggregator1), false);

    vm.prank(owner);
    IChainlinkRelayer chainlinkRelayer0 = IChainlinkRelayer(
      relayerFactory.deployRelayer(rateFeedId, "cUSD/FOO", 0, aggregatorList0)
    );

    sortedOracles.addOracle(rateFeedId, address(chainlinkRelayer0));

    chainlinkAggregator0.setRoundData(1000000, block.timestamp);
    chainlinkRelayer0.relay();

    vm.warp(block.timestamp + 100);

    vm.prank(owner);
    IChainlinkRelayer chainlinkRelayer1 = IChainlinkRelayer(
      relayerFactory.redeployRelayer(rateFeedId, "cUSD/FOO", 0, aggregatorList1)
    );

    sortedOracles.addOracle(rateFeedId, address(chainlinkRelayer1));

    chainlinkAggregator1.setRoundData(1000000, block.timestamp);
    chainlinkRelayer1.relay();
    assertEq(sortedOracles.numRates(rateFeedId), 2);

    vm.warp(block.timestamp + 1000);

    chainlinkAggregator1.setRoundData(1000000, block.timestamp);
    chainlinkRelayer1.relay();
    assertEq(sortedOracles.numRates(rateFeedId), 1);
  }
}

contract ChainlinkRelayerIntegration_CircuitBreakerInteraction is ChainlinkRelayerIntegration {
  // Fictional rate feed ID
  address rateFeedId = address(bytes20(keccak256(("cUSD/FOO"))));

  MockAggregatorV3 chainlinkAggregator;
  IChainlinkRelayer chainlinkRelayer;

  function setUp() public override {
    super.setUp();

    setUpRelayer();
    setUpCircuitBreaker();
  }

  function setUpRelayer() public {
    chainlinkAggregator = new MockAggregatorV3(8);
    IChainlinkRelayer.ChainlinkAggregator[] memory aggregators = new IChainlinkRelayer.ChainlinkAggregator[](1);
    aggregators[0] = IChainlinkRelayer.ChainlinkAggregator(address(chainlinkAggregator), false);
    vm.prank(owner);
    chainlinkRelayer = IChainlinkRelayer(relayerFactory.deployRelayer(rateFeedId, "CELO/USD", 0, aggregators));

    sortedOracles.addOracle(rateFeedId, address(chainlinkRelayer));
  }

  function setUpCircuitBreaker() public {
    setUpBreakerBox();
    setUpBreaker();
  }

  function setUpBreakerBox() public {
    breakerBox.addRateFeed(rateFeedId);
    breakerBox.toggleBreaker(address(valueDeltaBreaker), rateFeedId, true);
  }

  function setUpBreaker() public {
    address[] memory rateFeeds = new address[](1);
    rateFeeds[0] = rateFeedId;
    uint256[] memory thresholds = new uint256[](1);
    thresholds[0] = 10 ** 23; // 10%
    uint256[] memory cooldownTimes = new uint256[](1);
    cooldownTimes[0] = 1 minutes;
    uint256[] memory referenceValues = new uint256[](1);
    referenceValues[0] = 10 ** 24;

    valueDeltaBreaker.setRateChangeThresholds(rateFeeds, thresholds);
    valueDeltaBreaker.setCooldownTimes(rateFeeds, cooldownTimes);
    valueDeltaBreaker.setReferenceValues(rateFeeds, referenceValues);
  }

  function test_initiallyNoPrice() public view {
    (uint256 price, uint256 denominator) = sortedOracles.medianRate(rateFeedId);
    uint8 tradingMode = breakerBox.getRateFeedTradingMode(rateFeedId);
    assertEq(price, 0);
    assertEq(denominator, 0);
    assertEq(uint256(tradingMode), 0);
  }

  function test_passesPriceFromAggregatorToSortedOracles() public {
    chainlinkAggregator.setRoundData(10 ** 8, block.timestamp - 1);
    chainlinkRelayer.relay();
    (uint256 price, uint256 denominator) = sortedOracles.medianRate(rateFeedId);
    uint8 tradingMode = breakerBox.getRateFeedTradingMode(rateFeedId);
    assertEq(price, 10 ** 24);
    assertEq(denominator, 10 ** 24);
    assertEq(uint256(tradingMode), 0);
  }

  function test_whenPriceBeyondThresholdIsRelayed_breakerShouldTrigger() public {
    chainlinkAggregator.setRoundData(12 * 10 ** 7, block.timestamp - 1);
    chainlinkRelayer.relay();
    (uint256 price, uint256 denominator) = sortedOracles.medianRate(rateFeedId);
    uint8 tradingMode = breakerBox.getRateFeedTradingMode(rateFeedId);
    assertEq(price, 12 * 10 ** 23);
    assertEq(denominator, 10 ** 24);
    assertEq(uint256(tradingMode), 3);
  }

  function test_whenPriceBeyondThresholdIsRelayedThenRecovers_breakerShouldTriggerThenRecover() public {
    vm.warp(100000);
    uint256 t0 = block.timestamp;
    chainlinkAggregator.setRoundData(12 * 10 ** 7, t0 - 1);
    chainlinkRelayer.relay();
    uint8 tradingMode = breakerBox.getRateFeedTradingMode(rateFeedId);
    assertEq(uint256(tradingMode), 3);

    vm.warp(t0 + 1 minutes + 1);

    chainlinkAggregator.setRoundData(105 * 10 ** 6, t0 + 1 minutes + 1);
    chainlinkRelayer.relay();
    (uint256 price, uint256 denominator) = sortedOracles.medianRate(rateFeedId);
    tradingMode = breakerBox.getRateFeedTradingMode(rateFeedId);
    assertEq(price, 105 * 10 ** 22);
    assertEq(denominator, 10 ** 24);
    assertEq(uint256(tradingMode), 0);
  }

  function test_whenPriceBeyondThresholdIsRelayedAndCooldownIsntReached_breakerShouldTriggerAndNotRecover() public {
    vm.warp(100000);
    uint256 t0 = block.timestamp;
    chainlinkAggregator.setRoundData(12 * 10 ** 7, t0 - 1);
    chainlinkRelayer.relay();
    uint8 tradingMode = breakerBox.getRateFeedTradingMode(rateFeedId);
    assertEq(uint256(tradingMode), 3);

    vm.warp(t0 + 1 minutes - 1);

    chainlinkAggregator.setRoundData(105 * 10 ** 6, t0 + 1 minutes - 1);
    chainlinkRelayer.relay();
    (uint256 price, uint256 denominator) = sortedOracles.medianRate(rateFeedId);
    tradingMode = breakerBox.getRateFeedTradingMode(rateFeedId);
    assertEq(price, 105 * 10 ** 22);
    assertEq(denominator, 10 ** 24);
    assertEq(uint256(tradingMode), 3);
  }
}
