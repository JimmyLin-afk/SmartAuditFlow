// SPDX-License-Identifier: GPL-3.0-or-later
// solhint-disable func-name-mixedcase, var-name-mixedcase, state-visibility
// solhint-disable const-name-snakecase, max-states-count, contract-name-camelcase
pragma solidity ^0.8.18;

import { console } from "forge-std/console.sol";
import { Test } from "mento-std/Test.sol";

import "test/utils/mocks/MockAggregatorV3.sol";
import "contracts/interfaces/IChainlinkRelayer.sol";
import "contracts/oracles/ChainlinkRelayerV1.sol";

import { UD60x18, ud, intoUint256 } from "prb/math/UD60x18.sol";

interface ISortedOracles {
  function initialize(uint256) external;

  function addOracle(address, address) external;

  function removeOracle(address, address, uint256) external;

  function report(address, uint256, address, address) external;

  function setTokenReportExpiry(address, uint256) external;

  function medianRate(address) external returns (uint256, uint256);

  function medianTimestamp(address token) external returns (uint256);

  function getRates(address rateFeedId) external returns (address[] memory, uint256[] memory, uint256[] memory);
}

contract ChainlinkRelayerV1Test is Test {
  bytes constant EXPIRED_TIMESTAMP_ERROR = abi.encodeWithSignature("ExpiredTimestamp()");
  bytes constant INVALID_PRICE_ERROR = abi.encodeWithSignature("InvalidPrice()");
  bytes constant INVALID_AGGREGATOR_ERROR = abi.encodeWithSignature("InvalidAggregator()");
  bytes constant INVALID_MAX_TIMESTAMP_SPREAD_ERROR = abi.encodeWithSignature("InvalidMaxTimestampSpread()");
  bytes constant NO_AGGREGATORS_ERROR = abi.encodeWithSignature("NoAggregators()");
  bytes constant TIMESTAMP_NOT_NEW_ERROR = abi.encodeWithSignature("TimestampNotNew()");
  bytes constant TIMESTAMP_SPREAD_TOO_HIGH_ERROR = abi.encodeWithSignature("TimestampSpreadTooHigh()");
  bytes constant TOO_MANY_AGGREGATORS_ERROR = abi.encodeWithSignature("TooManyAggregators()");
  bytes constant TOO_MANY_EXISTING_REPORTS_ERROR = abi.encodeWithSignature("TooManyExistingReports()");

  ISortedOracles sortedOracles;

  IChainlinkRelayer.ChainlinkAggregator aggregator0;
  IChainlinkRelayer.ChainlinkAggregator aggregator1;
  IChainlinkRelayer.ChainlinkAggregator aggregator2;
  IChainlinkRelayer.ChainlinkAggregator aggregator3;

  MockAggregatorV3 mockAggregator0;
  MockAggregatorV3 mockAggregator1;
  MockAggregatorV3 mockAggregator2;
  MockAggregatorV3 mockAggregator3;

  bool invert0 = false;
  bool invert1 = true;
  bool invert2 = false;
  bool invert3 = true;

  int256 aggregatorPrice0;
  int256 aggregatorPrice1;
  int256 aggregatorPrice2;
  int256 aggregatorPrice3;

  uint256 expectedReport;
  uint256 aReport = 4100000000000000000000000;

  IChainlinkRelayer relayer;
  address rateFeedId = makeAddr("rateFeed");
  uint256 expirySeconds = 600;

  function setUpRelayer(uint256 aggregatorsCount) internal {
    aggregator0 = IChainlinkRelayer.ChainlinkAggregator(address(mockAggregator0), invert0);
    aggregator1 = IChainlinkRelayer.ChainlinkAggregator(address(mockAggregator1), invert1);
    aggregator2 = IChainlinkRelayer.ChainlinkAggregator(address(mockAggregator2), invert2);
    aggregator3 = IChainlinkRelayer.ChainlinkAggregator(address(mockAggregator3), invert3);

    IChainlinkRelayer.ChainlinkAggregator[] memory aggregators = new IChainlinkRelayer.ChainlinkAggregator[](
      aggregatorsCount
    );
    aggregators[0] = aggregator0;
    if (aggregatorsCount > 1) {
      aggregators[1] = aggregator1;
      if (aggregatorsCount > 2) {
        aggregators[2] = aggregator2;
        if (aggregatorsCount > 3) {
          aggregators[3] = aggregator3;
        }
      }
    }

    uint256 maxTimestampSpread = aggregatorsCount > 1 ? 300 : 0;
    relayer = IChainlinkRelayer(
      new ChainlinkRelayerV1(rateFeedId, "CELO/USD", address(sortedOracles), maxTimestampSpread, aggregators)
    );
    sortedOracles.addOracle(rateFeedId, address(relayer));
  }

  function setUp() public virtual {
    sortedOracles = ISortedOracles(deployCode("SortedOracles", abi.encode(true)));
    sortedOracles.initialize(expirySeconds);
    sortedOracles.setTokenReportExpiry(rateFeedId, expirySeconds);

    mockAggregator0 = new MockAggregatorV3(8);
    mockAggregator1 = new MockAggregatorV3(12);
    mockAggregator2 = new MockAggregatorV3(8);
    mockAggregator3 = new MockAggregatorV3(6);
  }

  function setAggregatorPrices() public {
    mockAggregator0.setRoundData(aggregatorPrice0, uint256(block.timestamp));
    mockAggregator1.setRoundData(aggregatorPrice1, uint256(block.timestamp));
    mockAggregator2.setRoundData(aggregatorPrice2, uint256(block.timestamp));
    mockAggregator3.setRoundData(aggregatorPrice3, uint256(block.timestamp));
  }
}

contract ChainlinkRelayerV1Test_constructor_invalid is ChainlinkRelayerV1Test {
  function test_constructorRevertsWhenAggregatorsIsEmpty() public {
    vm.expectRevert(NO_AGGREGATORS_ERROR);
    new ChainlinkRelayerV1(
      rateFeedId,
      "CELO/USD",
      address(sortedOracles),
      0,
      new IChainlinkRelayer.ChainlinkAggregator[](0)
    );
  }

  function test_constructorRevertsWhenTooManyAggregators() public {
    vm.expectRevert(TOO_MANY_AGGREGATORS_ERROR);
    new ChainlinkRelayerV1(
      rateFeedId,
      "CELO/USD",
      address(sortedOracles),
      300,
      new IChainlinkRelayer.ChainlinkAggregator[](5)
    );
  }

  function test_constructorRevertsWhenAggregatorsIsInvalid() public {
    vm.expectRevert(INVALID_AGGREGATOR_ERROR);
    new ChainlinkRelayerV1(
      rateFeedId,
      "CELO/USD",
      address(sortedOracles),
      0,
      new IChainlinkRelayer.ChainlinkAggregator[](1)
    );
  }

  function test_constructorRevertsWhenNoTimestampSpreadButMultipleAggregators() public {
    vm.expectRevert(INVALID_MAX_TIMESTAMP_SPREAD_ERROR);
    new ChainlinkRelayerV1(
      rateFeedId,
      "CELO/USD",
      address(sortedOracles),
      0,
      new IChainlinkRelayer.ChainlinkAggregator[](2)
    );
  }

  function test_constructorRevertsWhenTimestampSpreadPositiveButSingleAggregator() public {
    vm.expectRevert(INVALID_MAX_TIMESTAMP_SPREAD_ERROR);
    new ChainlinkRelayerV1(
      rateFeedId,
      "CELO/USD",
      address(sortedOracles),
      300,
      new IChainlinkRelayer.ChainlinkAggregator[](1)
    );
  }
}

contract ChainlinkRelayerV1Test_constructor_single is ChainlinkRelayerV1Test {
  function setUpRelayer() internal virtual {
    setUpRelayer(1);
  }

  function setUp() public override {
    super.setUp();
    setUpRelayer();
  }

  function test_constructorSetsRateFeedId() public {
    address _rateFeedId = relayer.rateFeedId();
    assertEq(_rateFeedId, rateFeedId);
  }

  function test_constructorSetsSortedOracles() public {
    address _sortedOracles = relayer.sortedOracles();
    assertEq(_sortedOracles, address(sortedOracles));
  }

  function test_constructorSetsRateFeedDescription() public {
    string memory rateFeedDescription = relayer.rateFeedDescription();
    assertEq(rateFeedDescription, "CELO/USD");
  }

  function test_constructorSetsAggregators() public virtual {
    IChainlinkRelayer.ChainlinkAggregator[] memory aggregators = relayer.getAggregators();
    assertEq(aggregators.length, 1);
    assertEq(aggregators[0].aggregator, address(aggregator0.aggregator));
    assertEq(aggregators[0].invert, aggregator0.invert);
  }
}

contract ChainlinkRelayerV1Test_constructor_double is ChainlinkRelayerV1Test_constructor_single {
  function setUpRelayer() internal override {
    setUpRelayer(2);
  }

  function test_constructorSetsAggregators() public override {
    IChainlinkRelayer.ChainlinkAggregator[] memory aggregators = relayer.getAggregators();
    assertEq(aggregators.length, 2);
    assertEq(aggregators[0].aggregator, address(aggregator0.aggregator));
    assertEq(aggregators[1].aggregator, address(aggregator1.aggregator));
    assertEq(aggregators[0].invert, aggregator0.invert);
    assertEq(aggregators[1].invert, aggregator1.invert);
  }
}

contract ChainlinkRelayerV1Test_constructor_triple is ChainlinkRelayerV1Test_constructor_single {
  function setUpRelayer() internal override {
    setUpRelayer(3);
  }

  function test_constructorSetsAggregators() public override {
    IChainlinkRelayer.ChainlinkAggregator[] memory aggregators = relayer.getAggregators();
    assertEq(aggregators.length, 3);
    assertEq(aggregators[0].aggregator, address(aggregator0.aggregator));
    assertEq(aggregators[1].aggregator, address(aggregator1.aggregator));
    assertEq(aggregators[2].aggregator, address(aggregator2.aggregator));
    assertEq(aggregators[0].invert, aggregator0.invert);
    assertEq(aggregators[1].invert, aggregator1.invert);
    assertEq(aggregators[2].invert, aggregator2.invert);
  }
}

contract ChainlinkRelayerV1Test_constructor_full is ChainlinkRelayerV1Test_constructor_single {
  function setUpRelayer() internal override {
    setUpRelayer(4);
  }

  function test_constructorSetsAggregators() public override {
    IChainlinkRelayer.ChainlinkAggregator[] memory aggregators = relayer.getAggregators();
    assertEq(aggregators.length, 4);
    assertEq(aggregators[0].aggregator, address(aggregator0.aggregator));
    assertEq(aggregators[1].aggregator, address(aggregator1.aggregator));
    assertEq(aggregators[2].aggregator, address(aggregator2.aggregator));
    assertEq(aggregators[3].aggregator, address(aggregator3.aggregator));
    assertEq(aggregators[0].invert, aggregator0.invert);
    assertEq(aggregators[1].invert, aggregator1.invert);
    assertEq(aggregators[2].invert, aggregator2.invert);
    assertEq(aggregators[3].invert, aggregator3.invert);
  }
}

contract ChainlinkRelayerV1Test_fuzz_single is ChainlinkRelayerV1Test {
  function setUp() public override {
    super.setUp();
    setUpRelayer(1);
  }

  function testFuzz_convertsChainlinkToUD60x18Correctly(int256 _rate) public {
    int256 rate = bound(_rate, 1, type(int256).max / 10 ** 18);
    mockAggregator0.setRoundData(rate, uint256(block.timestamp));

    relayer.relay();
    (uint256 medianRate, ) = sortedOracles.medianRate(rateFeedId);
    assertEq(medianRate, uint256(rate) * 10 ** (24 - mockAggregator0.decimals()));
  }
}

contract ChainlinkRelayerV1Test_fuzz_full is ChainlinkRelayerV1Test {
  function setUp() public override {
    super.setUp();
    setUpRelayer(4);
  }

  function testFuzz_calculatesReportsCorrectly(
    int256 _aggregatorPrice0,
    int256 _aggregatorPrice1,
    int256 _aggregatorPrice2,
    int256 _aggregatorPrice3,
    bool _invert0,
    bool _invert1,
    bool _invert2,
    bool _invert3
  ) public {
    aggregatorPrice0 = bound(_aggregatorPrice0, 1, 1e5 * 1e6);
    aggregatorPrice1 = bound(_aggregatorPrice1, 1, 1e5 * 1e6);
    aggregatorPrice2 = bound(_aggregatorPrice2, 1, 1e5 * 1e6);
    aggregatorPrice3 = bound(_aggregatorPrice3, 1, 1e5 * 1e6);

    invert0 = _invert0;
    invert1 = _invert1;
    invert2 = _invert2;
    invert3 = _invert3;

    setUpRelayer(4);
    setAggregatorPrices();

    relayer.relay();
    (uint256 medianRate, ) = sortedOracles.medianRate(rateFeedId);
    assertEq(medianRate, expectedPriceInFixidity());
  }

  function expectedPriceInFixidity() internal view returns (uint256) {
    UD60x18 price0 = ud(uint256(aggregatorPrice0) * 1e10);
    if (invert0) {
      price0 = ud(1e18).div(price0);
    }
    UD60x18 price1 = ud(uint256(aggregatorPrice1) * 1e6);
    if (invert1) {
      price1 = ud(1e18).div(price1);
    }
    UD60x18 price2 = ud(uint256(aggregatorPrice2) * 1e10);
    if (invert2) {
      price2 = ud(1e18).div(price2);
    }
    UD60x18 price3 = ud(uint256(aggregatorPrice3) * 1e12);
    if (invert3) {
      price3 = ud(1e18).div(price3);
    }
    return intoUint256(price0.mul(price1).mul(price2).mul(price3)) * 1e6; // 1e18 -> 1e24
  }
}

contract ChainlinkRelayerV1Test_relay_single is ChainlinkRelayerV1Test {
  modifier withReport(uint256 report) {
    vm.warp(2);
    vm.prank(address(relayer));
    sortedOracles.report(rateFeedId, report, address(0), address(0));
    _;
  }

  function setUpRelayer() internal virtual {
    setUpRelayer(1);
  }

  function setUpExpectations() internal virtual {
    aggregatorPrice0 = 420000000;
    expectedReport = 4200000000000000000000000;
  }

  function setUp() public override {
    super.setUp();
    setUpRelayer();
    setUpExpectations();
    setAggregatorPrices();
  }

  function relayAndLogGas(string memory label) internal {
    uint256 gasBefore = gasleft();
    relayer.relay();
    uint256 gasCost = gasBefore - gasleft();
    console.log("RelayerV1[%s] cost = %d", label, gasCost);
  }

  function test_relaysTheRate() public {
    relayAndLogGas("happy path");
    // relayer.relay();
    (uint256 medianRate, ) = sortedOracles.medianRate(rateFeedId);
    assertEq(medianRate, expectedReport);
  }

  function test_relaysTheRate_withLesserGreater_whenLesser_andCantExpireDirectly() public {
    uint256 t0 = block.timestamp;
    address oldRelayer = makeAddr("oldRelayer");
    sortedOracles.addOracle(rateFeedId, oldRelayer);

    vm.prank(oldRelayer);
    sortedOracles.report(rateFeedId, expectedReport + 2, address(0), address(0));

    vm.warp(t0 + 200); // Not enough to be able to expire the first report
    setAggregatorPrices(); // Update timestamps
    relayAndLogGas("other report no expiry");
    // relayer.relay();

    (address[] memory oracles, uint256[] memory rates, ) = sortedOracles.getRates(rateFeedId);
    assertEq(oracles.length, 2);
    assertEq(oracles[1], address(relayer));
    assertEq(rates[1], expectedReport);

    vm.warp(t0 + 600); // First report should be expired
    setAggregatorPrices(); // Update timestamps
    relayAndLogGas("2 reports with expiry");
    // relayer.relay();

    // Next report should also cause an expiry, resulting in only one report remaining
    // from this oracle.
    (oracles, rates, ) = sortedOracles.getRates(rateFeedId);
    assertEq(oracles.length, 1);
    assertEq(oracles[0], address(relayer));
    assertEq(rates[0], expectedReport);
  }

  function test_relaysTheRate_withLesserGreater_whenGreater_andCantExpireDirectly() public {
    uint256 t0 = block.timestamp;
    address oldRelayer = makeAddr("oldRelayer");
    sortedOracles.addOracle(rateFeedId, oldRelayer);

    vm.prank(oldRelayer);
    sortedOracles.report(rateFeedId, expectedReport - 2, address(0), address(0));

    vm.warp(t0 + 200); // Not enough to be able to expire the first report
    setAggregatorPrices(); // Update timestamps
    relayAndLogGas("other report no expiry");
    // relayer.relay();

    (address[] memory oracles, uint256[] memory rates, ) = sortedOracles.getRates(rateFeedId);
    assertEq(oracles.length, 2);
    assertEq(oracles[0], address(relayer));
    assertEq(rates[0], expectedReport);

    vm.warp(t0 + 600); // First report should be expired
    setAggregatorPrices(); // Update timestamps
    relayAndLogGas("2 reports with expiry");
    // relayer.relay();

    // Next report should also cause an expiry, resulting in only one report remaining
    // from this oracle.
    (oracles, rates, ) = sortedOracles.getRates(rateFeedId);
    assertEq(oracles.length, 1);
    assertEq(oracles[0], address(relayer));
    assertEq(rates[0], expectedReport);
  }

  function test_relaysTheRate_withLesserGreater_whenLesser_andCanExpire() public {
    uint256 t0 = block.timestamp;
    address oldRelayer = makeAddr("oldRelayer");
    sortedOracles.addOracle(rateFeedId, oldRelayer);

    vm.prank(oldRelayer);
    sortedOracles.report(rateFeedId, expectedReport + 2, address(0), address(0));

    vm.warp(t0 + 600); // Not enough to be able to expire the first report
    setAggregatorPrices(); // Update timestamps
    relayAndLogGas("other report with expiry");

    (address[] memory oracles, uint256[] memory rates, ) = sortedOracles.getRates(rateFeedId);
    assertEq(oracles.length, 1);
    assertEq(oracles[0], address(relayer));
    assertEq(rates[0], expectedReport);
  }

  function test_relaysTheRate_withLesserGreater_whenGreater_andCanExpire() public {
    address oldRelayer = makeAddr("oldRelayer");
    sortedOracles.addOracle(rateFeedId, oldRelayer);

    vm.prank(oldRelayer);
    sortedOracles.report(rateFeedId, expectedReport - 2, address(0), address(0));

    vm.warp(block.timestamp + 600); // Not enough to be able to expire the first report
    setAggregatorPrices(); // Update timestamps
    relayAndLogGas("other report with expiry");
    // relayer.relay();

    (address[] memory oracles, uint256[] memory rates, ) = sortedOracles.getRates(rateFeedId);
    assertEq(oracles.length, 1);
    assertEq(oracles[0], address(relayer));
    assertEq(rates[0], expectedReport);
  }

  function test_revertsWhenComputingLesserGreaterWithTooManyReporters() public {
    address oracle0 = makeAddr("oracle0");
    address oracle1 = makeAddr("oracle1");
    sortedOracles.addOracle(rateFeedId, oracle0);
    sortedOracles.addOracle(rateFeedId, oracle1);

    vm.prank(oracle0);
    sortedOracles.report(rateFeedId, expectedReport - 2, address(0), address(0));
    vm.prank(oracle1);
    sortedOracles.report(rateFeedId, expectedReport, oracle0, address(0));

    vm.warp(block.timestamp + 100); // Not enough to be able to expire the first report
    setAggregatorPrices(); // Update timestamps
    vm.expectRevert(TOO_MANY_EXISTING_REPORTS_ERROR);
    relayer.relay();
  }

  function test_revertsOnNegativePrice0() public {
    mockAggregator0.setRoundData(-1 * aggregatorPrice0, block.timestamp);
    vm.expectRevert(INVALID_PRICE_ERROR);
    relayer.relay();
  }

  function test_revertsOnZeroPrice0() public {
    mockAggregator0.setRoundData(0, block.timestamp);
    vm.expectRevert(INVALID_PRICE_ERROR);
    relayer.relay();
  }

  function test_revertsOnEarlierTimestamp() public virtual withReport(aReport) {
    uint256 latestTimestamp = sortedOracles.medianTimestamp(rateFeedId);
    mockAggregator0.setRoundData(aggregatorPrice0, latestTimestamp - 1);
    vm.expectRevert(TIMESTAMP_NOT_NEW_ERROR);
    relayer.relay();
  }

  function test_relaysOnRecentTimestamp0() public withReport(aReport) {
    vm.expectRevert(TIMESTAMP_NOT_NEW_ERROR);
    relayer.relay();
    skip(50);
    mockAggregator0.setRoundData(aggregatorPrice0, block.timestamp);
    relayer.relay();
    uint256 timestamp = sortedOracles.medianTimestamp(rateFeedId);
    assertEq(timestamp, block.timestamp);
  }

  function test_revertsOnRepeatTimestamp0() public withReport(aReport) {
    uint256 latestTimestamp = sortedOracles.medianTimestamp(rateFeedId);
    mockAggregator0.setRoundData(aggregatorPrice0, latestTimestamp);
    vm.expectRevert(TIMESTAMP_NOT_NEW_ERROR);
    relayer.relay();
  }

  function test_revertsWhenMostRecentTimestampIsExpired() public virtual withReport(aReport) {
    mockAggregator0.setRoundData(aggregatorPrice0, block.timestamp + 1);
    vm.warp(block.timestamp + expirySeconds + 1);
    vm.expectRevert(EXPIRED_TIMESTAMP_ERROR);
    relayer.relay();
  }
}

contract ChainlinkRelayerV1Test_relay_double is ChainlinkRelayerV1Test_relay_single {
  function setUpExpectations() internal virtual override {
    aggregatorPrice0 = 420000000; // 4.2 * 1e8
    aggregatorPrice1 = 2000000000000; // 2 * 1e12
    // ^ results in 4.2 * 1 / 2 = 2.1
    expectedReport = 2100000000000000000000000;
  }

  function setUpRelayer() internal virtual override {
    setUpRelayer(2);
  }

  function test_revertsOnNegativePrice1() public {
    mockAggregator1.setRoundData(-1 * aggregatorPrice1, block.timestamp);
    vm.expectRevert(INVALID_PRICE_ERROR);
    relayer.relay();
  }

  function test_revertsOnZeroPrice1() public {
    mockAggregator1.setRoundData(0, block.timestamp);
    vm.expectRevert(INVALID_PRICE_ERROR);
    relayer.relay();
  }

  function test_revertsOnEarlierTimestamp() public virtual override withReport(aReport) {
    uint256 latestTimestamp = sortedOracles.medianTimestamp(rateFeedId);

    mockAggregator0.setRoundData(aggregatorPrice0, latestTimestamp - 1);
    mockAggregator1.setRoundData(aggregatorPrice1, latestTimestamp - 1);

    vm.expectRevert(TIMESTAMP_NOT_NEW_ERROR);
    relayer.relay();
  }

  function test_relaysOnRecentTimestamp1() public withReport(aReport) {
    vm.expectRevert(TIMESTAMP_NOT_NEW_ERROR);
    relayer.relay();
    skip(50);
    mockAggregator1.setRoundData(aggregatorPrice1, block.timestamp);
    relayer.relay();
    uint256 timestamp = sortedOracles.medianTimestamp(rateFeedId);
    assertEq(timestamp, block.timestamp);
  }

  function test_revertsOnRepeatTimestamp1() public withReport(aReport) {
    uint256 latestTimestamp = sortedOracles.medianTimestamp(rateFeedId);
    mockAggregator1.setRoundData(aggregatorPrice1, latestTimestamp);
    vm.expectRevert(TIMESTAMP_NOT_NEW_ERROR);
    relayer.relay();
  }

  function test_revertsWhenMostRecentTimestampIsExpired() public virtual override withReport(aReport) {
    mockAggregator0.setRoundData(aggregatorPrice0, block.timestamp + 1);
    mockAggregator1.setRoundData(aggregatorPrice1, block.timestamp + 1);
    vm.warp(block.timestamp + expirySeconds + 1);
    vm.expectRevert(EXPIRED_TIMESTAMP_ERROR);
    relayer.relay();
  }

  function test_revertsWhenTimestampSpreadTooLarge() public virtual {
    mockAggregator0.setRoundData(aggregatorPrice0, block.timestamp);
    vm.warp(block.timestamp + 301);
    mockAggregator1.setRoundData(aggregatorPrice1, block.timestamp);
    vm.expectRevert(TIMESTAMP_SPREAD_TOO_HIGH_ERROR);
    relayer.relay();
  }
}

contract ChainlinkRelayerV1Test_relay_triple is ChainlinkRelayerV1Test_relay_double {
  function setUpExpectations() internal virtual override {
    aggregatorPrice0 = 420000000; // 4.2 * 1e8
    aggregatorPrice1 = 2000000000000; // 2 * 1e12
    aggregatorPrice2 = 300000000; // 3 * 1e8
    // ^ results in 4.2 * (1 / 2) * 3 = 6.3
    expectedReport = 6300000000000000000000000;
  }

  function setUpRelayer() internal virtual override {
    setUpRelayer(3);
  }

  function test_revertsOnNegativePrice2() public {
    mockAggregator2.setRoundData(-1 * aggregatorPrice2, block.timestamp);
    vm.expectRevert(INVALID_PRICE_ERROR);
    relayer.relay();
  }

  function test_revertsOnZeroPrice2() public {
    mockAggregator2.setRoundData(0, block.timestamp);
    vm.expectRevert(INVALID_PRICE_ERROR);
    relayer.relay();
  }

  function test_revertsOnEarlierTimestamp() public virtual override withReport(aReport) {
    uint256 latestTimestamp = sortedOracles.medianTimestamp(rateFeedId);

    mockAggregator0.setRoundData(aggregatorPrice0, latestTimestamp - 1);
    mockAggregator1.setRoundData(aggregatorPrice1, latestTimestamp - 1);
    mockAggregator2.setRoundData(aggregatorPrice2, latestTimestamp - 1);

    vm.expectRevert(TIMESTAMP_NOT_NEW_ERROR);
    relayer.relay();
  }

  function test_relaysOnRecentTimestamp2() public withReport(aReport) {
    vm.expectRevert(TIMESTAMP_NOT_NEW_ERROR);
    relayer.relay();
    skip(50);
    mockAggregator2.setRoundData(aggregatorPrice0, block.timestamp);
    relayer.relay();
    uint256 timestamp = sortedOracles.medianTimestamp(rateFeedId);
    assertEq(timestamp, block.timestamp);
  }

  function test_revertsOnRepeatTimestamp2() public withReport(aReport) {
    uint256 latestTimestamp = sortedOracles.medianTimestamp(rateFeedId);
    mockAggregator2.setRoundData(aggregatorPrice2, latestTimestamp);
    vm.expectRevert(TIMESTAMP_NOT_NEW_ERROR);
    relayer.relay();
  }

  function test_revertsWhenMostRecentTimestampIsExpired() public virtual override withReport(aReport) {
    mockAggregator0.setRoundData(aggregatorPrice0, block.timestamp + 1);
    mockAggregator1.setRoundData(aggregatorPrice1, block.timestamp + 1);
    mockAggregator2.setRoundData(aggregatorPrice2, block.timestamp + 1);
    vm.warp(block.timestamp + expirySeconds + 1);
    vm.expectRevert(EXPIRED_TIMESTAMP_ERROR);
    relayer.relay();
  }

  function test_revertsWhenTimestampSpreadTooLarge() public virtual override {
    mockAggregator0.setRoundData(aggregatorPrice0, block.timestamp);
    vm.warp(block.timestamp + 301);
    mockAggregator2.setRoundData(aggregatorPrice2, block.timestamp);
    vm.expectRevert(TIMESTAMP_SPREAD_TOO_HIGH_ERROR);
    relayer.relay();
  }
}

contract ChainlinkRelayerV1Test_relay_full is ChainlinkRelayerV1Test_relay_triple {
  function setUpExpectations() internal override {
    aggregatorPrice0 = 420000000; // 4.2 * 1e8
    aggregatorPrice1 = 2000000000000; // 2 * 1e12
    aggregatorPrice2 = 300000000; // 3 * 1e8
    aggregatorPrice3 = 5000000; // 5 * 1e6
    // ^ results in 4.2 * (1 / 2) * 3 * (1/5) = 1.26
    // in the tests price1 and price3 are inverted
    expectedReport = 1260000000000000000000000;
  }

  function setUpRelayer() internal override {
    setUpRelayer(4);
  }

  function test_revertsOnNegativePrice3() public {
    mockAggregator3.setRoundData(-1 * aggregatorPrice3, block.timestamp);
    vm.expectRevert(INVALID_PRICE_ERROR);
    relayer.relay();
  }

  function test_revertsOnZeroPrice3() public {
    mockAggregator3.setRoundData(0, block.timestamp);
    vm.expectRevert(INVALID_PRICE_ERROR);
    relayer.relay();
  }

  function test_revertsOnEarlierTimestamp() public override withReport(aReport) {
    uint256 latestTimestamp = sortedOracles.medianTimestamp(rateFeedId);

    mockAggregator0.setRoundData(aggregatorPrice0, latestTimestamp - 1);
    mockAggregator1.setRoundData(aggregatorPrice1, latestTimestamp - 1);
    mockAggregator2.setRoundData(aggregatorPrice2, latestTimestamp - 1);
    mockAggregator3.setRoundData(aggregatorPrice3, latestTimestamp - 1);

    vm.expectRevert(TIMESTAMP_NOT_NEW_ERROR);
    relayer.relay();
  }

  function test_relaysOnRecentTimestamp3() public withReport(aReport) {
    vm.expectRevert(TIMESTAMP_NOT_NEW_ERROR);
    relayer.relay();
    skip(50);
    mockAggregator3.setRoundData(aggregatorPrice3, block.timestamp);
    relayer.relay();
    uint256 timestamp = sortedOracles.medianTimestamp(rateFeedId);
    assertEq(timestamp, block.timestamp);
  }

  function test_revertsOnRepeatTimestamp3() public withReport(aReport) {
    uint256 latestTimestamp = sortedOracles.medianTimestamp(rateFeedId);
    mockAggregator3.setRoundData(aggregatorPrice3, latestTimestamp);
    vm.expectRevert(TIMESTAMP_NOT_NEW_ERROR);
    relayer.relay();
  }

  function test_revertsWhenMostRecentTimestampIsExpired() public override withReport(aReport) {
    mockAggregator0.setRoundData(aggregatorPrice0, block.timestamp + 1);
    mockAggregator1.setRoundData(aggregatorPrice1, block.timestamp + 1);
    mockAggregator2.setRoundData(aggregatorPrice2, block.timestamp + 1);
    mockAggregator2.setRoundData(aggregatorPrice3, block.timestamp + 1);

    vm.warp(block.timestamp + expirySeconds + 1);
    vm.expectRevert(EXPIRED_TIMESTAMP_ERROR);
    relayer.relay();
  }

  function test_revertsWhenTimestampSpreadTooLarge() public virtual override {
    mockAggregator0.setRoundData(aggregatorPrice0, block.timestamp);
    vm.warp(block.timestamp + 301);
    mockAggregator3.setRoundData(aggregatorPrice3, block.timestamp);
    vm.expectRevert(TIMESTAMP_SPREAD_TOO_HIGH_ERROR);
    relayer.relay();
  }
}
