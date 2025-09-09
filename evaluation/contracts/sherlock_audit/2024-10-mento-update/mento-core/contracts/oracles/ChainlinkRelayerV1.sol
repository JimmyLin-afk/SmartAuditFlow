// SPDX-License-Identifier: GPL-3.0-or-later
// solhint-disable immutable-vars-naming
pragma solidity 0.8.18;

import "../interfaces/IChainlinkRelayer.sol";
import "foundry-chainlink-toolkit/src/interfaces/feeds/AggregatorV3Interface.sol";
import { UD60x18, ud, intoUint256 } from "prb/math/UD60x18.sol";

/**
 * @notice The minimal subset of the SortedOracles interface needed by the
 * relayer.
 * @dev SortedOracles is a Solidity 5.13 contract, thus we can't import the
 * interface directly, so we use a minimal hand-copied one.
 * See https://github.com/mento-protocol/mento-core/blob/develop/contracts/common/SortedOracles.sol
 */
interface ISortedOraclesMin {
  function report(address rateFeedId, uint256 value, address lesserKey, address greaterKey) external;

  function getRates(address rateFeedId) external returns (address[] memory, uint256[] memory, uint256[] memory);

  function medianTimestamp(address rateFeedId) external view returns (uint256);

  function getTokenReportExpirySeconds(address rateFeedId) external view returns (uint256);

  function removeExpiredReports(address rateFeedId, uint256 n) external;
}

/**
 * @title ChainlinkRelayer
 * @notice The ChainlinkRelayer relays rate feed data from a Chainlink price feed, or
 * an aggregation of multiple Chainlink price feeds to the SortedOracles contract.
 * A separate instance should be deployed for each rate feed.
 * @dev Assumes that it is the only reporter for the given SortedOracles feed.
 * This contract aggregates multiple Chainlink price feeds in order to provide derived rate feeds
 * to the rest of the protocol. This is needed because it is more efficient for oracle providers
 * to report FX rates against the dollar and crypto-asset rates against the dollar,
 * instead of all possible combinations.
 * For example, for the Philippine Peso, Chainlink reports PHP/USD, but does not report CELO/PHP
 * which is required to pay for gas in a PHP stable token. But using both PHP/USD and CELO/USD,
 * one can create a path: CELO/USD * inverse(PHP/USD) = CELO/PHP.
 * Because of this we can provide up to four Chainlink price sources with inversion settings
 * to the relayer, a price path. The path segments are chained through multiplication and
 * inversion to derive the rate.
 */
contract ChainlinkRelayerV1 is IChainlinkRelayer {
  /**
   * @notice The number of digits after the decimal point in FixidityLib values, as used by SortedOracles.
   * @dev See contracts/common/FixidityLib.sol
   */
  uint256 private constant UD60X18_TO_FIXIDITY_SCALE = 1e6; // 10 ** (24 - 18)

  /// @notice The rateFeedId this relayer relays for.
  address public immutable rateFeedId;

  /// @notice The address of the SortedOracles contract to report to.
  address public immutable sortedOracles;

  /**
   * @dev We store an array of up to four IChainlinkRelayer.ChainlinkAggregator structs
   * in the following immutable variables.
   * aggregator<i> stores the i-th ChainlinkAggregator.aggregator member.
   * invert<i> stores the i-th ChainlinkAggregator.invert member.
   * aggregatorCount stores the length of the array.
   * These are built back up into an in-memory array in the buildAggregatorArray function.
   */

  /// @notice The addresses of the Chainlink aggregators this contract fetches data from.
  address private immutable aggregator0;
  address private immutable aggregator1;
  address private immutable aggregator2;
  address private immutable aggregator3;

  /// @notice The invert setting for each aggregator, if true it flips the rate feed, i.e. CELO/USD -> USD/CELO.
  bool private immutable invert0;
  bool private immutable invert1;
  bool private immutable invert2;
  bool private immutable invert3;

  /// @notice The number of aggregators provided during construction 1 <= aggregatorCount <= 4.
  uint256 private immutable aggregatorCount;

  /**
   * @notice Maximum timestamp deviation allowed between all report timestamps pulled
   * from the Chainlink aggregators.
   * @dev Only relevant when aggregatorCount > 1.
   */
  uint256 public immutable maxTimestampSpread;

  /**
   * @notice Human-readable description of the rate feed.
   * @dev Should only be used off-chain for easier debugging / UI generation,
   * thus the only storage related gas spend occurs in the constructor.
   */
  string public rateFeedDescription;

  /// @notice Used when an empty array of aggregators is passed into the constructor.
  error NoAggregators();

  /// @notice Used when more than four aggregators are passed into the constructor.
  error TooManyAggregators();

  /// @notice Used when a) there is more than 1 aggregator and the maxTimestampSpread is 0,
  /// OR b) when there is only 1 aggregator and the maxTimestampSpread is not 0.
  error InvalidMaxTimestampSpread();

  /// @notice Used when a new price's timestamp is not newer than the most recent SortedOracles timestamp.
  error TimestampNotNew();

  /// @notice Used when a new price's timestamp would be considered expired by SortedOracles.
  error ExpiredTimestamp();

  /// @notice Used when a negative or zero price is returned by the Chainlink aggregator.
  error InvalidPrice();
  /**
   * @notice Used when the spread between the earliest and latest timestamp
   * of the aggregators is above the maximum allowed.
   */
  error TimestampSpreadTooHigh();
  /**
   * @notice Used when trying to recover from a lesser/greater revert and there are
   * too many existing reports in SortedOracles.
   */
  error TooManyExistingReports();

  /**
   * @notice Used in the constructor when a ChainlinkAggregator
   * has address(0) for an aggregator.
   */
  error InvalidAggregator();

  /**
   * @notice Initializes the contract and sets immutable parameters.
   * @param _rateFeedId ID of the rate feed this relayer instance relays for.
   * @param _rateFeedDescription The human-readable description of the reported rate feed.
   * @param _sortedOracles Address of the SortedOracles contract to relay to.
   * @param _maxTimestampSpread Max difference in milliseconds between the earliest and
   *        latest timestamp of all aggregators in the price path.
   * @param _aggregators Array of ChainlinkAggregator structs defining the price path.
   */
  constructor(
    address _rateFeedId,
    string memory _rateFeedDescription,
    address _sortedOracles,
    uint256 _maxTimestampSpread,
    ChainlinkAggregator[] memory _aggregators
  ) {
    rateFeedId = _rateFeedId;
    sortedOracles = _sortedOracles;
    maxTimestampSpread = _maxTimestampSpread;
    rateFeedDescription = _rateFeedDescription;

    aggregatorCount = _aggregators.length;
    if (aggregatorCount == 0) {
      revert NoAggregators();
    }

    if (aggregatorCount > 4) {
      revert TooManyAggregators();
    }

    if ((aggregatorCount > 1 && _maxTimestampSpread == 0) || (aggregatorCount == 1 && _maxTimestampSpread != 0)) {
      revert InvalidMaxTimestampSpread();
    }

    ChainlinkAggregator[] memory aggregators = new ChainlinkAggregator[](4);
    for (uint256 i = 0; i < _aggregators.length; i++) {
      if (_aggregators[i].aggregator == address(0)) {
        revert InvalidAggregator();
      }

      aggregators[i] = _aggregators[i];
    }

    aggregator0 = aggregators[0].aggregator;
    aggregator1 = aggregators[1].aggregator;
    aggregator2 = aggregators[2].aggregator;
    aggregator3 = aggregators[3].aggregator;
    invert0 = aggregators[0].invert;
    invert1 = aggregators[1].invert;
    invert2 = aggregators[2].invert;
    invert3 = aggregators[3].invert;
  }

  /**
   * @notice Get the Chainlink aggregators and their invert settings.
   * @return An array of ChainlinkAggregator segments that compose the price path.
   */
  function getAggregators() external view returns (ChainlinkAggregator[] memory) {
    return buildAggregatorArray();
  }

  /**
   * @notice Relays data from the configured Chainlink aggregator to SortedOracles.
   * @dev Checks the price is non-negative (Chainlink uses `int256` rather than `uint256`.
   * @dev Converts the price to a Fixidity value, as expected by SortedOracles.
   * @dev Performs checks on the timestamp, will revert if any fails:
   *      - The most recent Chainlink timestamp should be strictly newer than the most
   *        recent timestamp in SortedOracles.
   *      - The most recent Chainlink timestamp should not be considered expired by SortedOracles.
   *      - The spread between aggregator timestamps is less than the maxTimestampSpread.
   */
  function relay() external {
    ISortedOraclesMin _sortedOracles = ISortedOraclesMin(sortedOracles);
    ChainlinkAggregator[] memory aggregators = buildAggregatorArray();

    (UD60x18 report, uint256 timestamp) = readChainlinkAggregator(aggregators[0]);
    uint256 oldestChainlinkTs = timestamp;
    uint256 newestChainlinkTs = timestamp;

    UD60x18 nextReport;
    for (uint256 i = 1; i < aggregators.length; i++) {
      (nextReport, timestamp) = readChainlinkAggregator(aggregators[i]);
      report = report.mul(nextReport);
      oldestChainlinkTs = timestamp < oldestChainlinkTs ? timestamp : oldestChainlinkTs;
      newestChainlinkTs = timestamp > newestChainlinkTs ? timestamp : newestChainlinkTs;
    }

    if (newestChainlinkTs - oldestChainlinkTs > maxTimestampSpread) {
      revert TimestampSpreadTooHigh();
    }

    uint256 lastReportTs = _sortedOracles.medianTimestamp(rateFeedId);

    if (lastReportTs > 0 && newestChainlinkTs <= lastReportTs) {
      revert TimestampNotNew();
    }

    if (isTimestampExpired(newestChainlinkTs)) {
      revert ExpiredTimestamp();
    }

    reportRate(intoUint256(report) * UD60X18_TO_FIXIDITY_SCALE);
  }

  /**
   * @notice Report by looking up existing reports and building the lesser and greater keys.
   * @dev Depending on the state in SortedOracles we can be in the:
   *   - Happy path: No reports, or a single report from this relayer.
   *     We can report with lesser and greater keys as address(0)
   *   - Unhappy path: There are reports from other oracles.
   *     We restrain this path by only computing lesser and greater keys when there is
   *     at most one report from a different oracle.
   *     We also attempt to expire reports in order to get back to the happy path.
   * @param rate The rate to report.
   */
  function reportRate(uint256 rate) internal {
    // slither-disable-next-line unused-return
    (address[] memory oracles, uint256[] memory rates, ) = ISortedOraclesMin(sortedOracles).getRates(rateFeedId);
    uint256 numRates = oracles.length;

    if (numRates == 0 || (numRates == 1 && oracles[0] == address(this))) {
      // Happy path: SortedOracles is empty, or there is a single report from this relayer.
      ISortedOraclesMin(sortedOracles).report(rateFeedId, rate, address(0), address(0));
      return;
    }

    if (numRates > 2 || (numRates == 2 && oracles[0] != address(this) && oracles[1] != address(this))) {
      revert TooManyExistingReports();
    }

    // At this point we have ensured that either:
    // - There is a single report from another oracle.
    // - There are two reports and one is from this relayer.

    address otherOracle;
    uint256 otherRate;

    if (numRates == 1 || oracles[0] != address(this)) {
      otherOracle = oracles[0];
      otherRate = rates[0];
    } else {
      otherOracle = oracles[1];
      otherRate = rates[1];
    }

    // slither-disable-start uninitialized-local
    address lesserKey;
    address greaterKey;
    // slither-disable-end uninitialized-local

    if (otherRate < rate) {
      lesserKey = otherOracle;
    } else {
      greaterKey = otherOracle;
    }

    ISortedOraclesMin(sortedOracles).report(rateFeedId, rate, lesserKey, greaterKey);
    ISortedOraclesMin(sortedOracles).removeExpiredReports(rateFeedId, 1);
  }

  /**
   * @notice Read and validate a Chainlink report from an aggregator.
   * It inverts the value if necessary.
   * @return price UD60x18 report value.
   * @return timestamp uint256 timestamp of the report.
   */
  function readChainlinkAggregator(ChainlinkAggregator memory aggCfg) internal view returns (UD60x18, uint256) {
    // slither-disable-next-line unused-return,calls-loop
    (, int256 _price, , uint256 timestamp, ) = AggregatorV3Interface(aggCfg.aggregator).latestRoundData();
    if (_price <= 0) {
      revert InvalidPrice();
    }
    UD60x18 price = chainlinkToUD60x18(_price, aggCfg.aggregator);
    if (aggCfg.invert) {
      price = price.inv();
    }
    return (price, timestamp);
  }

  /**
   * @notice Compose immutable variables into an in-memory array for better handling.
   * @return aggregators An array of ChainlinkAggregator structs.
   */
  function buildAggregatorArray() internal view returns (ChainlinkAggregator[] memory aggregators) {
    aggregators = new ChainlinkAggregator[](aggregatorCount);
    unchecked {
      aggregators[0] = ChainlinkAggregator(aggregator0, invert0);
      if (aggregatorCount > 1) {
        aggregators[1] = ChainlinkAggregator(aggregator1, invert1);
        if (aggregatorCount > 2) {
          aggregators[2] = ChainlinkAggregator(aggregator2, invert2);
          if (aggregatorCount > 3) {
            aggregators[3] = ChainlinkAggregator(aggregator3, invert3);
          }
        }
      }
    }
  }

  /**
   * @notice Checks if a Chainlink price's timestamp would be expired in SortedOracles.
   * @param timestamp The timestamp returned by the Chainlink aggregator.
   * @return `true` if expired based on SortedOracles expiry parameter.
   */
  function isTimestampExpired(uint256 timestamp) internal view returns (bool) {
    return block.timestamp - timestamp >= ISortedOraclesMin(sortedOracles).getTokenReportExpirySeconds(rateFeedId);
  }

  /**
   * @notice Converts a Chainlink price to a UD60x18 value.
   * @param price A price from the Chainlink aggregator.
   * @return The converted UD60x18 value.
   */
  function chainlinkToUD60x18(int256 price, address aggregator) internal view returns (UD60x18) {
    // slither-disable-next-line calls-loop
    uint256 chainlinkDecimals = uint256(AggregatorV3Interface(aggregator).decimals());
    return ud(uint256(price) * 10 ** (18 - chainlinkDecimals));
  }
}
