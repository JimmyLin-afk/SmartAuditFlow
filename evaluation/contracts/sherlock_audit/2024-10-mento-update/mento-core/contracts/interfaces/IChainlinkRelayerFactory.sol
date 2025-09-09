// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity >=0.5.13 <0.9;
pragma experimental ABIEncoderV2;

import "./IChainlinkRelayer.sol";

interface IChainlinkRelayerFactory {
  /**
   * @notice Emitted when a relayer is deployed.
   * @param relayerAddress Address of the newly deployed relayer.
   * @param rateFeedId Rate feed ID for which the relayer will report.
   * @param rateFeedDescription Human-readable rate feed, which the relayer will report on, i.e. "CELO/USD"
   * @param aggregators List of ChainlinkAggregator that the relayer chains.
   */
  event RelayerDeployed(
    address indexed relayerAddress,
    address indexed rateFeedId,
    string rateFeedDescription,
    IChainlinkRelayer.ChainlinkAggregator[] aggregators
  );

  /**
   * @notice Emitted when the relayer deployer is updated.
   * @param newRelayerDeployer Address of the new relayer deployer.
   * @param oldRelayerDeployer Address of the old relayer deployer.
   */
  event RelayerDeployerUpdated(address indexed newRelayerDeployer, address indexed oldRelayerDeployer);

  /**
   * @notice Emitted when a relayer is removed.
   * @param relayerAddress Address of the removed relayer.
   * @param rateFeedId Rate feed ID for which the relayer reported.
   */
  event RelayerRemoved(address indexed relayerAddress, address indexed rateFeedId);

  function initialize(address _sortedOracles, address _relayerDeployer) external;

  function sortedOracles() external returns (address);

  function setRelayerDeployer(address _relayerDeployer) external;

  function relayerDeployer() external returns (address);

  function deployRelayer(
    address rateFeedId,
    string calldata rateFeedDescription,
    uint256 maxTimestampSpread,
    IChainlinkRelayer.ChainlinkAggregator[] calldata aggregators
  ) external returns (address);

  function removeRelayer(address rateFeedId) external;

  function redeployRelayer(
    address rateFeedId,
    string calldata rateFeedDescription,
    uint256 maxTimestampSpread,
    IChainlinkRelayer.ChainlinkAggregator[] calldata aggregators
  ) external returns (address);

  function getRelayer(address rateFeedId) external view returns (address);

  function getRelayers() external view returns (address[] memory);

  function computedRelayerAddress(
    address rateFeedId,
    string calldata rateFeedDescription,
    uint256 maxTimestampSpread,
    IChainlinkRelayer.ChainlinkAggregator[] calldata aggregators
  ) external returns (address);
}
