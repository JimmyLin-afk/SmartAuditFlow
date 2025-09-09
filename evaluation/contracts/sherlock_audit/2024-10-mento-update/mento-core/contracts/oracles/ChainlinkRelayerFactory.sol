// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;

import { OwnableUpgradeable } from "openzeppelin-contracts-upgradeable/contracts/access/OwnableUpgradeable.sol";
import { ChainlinkRelayerV1 } from "./ChainlinkRelayerV1.sol";
import { IChainlinkRelayer } from "../interfaces/IChainlinkRelayer.sol";
import { IChainlinkRelayerFactory } from "../interfaces/IChainlinkRelayerFactory.sol";

/**
 * @title ChainlinkRelayerFactory
 * @notice The ChainlinkRelayerFactory creates and keeps track of ChainlinkRelayers.
 */
contract ChainlinkRelayerFactory is IChainlinkRelayerFactory, OwnableUpgradeable {
  /// @notice Address of the SortedOracles contract.
  address public sortedOracles;

  /// @notice Maps a rate feed ID to the relayer contract most recently deployed by this contract.
  mapping(address rateFeedId => ChainlinkRelayerV1 relayer) public deployedRelayers;

  /**
   * @notice List of rate feed IDs for which a relayer has been deployed.
   * @dev Used to enumerate the `deployedRelayer` mapping.
   */
  address[] public rateFeeds;

  /**
   * @notice Account that is allowed to deploy relayers.
   */
  address public relayerDeployer;

  /**
   * @notice Thrown when trying to deploy a relayer to an address that already has code.
   * @param contractAddress Address at which the relayer could not be deployed.
   * @param rateFeedId Rate feed ID for which the relayer would have reported.
   */
  error ContractAlreadyExists(address contractAddress, address rateFeedId);

  /**
   * @notice Thrown when trying to deploy a relayer for a rate feed ID that already has a relayer.
   * @param rateFeedId The rate feed ID for which a relayer already exists.
   * @dev A new relayer *can* be deployed for the same rate feed ID, but only with a different
   *      chainlink aggregator or bytecode with `redeployRelayer`.
   */
  error RelayerForFeedExists(address rateFeedId);

  /**
   * @notice Thrown when the sanity check to verify the CREATE2 address computation fails.
   * @param expectedAddress The address expected by local computation of the CREATE2 address.
   * @param returnedAddress The address actually returned by CREATE2.
   */
  error UnexpectedAddress(address expectedAddress, address returnedAddress);

  /**
   * @notice Thrown when trying to remove a relayer for a rate feed ID that doesn't have a relayer.
   * @param rateFeedId The rate feed ID.
   */
  error NoRelayerForRateFeedId(address rateFeedId);

  /// @notice Thrown when a non-deployer tries to call a deployer-only function.
  error NotAllowed();

  /// @notice Modifier to restrict a function to the deployer.
  modifier onlyDeployer() {
    if (msg.sender != relayerDeployer && msg.sender != owner()) {
      revert NotAllowed();
    }
    _;
  }

  /**
   * @notice Constructor for the logic contract.
   * @param disable If `true`, disables the initializer.
   * @dev This contract is meant to be deployed with an upgradeable proxy in
   * front of it. Set `disable` to `true` in production environments to disable
   * contract initialization on the logic contract, only allowing initialization
   * on the proxy.
   */
  constructor(bool disable) {
    if (disable) {
      _disableInitializers();
    }
  }

  /**
   * @notice Initializes the factory.
   * @param _sortedOracles The SortedOracles instance deployed relayers should report to.
   */
  function initialize(address _sortedOracles, address _relayerDeployer) external initializer {
    __Ownable_init();
    sortedOracles = _sortedOracles;
    relayerDeployer = _relayerDeployer;
  }

  /**
   * @notice Sets the address of the relayer deployer.
   * @param newRelayerDeployer The address of the relayer deployer.
   */
  function setRelayerDeployer(address newRelayerDeployer) external onlyOwner {
    address oldRelayerDeployer = relayerDeployer;
    relayerDeployer = newRelayerDeployer;
    emit RelayerDeployerUpdated(newRelayerDeployer, oldRelayerDeployer);
  }

  /**
   * @notice Deploys a new relayer contract.
   * @param rateFeedId The rate feed ID for which the relayer will report.
   * @param rateFeedDescription Human-readable rate feed, which the relayer will report on, i.e. "CELO/USD".
   * @param maxTimestampSpread Max difference in milliseconds between the earliest and
   *        latest timestamp of all aggregators in the price path.
   * @param aggregators Array of ChainlinkAggregator structs defining the price path.
   *        See contract-level @dev comment in the ChainlinkRelayerV1 contract,
   *        for an explanation on price paths.
   * @return relayerAddress The address of the newly deployed relayer contract.
   */
  function deployRelayer(
    address rateFeedId,
    string calldata rateFeedDescription,
    uint256 maxTimestampSpread,
    IChainlinkRelayer.ChainlinkAggregator[] calldata aggregators
  ) public onlyDeployer returns (address relayerAddress) {
    if (address(deployedRelayers[rateFeedId]) != address(0)) {
      revert RelayerForFeedExists(rateFeedId);
    }

    address expectedAddress = computedRelayerAddress(rateFeedId, rateFeedDescription, maxTimestampSpread, aggregators);
    if (expectedAddress.code.length > 0) {
      revert ContractAlreadyExists(expectedAddress, rateFeedId);
    }

    bytes32 salt = _getSalt();
    ChainlinkRelayerV1 relayer = new ChainlinkRelayerV1{ salt: salt }(
      rateFeedId,
      rateFeedDescription,
      sortedOracles,
      maxTimestampSpread,
      aggregators
    );

    if (address(relayer) != expectedAddress) {
      revert UnexpectedAddress(expectedAddress, address(relayer));
    }

    deployedRelayers[rateFeedId] = relayer;
    rateFeeds.push(rateFeedId);

    emit RelayerDeployed(address(relayer), rateFeedId, rateFeedDescription, aggregators);

    return address(relayer);
  }

  /**
   * @notice Removes a relayer from the list of deployed relayers.
   * @param rateFeedId The rate feed whose relayer should be removed.
   */
  function removeRelayer(address rateFeedId) public onlyDeployer {
    address relayerAddress = address(deployedRelayers[rateFeedId]);

    if (relayerAddress == address(0)) {
      revert NoRelayerForRateFeedId(rateFeedId);
    }

    delete deployedRelayers[rateFeedId];

    uint256 lastRateFeedIndex = rateFeeds.length - 1;

    for (uint256 i = 0; i <= lastRateFeedIndex; i++) {
      if (rateFeeds[i] == rateFeedId) {
        rateFeeds[i] = rateFeeds[lastRateFeedIndex];
        rateFeeds.pop();
        break;
      }
    }

    emit RelayerRemoved(relayerAddress, rateFeedId);
  }

  /**
   * @notice Removes the current relayer and redeploys a new one with a different
   *         Chainlink aggregator (and/or different bytecode if the factory
   *         has been upgraded since the last deployment of the relayer).
   * @param rateFeedId The rate feed ID for which the relayer will report.
   * @param rateFeedDescription Human-readable rate feed, which the relayer will report on, i.e. "CELO/USD".
   * @param maxTimestampSpread Max difference in milliseconds between the earliest and
   *        latest timestamp of all aggregators in the price path.
   * @param aggregators Array of ChainlinkAggregator structs defining the price path.
   * @return relayerAddress The address of the newly deployed relayer contract.
   */
  function redeployRelayer(
    address rateFeedId,
    string calldata rateFeedDescription,
    uint256 maxTimestampSpread,
    IChainlinkRelayer.ChainlinkAggregator[] calldata aggregators
  ) external onlyDeployer returns (address relayerAddress) {
    removeRelayer(rateFeedId);
    return deployRelayer(rateFeedId, rateFeedDescription, maxTimestampSpread, aggregators);
  }

  /**
   * @notice Returns the address of the currently deployed relayer for a given rate feed ID.
   * @param rateFeedId The rate feed ID whose relayer we want to get.
   * @return relayerAddress Address of the relayer contract.
   */
  function getRelayer(address rateFeedId) external view returns (address relayerAddress) {
    return address(deployedRelayers[rateFeedId]);
  }

  /**
   * @notice Returns a list of all currently deployed relayers.
   * @return relayerAddresses An array of all relayer contract addresses.
   */
  function getRelayers() external view returns (address[] memory relayerAddresses) {
    address[] memory relayers = new address[](rateFeeds.length);
    // slither-disable-next-line cache-array-length
    for (uint256 i = 0; i < rateFeeds.length; i++) {
      relayers[i] = address(deployedRelayers[rateFeeds[i]]);
    }
    return relayers;
  }

  /**
   * @notice Returns the salt used for CREATE2 deployment of relayer contracts.
   * @return salt The `bytes32` constant `keccak256("mento.chainlinkRelayer")`.
   * @dev We're using CREATE2 and all the data we want to use for address
   *      generation is included in the init code and constructor arguments, so a
   *      constant salt is enough.
   */
  function _getSalt() internal pure returns (bytes32 salt) {
    return keccak256("mento.chainlinkRelayer");
  }

  /**
   * @notice Computes the expected CREATE2 address for given relayer parameters.
   * @param rateFeedId The rate feed ID.
   * @param rateFeedDescription The human readable description of the reported rate feed.
   * @param maxTimestampSpread Max difference in milliseconds between the earliest and
   *        latest timestamp of all aggregators in the price path.
   * @param aggregators Array of ChainlinkAggregator structs defining the price path.
   * @dev See https://eips.ethereum.org/EIPS/eip-1014.
   */
  function computedRelayerAddress(
    address rateFeedId,
    string calldata rateFeedDescription,
    uint256 maxTimestampSpread,
    IChainlinkRelayer.ChainlinkAggregator[] calldata aggregators
  ) public view returns (address) {
    bytes32 salt = _getSalt();
    return
      address(
        uint160(
          uint256(
            keccak256(
              abi.encodePacked(
                bytes1(0xff),
                address(this),
                salt,
                keccak256(
                  abi.encodePacked(
                    type(ChainlinkRelayerV1).creationCode,
                    abi.encode(rateFeedId, rateFeedDescription, sortedOracles, maxTimestampSpread, aggregators)
                  )
                )
              )
            )
          )
        )
      );
  }
}
