// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;
// solhint-disable max-line-length

import {
  GovernorUpgradeable,
  IGovernorUpgradeable
} from "openzeppelin-contracts-upgradeable/contracts/governance/GovernorUpgradeable.sol";
import {
  GovernorSettingsUpgradeable
} from "openzeppelin-contracts-upgradeable/contracts/governance/extensions/GovernorSettingsUpgradeable.sol";
import {
  GovernorCompatibilityBravoUpgradeable
} from "openzeppelin-contracts-upgradeable/contracts/governance/compatibility/GovernorCompatibilityBravoUpgradeable.sol";
import {
  GovernorVotesUpgradeable,
  IVotesUpgradeable
} from "openzeppelin-contracts-upgradeable/contracts/governance/extensions/GovernorVotesUpgradeable.sol";
import {
  GovernorVotesQuorumFractionUpgradeable
} from "openzeppelin-contracts-upgradeable/contracts/governance/extensions/GovernorVotesQuorumFractionUpgradeable.sol";
import {
  GovernorTimelockControlUpgradeable
} from "openzeppelin-contracts-upgradeable/contracts/governance/extensions/GovernorTimelockControlUpgradeable.sol";
import {
  TimelockControllerUpgradeable,
  IERC165Upgradeable
} from "openzeppelin-contracts-upgradeable/contracts/governance/TimelockControllerUpgradeable.sol";

/**
 * @title Mento Governor
 * @author Mento Labs
 * @notice Governor contract extending on OpenZeppelin's upgradeable governance contracts.
 */
contract MentoGovernor is
  GovernorUpgradeable,
  GovernorSettingsUpgradeable,
  GovernorCompatibilityBravoUpgradeable,
  GovernorVotesUpgradeable,
  GovernorVotesQuorumFractionUpgradeable,
  GovernorTimelockControlUpgradeable
{
  /**
   * @notice Initializes the MentoGovernor with voting, settings, compatibility, and timelock configurations.
   * @param veToken The escrowed Mento Token used for voting.
   * @param timelockController The timelock controller used by the governor.
   * @param votingDelay_ The delay time in blocks between the proposal creation and the start of voting.
   * @param votingPeriod_ The voting duration in blocks between the vote start and vote end.
   * @param threshold_ The number of votes required in order for a voter to become a proposer.
   * @param quorum_ The minimum number of votes in percent of total supply required in order for a proposal to succeed.
   */
  // solhint-disable-next-line func-name-mixedcase
  function __MentoGovernor_init(
    IVotesUpgradeable veToken,
    TimelockControllerUpgradeable timelockController,
    uint256 votingDelay_,
    uint256 votingPeriod_,
    uint256 threshold_,
    uint256 quorum_
  ) external initializer {
    __Governor_init("MentoGovernor");
    __GovernorSettings_init(votingDelay_, votingPeriod_, threshold_);
    __GovernorCompatibilityBravo_init();
    __GovernorVotes_init(veToken);
    __GovernorVotesQuorumFraction_init(quorum_);
    __GovernorTimelockControl_init(timelockController);
  }

  function votingDelay() public view override(IGovernorUpgradeable, GovernorSettingsUpgradeable) returns (uint256) {
    return super.votingDelay();
  }

  function votingPeriod() public view override(IGovernorUpgradeable, GovernorSettingsUpgradeable) returns (uint256) {
    return super.votingPeriod();
  }

  function proposalThreshold()
    public
    view
    override(GovernorUpgradeable, GovernorSettingsUpgradeable)
    returns (uint256)
  {
    return super.proposalThreshold();
  }

  function quorum(
    uint256 blockNumber
  ) public view override(IGovernorUpgradeable, GovernorVotesQuorumFractionUpgradeable) returns (uint256) {
    return super.quorum(blockNumber);
  }

  function getVotes(
    address account,
    uint256 blockNumber
  ) public view override(GovernorUpgradeable, IGovernorUpgradeable) returns (uint256) {
    return super.getVotes(account, blockNumber);
  }

  function state(
    uint256 proposalId
  )
    public
    view
    override(GovernorUpgradeable, IGovernorUpgradeable, GovernorTimelockControlUpgradeable)
    returns (ProposalState)
  {
    return super.state(proposalId);
  }

  function propose(
    address[] memory targets,
    uint256[] memory values,
    bytes[] memory calldatas,
    string memory description
  )
    public
    override(GovernorUpgradeable, GovernorCompatibilityBravoUpgradeable, IGovernorUpgradeable)
    returns (uint256)
  {
    return super.propose(targets, values, calldatas, description);
  }

  function _execute(
    uint256 proposalId,
    address[] memory targets,
    uint256[] memory values,
    bytes[] memory calldatas,
    bytes32 descriptionHash
  ) internal override(GovernorUpgradeable, GovernorTimelockControlUpgradeable) {
    super._execute(proposalId, targets, values, calldatas, descriptionHash);
  }

  function _cancel(
    address[] memory targets,
    uint256[] memory values,
    bytes[] memory calldatas,
    bytes32 descriptionHash
  ) internal override(GovernorUpgradeable, GovernorTimelockControlUpgradeable) returns (uint256) {
    return super._cancel(targets, values, calldatas, descriptionHash);
  }

  function _executor()
    internal
    view
    override(GovernorUpgradeable, GovernorTimelockControlUpgradeable)
    returns (address)
  {
    return super._executor();
  }

  function supportsInterface(
    bytes4 interfaceId
  ) public view override(GovernorUpgradeable, IERC165Upgradeable, GovernorTimelockControlUpgradeable) returns (bool) {
    return super.supportsInterface(interfaceId);
  }
}
