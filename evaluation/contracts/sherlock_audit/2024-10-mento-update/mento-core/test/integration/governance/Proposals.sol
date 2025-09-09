// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;
// solhint-disable func-name-mixedcase, max-line-length

import { uints, addresses, bytesList } from "mento-std/Array.sol";

import { ProxyAdmin } from "openzeppelin-contracts-next/contracts/proxy/transparent/ProxyAdmin.sol";

import { MentoGovernor } from "contracts/governance/MentoGovernor.sol";
import { TimelockController } from "contracts/governance/TimelockController.sol";
import { Locking } from "contracts/governance/locking/Locking.sol";
import { Emission } from "contracts/governance/Emission.sol";

library Proposals {
  function _proposeChangeEmissionTarget(
    MentoGovernor mentoGovernor,
    Emission emission,
    address newTarget
  )
    internal
    returns (
      uint256 proposalId,
      address[] memory targets,
      uint256[] memory values,
      bytes[] memory calldatas,
      string memory description
    )
  {
    targets = addresses(address(emission));
    values = uints(0);
    calldatas = bytesList(abi.encodeWithSelector(emission.setEmissionTarget.selector, newTarget));
    description = "Change emission target";

    proposalId = mentoGovernor.propose(targets, values, calldatas, description);
  }

  struct changeSettingsVars {
    uint256 votingDelay;
    uint256 votingPeriod;
    uint256 threshold;
    uint256 quorum;
    uint256 minDelay;
    uint32 minCliff;
    uint32 minSlope;
  }

  struct changeSettingsContracts {
    MentoGovernor mentoGovernor;
    TimelockController timelockController;
    Locking locking;
  }

  function _proposeChangeSettings(
    changeSettingsContracts memory _targets,
    changeSettingsVars memory vars
  )
    internal
    returns (
      uint256 proposalId,
      address[] memory targets,
      uint256[] memory values,
      bytes[] memory calldatas,
      string memory description
    )
  {
    targets = addresses(
      address(_targets.mentoGovernor),
      address(_targets.mentoGovernor),
      address(_targets.mentoGovernor),
      address(_targets.mentoGovernor),
      address(_targets.timelockController),
      address(_targets.locking),
      address(_targets.locking)
    );
    values = uints(0, 0, 0, 0, 0, 0, 0);
    calldatas = bytesList(
      abi.encodeWithSelector(_targets.mentoGovernor.setVotingDelay.selector, vars.votingDelay),
      abi.encodeWithSelector(_targets.mentoGovernor.setVotingPeriod.selector, vars.votingPeriod),
      abi.encodeWithSelector(_targets.mentoGovernor.setProposalThreshold.selector, vars.threshold),
      abi.encodeWithSelector(_targets.mentoGovernor.updateQuorumNumerator.selector, vars.quorum),
      abi.encodeWithSelector(_targets.timelockController.updateDelay.selector, vars.minDelay),
      abi.encodeWithSelector(_targets.locking.setMinCliffPeriod.selector, vars.minCliff),
      abi.encodeWithSelector(_targets.locking.setMinSlopePeriod.selector, vars.minSlope)
    );
    description = "Change governance config";

    proposalId = _targets.mentoGovernor.propose(targets, values, calldatas, description);
  }

  function _proposeUpgradeContracts(
    MentoGovernor mentoGovernor,
    ProxyAdmin proxyAdmin,
    address[] memory proxies,
    address[] memory newImplementations
  )
    internal
    returns (
      uint256 proposalId,
      address[] memory targets,
      uint256[] memory values,
      bytes[] memory calldatas,
      string memory description
    )
  {
    targets = new address[](proxies.length);
    calldatas = new bytes[](proxies.length);
    values = new uint256[](proxies.length);
    for (uint256 i = 0; i < proxies.length; i++) {
      targets[i] = address(proxyAdmin);
      calldatas[i] = abi.encodeWithSelector(proxyAdmin.upgrade.selector, proxies[i], newImplementations[i]);
    }
    description = "Upgrade upgradeable contracts";

    proposalId = mentoGovernor.propose(targets, values, calldatas, description);
  }

  function _proposeCancelQueuedTx(
    MentoGovernor mentoGovernor,
    TimelockController mentoLabsTreasury,
    bytes32 id
  )
    internal
    returns (
      uint256 proposalId,
      address[] memory targets,
      uint256[] memory values,
      bytes[] memory calldatas,
      string memory description
    )
  {
    targets = addresses(address(mentoLabsTreasury));
    values = uints(0);
    calldatas = bytesList(abi.encodeWithSelector(mentoLabsTreasury.cancel.selector, id));
    description = "Cancel queued tx";

    proposalId = mentoGovernor.propose(targets, values, calldatas, description);
  }
}
