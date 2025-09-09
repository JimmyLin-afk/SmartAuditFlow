// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;
// solhint-disable func-name-mixedcase, max-line-length

import { IVotesUpgradeable } from "openzeppelin-contracts-upgradeable/contracts/governance/extensions/GovernorVotesUpgradeable.sol";

import { GovernanceTest } from "./GovernanceTest.sol";

import { TimelockController } from "contracts/governance/TimelockController.sol";
import { MentoGovernor } from "contracts/governance/MentoGovernor.sol";

import { MockOwnable } from "test/utils/mocks/MockOwnable.sol";
import { MockVeMento } from "test/utils/mocks/MockVeMento.sol";

contract MentoGovernorTest is GovernanceTest {
  TimelockController public timelockController;
  MentoGovernor public mentoGovernor;

  MockOwnable public mockOwnable;
  MockVeMento public mockVeMento;

  uint256 public votingDelay = BLOCKS_DAY;
  uint256 public votingPeriod = BLOCKS_WEEK;
  uint256 public threshold = 1_000e18;
  uint256 public quorum = 10;

  address public communityMultisig = makeAddr("communityMultisig");

  function setUp() public {
    vm.startPrank(owner);

    mockVeMento = new MockVeMento();
    mockOwnable = new MockOwnable();

    timelockController = new TimelockController();
    mentoGovernor = new MentoGovernor();

    address[] memory proposers;
    address[] memory executors;

    timelockController.__MentoTimelockController_init(1 days, proposers, executors, owner, communityMultisig);
    mentoGovernor.__MentoGovernor_init(
      IVotesUpgradeable(address(mockVeMento)),
      timelockController,
      votingDelay,
      votingPeriod,
      threshold,
      quorum
    );

    mockOwnable.transferOwnership(address(timelockController));

    bytes32 proposerRole = timelockController.PROPOSER_ROLE();
    bytes32 executorRole = timelockController.EXECUTOR_ROLE();
    bytes32 adminRole = timelockController.TIMELOCK_ADMIN_ROLE();

    timelockController.grantRole(proposerRole, address(mentoGovernor));
    timelockController.grantRole(executorRole, address(0));
    timelockController.revokeRole(adminRole, owner);

    vm.stopPrank();
  }

  function test_init_shouldSetStateCorrectly() public view {
    assertEq(mentoGovernor.votingDelay(), BLOCKS_DAY);
    assertEq(mentoGovernor.votingPeriod(), BLOCKS_WEEK);
    assertEq(mentoGovernor.proposalThreshold(), 1_000e18);
    assertEq(mentoGovernor.quorumNumerator(), 10);
    assertEq(timelockController.getMinDelay(), 1 days);
  }

  function test_hasRole_shouldReturnCorrectRoles() public view {
    bytes32 proposerRole = timelockController.PROPOSER_ROLE();
    bytes32 executorRole = timelockController.EXECUTOR_ROLE();
    bytes32 adminRole = timelockController.TIMELOCK_ADMIN_ROLE();
    bytes32 cancellerRole = timelockController.CANCELLER_ROLE();

    assert(timelockController.hasRole(proposerRole, address(mentoGovernor)));
    assert(timelockController.hasRole(executorRole, (address(0))));
    assertFalse(timelockController.hasRole(adminRole, owner));
    assert(timelockController.hasRole(cancellerRole, communityMultisig));
  }

  function test_propose_whenProposerBelowThreshold_shouldRevert() public {
    vm.startPrank(alice);

    vm.expectRevert("Governor: proposer votes below proposal threshold");
    _proposeCallProtectedFunction();

    mockVeMento.mint(alice, threshold - 1e18);
    vm.expectRevert("Governor: proposer votes below proposal threshold");
    _proposeCallProtectedFunction();

    mockVeMento.mint(alice, 1e18 - 1);
    vm.expectRevert("Governor: proposer votes below proposal threshold");
    _proposeCallProtectedFunction();

    vm.stopPrank();
  }

  function test_propose_whenProposerAboveThreshold_shouldCreateProposal() public {
    mockVeMento.mint(alice, threshold);

    vm.prank(alice);
    (
      uint256 proposalId,
      address[] memory targets,
      uint256[] memory values,
      bytes[] memory calldatas,
      string memory description
    ) = _proposeCallProtectedFunction();

    uint256 hashedParams = _hashProposal(targets, values, calldatas, description);
    assertEq(proposalId, hashedParams);
    assertEq(uint256(mentoGovernor.state(hashedParams)), 0);
  }

  function test_castVote_whenInVotingDelay_shouldRevert() public {
    mockVeMento.mint(alice, threshold);

    vm.prank(alice);
    (uint256 proposalId, , , , ) = _proposeCallProtectedFunction();

    vm.expectRevert("Governor: vote not currently active");
    vm.prank(bob);
    mentoGovernor.castVote(proposalId, 1);
  }

  function test_castVote_whenVotingPeriodEnds_shouldRevert() public {
    mockVeMento.mint(alice, threshold);
    vm.prank(alice);
    (uint256 proposalId, , , , ) = _proposeCallProtectedFunction();

    vm.roll(block.number + votingDelay + votingPeriod + 1);

    assertEq(uint256(mentoGovernor.state(proposalId)), 3); // defeated

    vm.expectRevert("Governor: vote not currently active");
    vm.prank(bob);
    mentoGovernor.castVote(proposalId, 1);
  }

  function test_castVote_whenNotEnoughForVotes_shouldDefeatProposal() public {
    mockVeMento.mint(alice, threshold);
    mockVeMento.mint(bob, 1_000e18);
    mockVeMento.mint(charlie, 1_001e18);

    vm.prank(alice);
    (uint256 proposalId, , , , ) = _proposeCallProtectedFunction();

    vm.roll(block.number + votingDelay + 1);

    vm.prank(bob);
    mentoGovernor.castVote(proposalId, 1);
    vm.prank(charlie);
    mentoGovernor.castVote(proposalId, 0);

    vm.roll(block.number + votingPeriod);
    assertEq(uint256(mentoGovernor.state(proposalId)), 3); // defeated
  }

  function test_castVote_whenNoQuorum_shouldDefeatProposal() public {
    mockVeMento.mint(alice, threshold);
    mockVeMento.mint(bob, 100e18);

    vm.prank(alice);
    (uint256 proposalId, , , , ) = _proposeCallProtectedFunction();

    vm.roll(block.number + votingDelay + 1);

    vm.prank(bob);
    mentoGovernor.castVote(proposalId, 1);

    vm.roll(block.number + votingPeriod);
    assertEq(uint256(mentoGovernor.state(proposalId)), 3); // defeated
  }

  function test_castVote_whenEnoughQuorumAndForVotes_shouldSucceedProposal() public {
    mockVeMento.mint(alice, threshold);
    mockVeMento.mint(bob, 1_000e18);
    mockVeMento.mint(charlie, 2_000e18);

    vm.prank(alice);
    (uint256 proposalId, , , , ) = _proposeCallProtectedFunction();

    vm.roll(block.number + votingDelay + 1);

    assertEq(uint256(mentoGovernor.state(proposalId)), 1); // active

    vm.prank(bob);
    mentoGovernor.castVote(proposalId, 0);
    vm.prank(charlie);
    mentoGovernor.castVote(proposalId, 1);

    vm.roll(block.number + votingPeriod);
    assertEq(uint256(mentoGovernor.state(proposalId)), 4); // succeeded
  }

  function test_queueAndExecute_whenNotCorrectState_shouldRevert() public {
    mockVeMento.mint(alice, threshold);
    mockVeMento.mint(bob, 1_000e18);
    mockVeMento.mint(charlie, 2_000e18);

    vm.prank(alice);
    (
      uint256 proposalId,
      address[] memory targets,
      uint256[] memory values,
      bytes[] memory calldatas,
      string memory description
    ) = _proposeCallProtectedFunction();

    vm.expectRevert("Governor: proposal not successful");
    mentoGovernor.queue(targets, values, calldatas, keccak256(bytes(description)));

    vm.expectRevert("Governor: proposal not successful");
    mentoGovernor.execute(targets, values, calldatas, keccak256(bytes(description)));

    // keeping block.ts and block.number in sync
    vm.roll(block.number + votingDelay + 1);
    skip(1 days);

    vm.expectRevert("Governor: proposal not successful");
    mentoGovernor.queue(targets, values, calldatas, keccak256(bytes(description)));

    vm.expectRevert("Governor: proposal not successful");
    mentoGovernor.execute(targets, values, calldatas, keccak256(bytes(description)));

    vm.prank(bob);
    mentoGovernor.castVote(proposalId, 1);

    vm.prank(charlie);
    mentoGovernor.castVote(proposalId, 0);

    vm.expectRevert("Governor: proposal not successful");
    mentoGovernor.queue(targets, values, calldatas, keccak256(bytes(description)));

    vm.expectRevert("Governor: proposal not successful");
    mentoGovernor.execute(targets, values, calldatas, keccak256(bytes(description)));

    vm.roll(block.number + votingPeriod);
    skip(7 days);

    vm.expectRevert("Governor: proposal not successful");
    mentoGovernor.queue(targets, values, calldatas, keccak256(bytes(description)));

    vm.expectRevert("Governor: proposal not successful");
    mentoGovernor.execute(targets, values, calldatas, keccak256(bytes(description)));
  }

  function test_execute_whenTimelocked_shouldRevert() public {
    mockVeMento.mint(alice, threshold);
    mockVeMento.mint(bob, 1_000e18);
    mockVeMento.mint(charlie, 2_000e18);

    vm.prank(alice);
    (
      uint256 proposalId,
      address[] memory targets,
      uint256[] memory values,
      bytes[] memory calldatas,
      string memory description
    ) = _proposeCallProtectedFunction();

    // keeping block.ts and block.number in sync
    vm.roll(block.number + votingDelay + 1);
    skip(1 days);

    vm.prank(bob);
    mentoGovernor.castVote(proposalId, 0);

    vm.prank(charlie);
    mentoGovernor.castVote(proposalId, 1);

    vm.roll(block.number + votingPeriod);
    skip(7 days);

    mentoGovernor.queue(targets, values, calldatas, keccak256(bytes(description)));

    vm.expectRevert("TimelockController: operation is not ready");
    mentoGovernor.execute(targets, values, calldatas, keccak256(bytes(description)));
  }

  function test_execute_shouldExecuteProposal_whenTimelockExpires() public {
    mockVeMento.mint(alice, threshold);
    mockVeMento.mint(bob, 1_000e18);
    mockVeMento.mint(charlie, 2_000e18);

    vm.prank(alice);
    (
      uint256 proposalId,
      address[] memory targets,
      uint256[] memory values,
      bytes[] memory calldatas,
      string memory description
    ) = _proposeCallProtectedFunction();

    // keeping block.ts and block.number in sync
    vm.roll(block.number + votingDelay + 1);
    skip(1 days);

    vm.prank(bob);
    mentoGovernor.castVote(proposalId, 0);

    vm.prank(charlie);
    mentoGovernor.castVote(proposalId, 1);

    vm.roll(block.number + votingPeriod);
    skip(7 days);

    mentoGovernor.queue(targets, values, calldatas, keccak256(bytes(description)));

    vm.roll(block.number + BLOCKS_DAY);
    skip(1 days);

    assertEq(mockOwnable.protected(), 0);

    mentoGovernor.execute(targets, values, calldatas, keccak256(bytes(description)));

    assertEq(mockOwnable.protected(), 1337);
  }

  function test_queueAndexecute_whenRetried_shouldRevert() public {
    mockVeMento.mint(alice, threshold);
    mockVeMento.mint(bob, 1_000e18);
    mockVeMento.mint(charlie, 2_000e18);

    vm.prank(alice);
    (
      uint256 proposalId,
      address[] memory targets,
      uint256[] memory values,
      bytes[] memory calldatas,
      string memory description
    ) = _proposeCallProtectedFunction();

    // keeping block.ts and block.number in sync
    vm.roll(block.number + votingDelay + 1);
    skip(1 days);

    vm.prank(bob);
    mentoGovernor.castVote(proposalId, 0);

    vm.prank(charlie);
    mentoGovernor.castVote(proposalId, 1);

    vm.roll(block.number + votingPeriod);
    skip(7 days);

    mentoGovernor.queue(targets, values, calldatas, keccak256(bytes(description)));

    vm.expectRevert("Governor: proposal not successful");
    mentoGovernor.queue(targets, values, calldatas, keccak256(bytes(description)));

    vm.roll(block.number + BLOCKS_DAY);
    skip(1 days);

    assertEq(mockOwnable.protected(), 0);

    mentoGovernor.execute(targets, values, calldatas, keccak256(bytes(description)));

    vm.expectRevert("Governor: proposal not successful");
    mentoGovernor.execute(targets, values, calldatas, keccak256(bytes(description)));

    assertEq(mockOwnable.protected(), 1337);
  }

  function test_cancel_whenCalledByCanceller_shouldBlockQueuedProposal() public {
    mockVeMento.mint(alice, threshold);
    mockVeMento.mint(bob, 1_000e18);
    mockVeMento.mint(charlie, 2_000e18);

    vm.prank(alice);
    (
      uint256 proposalId,
      address[] memory targets,
      uint256[] memory values,
      bytes[] memory calldatas,
      string memory description
    ) = _proposeCallProtectedFunction();

    bytes32 tlId = timelockController.hashOperationBatch(targets, values, calldatas, 0, keccak256(bytes(description)));

    // keeping block.ts and block.number in sync
    vm.roll(block.number + votingDelay + 1);
    skip(1 days);

    vm.prank(bob);
    mentoGovernor.castVote(proposalId, 0);

    vm.prank(charlie);
    mentoGovernor.castVote(proposalId, 1);

    vm.roll(block.number + votingPeriod);
    skip(7 days);

    mentoGovernor.queue(targets, values, calldatas, keccak256(bytes(description)));

    vm.prank(alice);
    vm.expectRevert();
    timelockController.cancel(tlId);

    vm.prank(communityMultisig);
    timelockController.cancel(tlId);

    vm.roll(block.number + BLOCKS_DAY);
    skip(1 days);

    vm.expectRevert("Governor: proposal not successful");
    mentoGovernor.execute(targets, values, calldatas, keccak256(bytes(description)));
  }

  function test_cancel_whenCalledBeforeQueue_shouldRevert() public {
    mockVeMento.mint(alice, threshold);
    mockVeMento.mint(bob, 1_000e18);
    mockVeMento.mint(charlie, 2_000e18);

    vm.prank(alice);
    (
      uint256 proposalId,
      address[] memory targets,
      uint256[] memory values,
      bytes[] memory calldatas,
      string memory description
    ) = _proposeCallProtectedFunction();

    bytes32 tlId = timelockController.hashOperationBatch(targets, values, calldatas, 0, keccak256(bytes(description)));

    vm.prank(communityMultisig);
    vm.expectRevert("TimelockController: operation cannot be cancelled");
    timelockController.cancel(tlId);

    // keeping block.ts and block.number in sync
    vm.roll(block.number + votingDelay + 1);
    skip(1 days);

    vm.prank(communityMultisig);
    vm.expectRevert("TimelockController: operation cannot be cancelled");
    timelockController.cancel(tlId);

    vm.prank(bob);
    mentoGovernor.castVote(proposalId, 0);

    vm.prank(charlie);
    mentoGovernor.castVote(proposalId, 1);

    vm.prank(communityMultisig);
    vm.expectRevert("TimelockController: operation cannot be cancelled");
    timelockController.cancel(tlId);

    vm.roll(block.number + votingPeriod);
    skip(7 days);

    vm.prank(communityMultisig);
    vm.expectRevert("TimelockController: operation cannot be cancelled");
    timelockController.cancel(tlId);
  }

  function _proposeCallProtectedFunction()
    internal
    returns (
      uint256 proposalId,
      address[] memory targets,
      uint256[] memory values,
      bytes[] memory calldatas,
      string memory description
    )
  {
    // A random int that will be set to the protected variable
    uint256 newProtected = 1337;

    targets = new address[](1);
    targets[0] = address(mockOwnable);

    values = new uint256[](1);
    values[0] = 0;

    calldatas = new bytes[](1);
    calldatas[0] = abi.encodeWithSelector(mockOwnable.protectedFunction.selector, newProtected);

    description = "Set protected value to 1337";

    proposalId = mentoGovernor.propose(targets, values, calldatas, description);
  }

  function _hashProposal(
    address[] memory targets,
    uint256[] memory values,
    bytes[] memory calldatas,
    string memory description
  ) private pure returns (uint256) {
    return uint256(keccak256(abi.encode(targets, values, calldatas, keccak256(bytes(description)))));
  }
}
