//SPDX-License-Identifier: MIT
pragma solidity 0.8.23;

import { NonReceivable } from "test/utils/NonReceivable.sol";
import { VVVToken } from "contracts/tokens/VvvToken.sol";
import { VVVAuthorizationRegistry } from "contracts/auth/VVVAuthorizationRegistry.sol";
import { VVVAuthorizationRegistryChecker } from "contracts/auth/VVVAuthorizationRegistryChecker.sol";
import { VVVStakingTestBase } from "test/staking/VVVStakingTestBase.sol";
import { VVVLaunchpadStaking } from "contracts/staking/VVVLaunchpadStaking.sol";

/**
 * @title VVVLaunchpadStaking Unit Tests
 * @dev use "forge test --match-contract VVVLaunchpadStakingUnitTests" to run tests
 * @dev use "forge coverage --match-contract VVVLaunchpadStaking" to run coverage
 */
contract VVVLaunchpadStakingUnitTests is VVVStakingTestBase {
    function setUp() public {
        vm.startPrank(deployer, deployer);

        //deploy a non-receivable address with 1 ether
        vm.deal(deployer, 1 ether);
        NonReceivableCaller = new NonReceivable{ value: 1 ether }();

        //set default staking durations
        setDefaultLaunchpadStakingDurations();

        AuthRegistry = new VVVAuthorizationRegistry(defaultAdminTransferDelay, deployer);
        LaunchpadStakingInstance = new VVVLaunchpadStaking(stakingDurations, address(AuthRegistry));
        VvvTokenInstance = new VVVToken(type(uint256).max, 0, address(AuthRegistry));

        //set auth registry permissions
        AuthRegistry.grantRole(launchpadStakingManagerRole, launchpadStakingManager);
        bytes4 setStakingDurationsSelector = LaunchpadStakingInstance.setStakingDurations.selector;
        bytes4 setPenaltyNumeratorSelector = LaunchpadStakingInstance.setPenaltyNumerator.selector;
        AuthRegistry.setPermission(
            address(LaunchpadStakingInstance),
            setStakingDurationsSelector,
            launchpadStakingManagerRole
        );
        AuthRegistry.setPermission(
            address(LaunchpadStakingInstance),
            setPenaltyNumeratorSelector,
            launchpadStakingManagerRole
        );

        vm.stopPrank();
    }

    function testDeployment() public {
        assertTrue(address(LaunchpadStakingInstance) != address(0));
    }

    //tests that a user can stake in a pool with msg.value for duration according to pool index
    function testStake() public {
        vm.deal(sampleUser, 1 ether);
        uint256 amountToStake = 1 ether;
        uint256 stakeDurationIndex = 1;

        vm.startPrank(sampleUser, sampleUser);
        //90 day stake based on default duration values
        LaunchpadStakingInstance.stake{ value: amountToStake }(stakeDurationIndex);
        vm.stopPrank();

        //check that the user stake mapping correctly indicates the above stake
        (, uint256 amountStaked, ) = LaunchpadStakingInstance.userStakes(sampleUser, stakeDurationIndex);

        assertEq(amountStaked, amountToStake);
    }

    //tests that a user can add to an existing stake in a given pool
    function testAddToStake() public {
        vm.deal(sampleUser, 2 ether);
        uint256 amountToStake = 1 ether;
        uint256 stakeDurationIndex = 1;
        uint256 stakeDurationInSeconds = stakingDurations[stakeDurationIndex];
        uint256 secondsToAdvance = stakeDurationInSeconds / 2;

        vm.startPrank(sampleUser, sampleUser);
        //90 day stake based on default duration values
        LaunchpadStakingInstance.stake{ value: amountToStake }(stakeDurationIndex);
        (, , uint256 startTimestamp1) = LaunchpadStakingInstance.userStakes(
            sampleUser,
            stakeDurationIndex
        );

        advanceBlockNumberAndTimestampInSeconds(secondsToAdvance);

        //add to stake of the same pool ID (i.e. duration array index)
        LaunchpadStakingInstance.stake{ value: amountToStake }(stakeDurationIndex);
        (, uint256 amountStaked2, uint256 startTimestamp2) = LaunchpadStakingInstance.userStakes(
            sampleUser,
            stakeDurationIndex
        );
        vm.stopPrank();

        assertEq(amountStaked2, amountToStake * 2);
        assertEq(startTimestamp2, startTimestamp1 + secondsToAdvance - 1);
    }

    //tests that staking in a non-existent pool id causes a revert with the InvalidPoolId error
    function testStakeNonExistentPoolId() public {
        vm.deal(sampleUser, 1 ether);
        uint256 amountToStake = 1 ether;
        uint256 nonExistentPoolId = stakingDurations.length + 1;

        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVLaunchpadStaking.InvalidPoolId.selector);
        LaunchpadStakingInstance.stake{ value: amountToStake }(nonExistentPoolId);
        vm.stopPrank();
    }

    //tests that staking with zero msg.value causes revert with ZeroStakeAmount error
    function testStakeZeroMsgValue() public {
        vm.deal(sampleUser, 1 ether);
        uint256 amountToStake = 0;
        uint256 stakeDurationIndex = 1;

        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVLaunchpadStaking.ZeroStakeAmount.selector);
        LaunchpadStakingInstance.stake{ value: amountToStake }(stakeDurationIndex);
        vm.stopPrank();
    }

    //tests that the Staked event is emitted when a user stakes
    function testStakedEmit() public {
        vm.deal(sampleUser, 1 ether);
        uint256 amountToStake = 1 ether;
        uint256 stakeDurationIndex = 1;

        vm.startPrank(sampleUser, sampleUser);
        vm.expectEmit(address(LaunchpadStakingInstance));
        emit VVVLaunchpadStaking.Stake(sampleUser, stakeDurationIndex, amountToStake);
        LaunchpadStakingInstance.stake{ value: amountToStake }(stakeDurationIndex);
        vm.stopPrank();
    }

    //tests that a user can unstake at the end of a stake duration
    function testUnstakeFullDuration() public {
        vm.deal(sampleUser, 1 ether);
        uint256 amountToStake = 1 ether;
        uint256 stakeDurationIndex = 1;
        uint256 stakeDurationInSeconds = stakingDurations[stakeDurationIndex];
        uint256 sampleUserBalanceBeforeStake = sampleUser.balance;
        uint256 sampleUserBalanceAfterStake;
        uint256 sampleUserBalanceAfterUnstake;

        vm.startPrank(sampleUser, sampleUser);
        //90 day stake based on default duration values
        LaunchpadStakingInstance.stake{ value: amountToStake }(stakeDurationIndex);
        sampleUserBalanceAfterStake = sampleUser.balance;
        advanceBlockNumberAndTimestampInSeconds(stakeDurationInSeconds + 1);
        LaunchpadStakingInstance.unstake(stakeDurationIndex);
        sampleUserBalanceAfterUnstake = sampleUser.balance;
        vm.stopPrank();

        assertEq(sampleUserBalanceAfterStake, sampleUserBalanceBeforeStake - amountToStake);
        assertEq(sampleUserBalanceAfterUnstake, sampleUserBalanceBeforeStake);
    }

    //tests that a user can unstake at 1/2 the duration with a penalty of penaltyNumerator/(PENALTY_DENOMINATOR*2)
    function testUnstakeHalfDuration() public {
        vm.deal(sampleUser, 1 ether);
        uint256 amountToStake = 1 ether;
        uint256 stakeDurationIndex = 1;
        uint256 stakeDurationInSeconds = stakingDurations[stakeDurationIndex];
        uint256 stakeDurationDivisor = 2;
        uint256 sampleUserBalanceBeforeStake = sampleUser.balance;
        uint256 sampleUserBalanceAfterUnstake;
        uint256 expectedUnstakeNumerator = 75;
        uint256 expectedUnstakeDenominator = 100;

        vm.startPrank(sampleUser, sampleUser);
        //90 day stake based on default duration values
        LaunchpadStakingInstance.stake{ value: amountToStake }(stakeDurationIndex);
        advanceBlockNumberAndTimestampInSeconds((stakeDurationInSeconds / stakeDurationDivisor) + 1);
        LaunchpadStakingInstance.unstake(stakeDurationIndex);
        sampleUserBalanceAfterUnstake = sampleUser.balance;
        vm.stopPrank();

        //75% of staked funds should be returned if withdrawing at 50% of the stake duration
        uint256 expectedUnstakeAmount = (amountToStake * expectedUnstakeNumerator) /
            expectedUnstakeDenominator;
        assertEq(
            sampleUserBalanceAfterUnstake,
            sampleUserBalanceBeforeStake - amountToStake + expectedUnstakeAmount
        );
        assertEq(address(0).balance, amountToStake - expectedUnstakeAmount);
    }

    //tests that a user can unstake immediately with a penalty of penaltyNumerator/PENALTY_DENOMINATOR
    function testUnstakeImmediate() public {
        vm.deal(sampleUser, 1 ether);
        uint256 amountToStake = 1 ether;
        uint256 stakeDurationIndex = 1;
        uint256 sampleUserBalanceBeforeStake = sampleUser.balance;
        uint256 sampleUserBalanceAfterUnstake;
        uint256 expectedUnstakeNumerator = 50;
        uint256 expectedUnstakeDenominator = 100;

        vm.startPrank(sampleUser, sampleUser);
        LaunchpadStakingInstance.stake{ value: amountToStake }(stakeDurationIndex);
        LaunchpadStakingInstance.unstake(stakeDurationIndex);
        sampleUserBalanceAfterUnstake = sampleUser.balance;
        vm.stopPrank();

        //50% of staked funds should be returned if withdrawing at 0% of the stake duration
        uint256 expectedUnstakeAmount = (amountToStake * expectedUnstakeNumerator) /
            expectedUnstakeDenominator;
        assertEq(
            sampleUserBalanceAfterUnstake,
            sampleUserBalanceBeforeStake - amountToStake + expectedUnstakeAmount
        );
        assertEq(address(0).balance, amountToStake - expectedUnstakeAmount);
    }

    //tests that the Unstaked event is emitted when a user unstakes
    function testUnstakedEmit() public {
        vm.deal(sampleUser, 1 ether);
        uint256 amountToStake = 1 ether;
        uint256 stakeDurationIndex = 1;
        uint256 expectedUnstakeNumerator = 50;
        uint256 expectedUnstakeDenominator = 100;

        vm.startPrank(sampleUser, sampleUser);
        LaunchpadStakingInstance.stake{ value: amountToStake }(stakeDurationIndex);

        uint256 expectedPenalty = (amountToStake * expectedUnstakeNumerator) / expectedUnstakeDenominator;

        vm.expectEmit(address(LaunchpadStakingInstance));
        emit VVVLaunchpadStaking.Unstake(sampleUser, stakeDurationIndex, amountToStake, expectedPenalty);
        LaunchpadStakingInstance.unstake(stakeDurationIndex);
        vm.stopPrank();
    }

    //tests that unstaking in a non-existent pool id causes a revert with the InvalidPoolId error
    function testUnstakeNonExistentPoolId() public {
        vm.deal(sampleUser, 1 ether);
        uint256 nonExistentPoolId = stakingDurations.length + 1;

        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVLaunchpadStaking.InvalidPoolId.selector);
        LaunchpadStakingInstance.unstake(nonExistentPoolId);
        vm.stopPrank();
    }

    //tests that unstaking for a pool with no staked balance reverts with the NoStakeForPool error
    function testUnstakeNoStakeForPool() public {
        vm.deal(sampleUser, 1 ether);
        uint256 stakeDurationIndex = 1;

        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVLaunchpadStaking.NoStakeForPool.selector);
        LaunchpadStakingInstance.unstake(stakeDurationIndex);
        vm.stopPrank();
    }

    //tests that a failed transfer of funds during unstake causes a revert with the FailedTransfer error
    function testUnstakeFailedTransfer() public {
        vm.deal(sampleUser, 1 ether);
        uint256 amountToStake = 1 ether;
        uint256 stakeDurationIndex = 1;
        address nonReceivable = address(NonReceivableCaller);

        vm.startPrank(nonReceivable, nonReceivable);
        LaunchpadStakingInstance.stake{ value: amountToStake }(stakeDurationIndex);
        vm.expectRevert(VVVLaunchpadStaking.TransferFailed.selector);
        LaunchpadStakingInstance.unstake(stakeDurationIndex);
        vm.stopPrank();
    }

    //tests that calculatePenalty correctly calculates the penalty for the full stake duration
    function testCalculatePenaltyFullDuration() public {
        uint256 stakeDurationIndex = 1;
        uint256 stakeDuration = stakingDurations[stakeDurationIndex];
        uint256 referencePenalty = 0;

        VVVLaunchpadStaking.StakeData memory stake = VVVLaunchpadStaking.StakeData({
            durationIndex: stakeDurationIndex,
            amount: 1 ether,
            startTimestamp: block.timestamp
        });

        advanceBlockNumberAndTimestampInSeconds(stakeDuration + 1);

        uint256 calculatedPenalty = LaunchpadStakingInstance.calculatePenalty(stake);

        assertEq(calculatedPenalty, referencePenalty);
    }

    //tests that calculatePenalty correctly calculates the penalty for 1/2 stake duration
    function testCalculatePenaltyHalfDuration() public {
        uint256 stakeDurationIndex = 1;
        uint256 stakeDuration = stakingDurations[stakeDurationIndex];
        uint256 stakeDurationDivisor = 2;
        uint256 referencePenalty = 0.25 ether;

        VVVLaunchpadStaking.StakeData memory stake = VVVLaunchpadStaking.StakeData({
            durationIndex: stakeDurationIndex,
            amount: 1 ether,
            startTimestamp: block.timestamp
        });

        advanceBlockNumberAndTimestampInSeconds((stakeDuration / stakeDurationDivisor) + 1);

        uint256 calculatedPenalty = LaunchpadStakingInstance.calculatePenalty(stake);

        assertEq(calculatedPenalty, referencePenalty);
    }

    //tests that calculatePenalty correctly calculates the penalty for zero duration
    function testCalculatePenaltyImmediate() public {
        uint256 stakeDurationIndex = 1;
        uint256 referencePenalty = 0.5 ether;

        VVVLaunchpadStaking.StakeData memory stake = VVVLaunchpadStaking.StakeData({
            durationIndex: stakeDurationIndex,
            amount: 1 ether,
            startTimestamp: block.timestamp
        });

        uint256 calculatedPenalty = LaunchpadStakingInstance.calculatePenalty(stake);

        assertEq(calculatedPenalty, referencePenalty);
    }

    //tests that after staking, the stake array is read via getStakesByAddress
    function testGetStakesByAddress() public {
        vm.deal(sampleUser, 4 ether);
        uint256 amountToStake = 1 ether;

        uint256[] memory durationIndices = new uint256[](3);
        durationIndices[0] = 0;
        durationIndices[1] = 1;
        durationIndices[2] = 2;

        uint256[] memory amountsToStake = new uint256[](3);
        amountsToStake[0] = amountToStake;
        amountsToStake[1] = amountToStake + 1;
        amountsToStake[2] = amountToStake + 2;

        //stake in pool ids (stake duration indices) 0-2
        //varying amounts and timestamps slightly to confirm written values
        vm.startPrank(sampleUser, sampleUser);
        advanceBlockNumberAndTimestampInSeconds(1);
        uint256 startTimestamp = block.timestamp;
        for (uint256 i = 0; i < durationIndices.length; ++i) {
            LaunchpadStakingInstance.stake{ value: amountsToStake[i] }(durationIndices[i]);
            advanceBlockNumberAndTimestampInSeconds(1);
        }
        vm.stopPrank();

        VVVLaunchpadStaking.StakeData[] memory userStakes = LaunchpadStakingInstance.getStakesByAddress(
            sampleUser
        );

        //assert that the lengths and contents of the stake structs match that assigned above
        assertEq(userStakes.length, 3);
        for (uint256 i = 0; i < userStakes.length; ++i) {
            assertEq(userStakes[i].durationIndex, durationIndices[i]);
            assertEq(userStakes[i].amount, amountsToStake[i]);
            assertEq(userStakes[i].startTimestamp, startTimestamp + i);
        }
    }

    //tests that the admin can set the array of staking durations
    function testAdminSetStakingDurations() public {
        //create new staking durations as all zero since default vaules set in constructor are all nonzero
        uint256[] memory newStakingDurations = new uint256[](5);

        vm.startPrank(launchpadStakingManager, launchpadStakingManager);
        LaunchpadStakingInstance.setStakingDurations(newStakingDurations);
        vm.stopPrank();

        for (uint256 i = 0; i < stakingDurations.length; i++) {
            assertEq(LaunchpadStakingInstance.stakingDurations(i), newStakingDurations[i]);
        }
    }

    //tests that a non-admin cannot set the array of staking durations
    function testNonAdminCannotSetStakingDurations() public {
        vm.startPrank(deployer, deployer);
        vm.expectRevert(VVVAuthorizationRegistryChecker.UnauthorizedCaller.selector);
        LaunchpadStakingInstance.setStakingDurations(stakingDurations);
        vm.stopPrank();
    }

    //tests that the StakingDurationsSet event is emitted when the staking durations are set
    function testStakingDurationsSetEvent() public {
        vm.startPrank(launchpadStakingManager, launchpadStakingManager);
        vm.expectEmit(address(LaunchpadStakingInstance));
        emit VVVLaunchpadStaking.StakingDurationsSet(stakingDurations);
        LaunchpadStakingInstance.setStakingDurations(stakingDurations);
        vm.stopPrank();
    }

    //tests that the penalty numerator can be set by the admin
    function testAdminSetPenaltyNumerator() public {
        uint256 newPenaltyNumerator = LaunchpadStakingInstance.PENALTY_DENOMINATOR();

        vm.startPrank(launchpadStakingManager, launchpadStakingManager);
        LaunchpadStakingInstance.setPenaltyNumerator(newPenaltyNumerator);
        vm.stopPrank();

        assertEq(LaunchpadStakingInstance.penaltyNumerator(), newPenaltyNumerator);
    }

    //test that a non-admin cannot set the penalty numerator
    function testNonAdminCannotSetPenaltyNumerator() public {
        uint256 newPenaltyNumerator = 12345;
        vm.startPrank(deployer, deployer);
        vm.expectRevert(VVVAuthorizationRegistryChecker.UnauthorizedCaller.selector);
        LaunchpadStakingInstance.setPenaltyNumerator(newPenaltyNumerator);
        vm.stopPrank();
    }

    //tests that NumeratorCannotExceedDenominator is thrown correctly when attempting to set a numerator higher than the denominator
    function testSetPenaltyNumeratorLargerThanDenominator() public {
        uint256 newPenaltyNumerator = LaunchpadStakingInstance.PENALTY_DENOMINATOR() + 1;

        vm.startPrank(launchpadStakingManager, launchpadStakingManager);
        vm.expectRevert(VVVLaunchpadStaking.NumeratorCannotExceedDenominator.selector);
        LaunchpadStakingInstance.setPenaltyNumerator(newPenaltyNumerator);
        vm.stopPrank();
    }

    ///tests that the PenaltyNumeratorSet event is emitted when the penalty numerator is set
    function testEmitPenaltyNumeratorSet() public {
        uint256 newPenaltyNumerator = 123;
        vm.startPrank(launchpadStakingManager, launchpadStakingManager);
        vm.expectEmit(address(LaunchpadStakingInstance));
        emit VVVLaunchpadStaking.PenaltyNumeratorSet(newPenaltyNumerator);
        LaunchpadStakingInstance.setPenaltyNumerator(newPenaltyNumerator);
        vm.stopPrank();
    }
}
