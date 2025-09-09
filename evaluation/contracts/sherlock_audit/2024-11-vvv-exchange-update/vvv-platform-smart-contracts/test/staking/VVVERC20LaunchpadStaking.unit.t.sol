//SPDX-License-Identifier: MIT
pragma solidity 0.8.23;

import { VVVToken } from "contracts/tokens/VvvToken.sol";
import { VVVAuthorizationRegistry } from "contracts/auth/VVVAuthorizationRegistry.sol";
import { VVVAuthorizationRegistryChecker } from "contracts/auth/VVVAuthorizationRegistryChecker.sol";
import { VVVStakingTestBase } from "test/staking/VVVStakingTestBase.sol";
import { VVVERC20LaunchpadStaking } from "contracts/staking/VVVERC20LaunchpadStaking.sol";

/**
 * @title VVVERC20LaunchpadStaking Unit Tests
 * @dev use "forge test --match-contract VVVERC20LaunchpadStakingUnitTests" to run tests
 * @dev use "forge coverage --match-contract VVVERC20LaunchpadStaking" to run coverage
 */
contract VVVERC20LaunchpadStakingUnitTests is VVVStakingTestBase {
    function setUp() public {
        vm.startPrank(deployer, deployer);
        setDefaultLaunchpadStakingDurations();

        AuthRegistry = new VVVAuthorizationRegistry(defaultAdminTransferDelay, deployer);

        VvvTokenInstance = new VVVToken(type(uint256).max, 0, address(AuthRegistry));

        ERC20LaunchpadStakingInstance = new VVVERC20LaunchpadStaking(
            address(VvvTokenInstance),
            stakingDurations,
            address(AuthRegistry)
        );

        //set auth registry permissions
        AuthRegistry.grantRole(launchpadStakingManagerRole, launchpadStakingManager);
        bytes4 setStakingDurationsSelector = ERC20LaunchpadStakingInstance.setStakingDurations.selector;
        bytes4 setPenaltyNumeratorSelector = ERC20LaunchpadStakingInstance.setPenaltyNumerator.selector;
        AuthRegistry.setPermission(
            address(ERC20LaunchpadStakingInstance),
            setStakingDurationsSelector,
            launchpadStakingManagerRole
        );
        AuthRegistry.setPermission(
            address(ERC20LaunchpadStakingInstance),
            setPenaltyNumeratorSelector,
            launchpadStakingManagerRole
        );

        VvvTokenInstance.mint(deployer, type(uint96).max);
        VvvTokenInstance.mint(sampleUser, type(uint96).max);
        VvvTokenInstance.approve(address(ERC20LaunchpadStakingInstance), type(uint256).max);
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        VvvTokenInstance.approve(address(ERC20LaunchpadStakingInstance), type(uint256).max);
        vm.stopPrank();
    }

    function testDeployment() public {
        assertTrue(address(ERC20LaunchpadStakingInstance) != address(0));
    }

    //tests that a user can stake in a pool with msg.value for duration according to pool index
    function testStake() public {
        uint256 tokenAmountToStake = 1 ether;
        uint256 stakeDurationIndex = 1;

        vm.startPrank(sampleUser, sampleUser);
        //90 day stake based on default duration values
        ERC20LaunchpadStakingInstance.stake(stakeDurationIndex, tokenAmountToStake);
        vm.stopPrank();

        //check that the user stake mapping correctly indicates the above stake
        (, uint256 amountStaked, ) = ERC20LaunchpadStakingInstance.userStakes(
            sampleUser,
            stakeDurationIndex
        );

        assertEq(amountStaked, tokenAmountToStake);
    }

    //tests that a user can add to an existing stake in a given pool
    function testAddToStake() public {
        uint256 tokenAmountToStake = 1 ether;
        uint256 stakeDurationIndex = 1;
        uint256 stakeDurationInSeconds = stakingDurations[stakeDurationIndex];
        uint256 secondsToAdvance = stakeDurationInSeconds / 2;

        vm.startPrank(sampleUser, sampleUser);
        //90 day stake based on default duration values
        ERC20LaunchpadStakingInstance.stake(stakeDurationIndex, tokenAmountToStake);
        (, , uint256 startTimestamp1) = ERC20LaunchpadStakingInstance.userStakes(
            sampleUser,
            stakeDurationIndex
        );

        advanceBlockNumberAndTimestampInSeconds(secondsToAdvance);

        //add to stake of the same pool ID (i.e. duration array index)
        ERC20LaunchpadStakingInstance.stake(stakeDurationIndex, tokenAmountToStake);
        (, uint256 amountStaked2, uint256 startTimestamp2) = ERC20LaunchpadStakingInstance.userStakes(
            sampleUser,
            stakeDurationIndex
        );
        vm.stopPrank();

        assertEq(amountStaked2, tokenAmountToStake * 2);
        assertEq(startTimestamp2, startTimestamp1 + secondsToAdvance - 1);
    }

    //tests that staking in a non-existent pool id causes a revert with the InvalidPoolId error
    function testStakeNonExistentPoolId() public {
        uint256 tokenAmountToStake = 1 ether;
        uint256 nonExistentPoolId = stakingDurations.length + 1;

        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVERC20LaunchpadStaking.InvalidPoolId.selector);
        ERC20LaunchpadStakingInstance.stake(nonExistentPoolId, tokenAmountToStake);
        vm.stopPrank();
    }

    //tests that staking with zero token amount causes revert with ZeroStakeAmount error
    function testStakeZeroTokenAmount() public {
        uint256 tokenAmountToStake = 0;
        uint256 stakeDurationIndex = 1;

        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVERC20LaunchpadStaking.ZeroStakeAmount.selector);
        ERC20LaunchpadStakingInstance.stake(stakeDurationIndex, tokenAmountToStake);
        vm.stopPrank();
    }

    //tests that the Staked event is emitted when a user stakes
    function testStakedEmit() public {
        vm.deal(sampleUser, 1 ether);
        uint256 tokenAmountToStake = 1 ether;
        uint256 stakeDurationIndex = 1;

        vm.startPrank(sampleUser, sampleUser);
        vm.expectEmit(address(ERC20LaunchpadStakingInstance));
        emit VVVERC20LaunchpadStaking.Stake(sampleUser, stakeDurationIndex, tokenAmountToStake);
        ERC20LaunchpadStakingInstance.stake(stakeDurationIndex, tokenAmountToStake);
        vm.stopPrank();
    }

    //tests that a user can unstake at the end of a stake duration
    function testUnstakeFullDuration() public {
        uint256 tokenAmountToStake = 1 ether;
        uint256 stakeDurationIndex = 1;
        uint256 stakeDurationInSeconds = stakingDurations[stakeDurationIndex];
        uint256 sampleUserBalanceBeforeStake = VvvTokenInstance.balanceOf(sampleUser);
        uint256 sampleUserBalanceAfterStake;
        uint256 sampleUserBalanceAfterUnstake;

        vm.startPrank(sampleUser, sampleUser);
        //90 day stake based on default duration values
        ERC20LaunchpadStakingInstance.stake(stakeDurationIndex, tokenAmountToStake);
        sampleUserBalanceAfterStake = VvvTokenInstance.balanceOf(sampleUser);
        advanceBlockNumberAndTimestampInSeconds(stakeDurationInSeconds + 1);
        ERC20LaunchpadStakingInstance.unstake(stakeDurationIndex);
        sampleUserBalanceAfterUnstake = VvvTokenInstance.balanceOf(sampleUser);
        vm.stopPrank();

        assertEq(sampleUserBalanceAfterStake, sampleUserBalanceBeforeStake - tokenAmountToStake);
        assertEq(sampleUserBalanceAfterUnstake, sampleUserBalanceBeforeStake);
    }

    //tests that a user can unstake at 1/2 the duration with a penalty of penaltyNumerator/(PENALTY_DENOMINATOR*2)
    function testUnstakeHalfDuration() public {
        uint256 tokenAmountToStake = 1 ether;
        uint256 stakeDurationIndex = 1;
        uint256 stakeDurationInSeconds = stakingDurations[stakeDurationIndex];
        uint256 stakeDurationDivisor = 2;
        uint256 sampleUserBalanceBeforeStake = VvvTokenInstance.balanceOf(sampleUser);
        uint256 sampleUserBalanceAfterUnstake;
        uint256 expectedUnstakeNumerator = 75;
        uint256 expectedUnstakeDenominator = 100;

        vm.startPrank(sampleUser, sampleUser);
        //90 day stake based on default duration values
        ERC20LaunchpadStakingInstance.stake(stakeDurationIndex, tokenAmountToStake);
        advanceBlockNumberAndTimestampInSeconds((stakeDurationInSeconds / stakeDurationDivisor) + 1);
        ERC20LaunchpadStakingInstance.unstake(stakeDurationIndex);
        sampleUserBalanceAfterUnstake = VvvTokenInstance.balanceOf(sampleUser);
        vm.stopPrank();

        //75% of staked funds should be returned if withdrawing at 50% of the stake duration
        uint256 expectedUnstakeAmount = (tokenAmountToStake * expectedUnstakeNumerator) /
            expectedUnstakeDenominator;
        assertEq(
            sampleUserBalanceAfterUnstake,
            sampleUserBalanceBeforeStake - tokenAmountToStake + expectedUnstakeAmount
        );
        assertEq(VvvTokenInstance.balanceOf(DEAD_ADDRESS), tokenAmountToStake - expectedUnstakeAmount);
    }

    //tests that a user can unstake immediately with a penalty of penaltyNumerator/PENALTY_DENOMINATOR
    function testUnstakeImmediate() public {
        uint256 tokenAmountToStake = 1 ether;
        uint256 stakeDurationIndex = 1;
        uint256 sampleUserBalanceBeforeStake = VvvTokenInstance.balanceOf(sampleUser);
        uint256 sampleUserBalanceAfterUnstake;
        uint256 expectedUnstakeNumerator = 50;
        uint256 expectedUnstakeDenominator = 100;

        vm.startPrank(sampleUser, sampleUser);
        ERC20LaunchpadStakingInstance.stake(stakeDurationIndex, tokenAmountToStake);
        ERC20LaunchpadStakingInstance.unstake(stakeDurationIndex);
        sampleUserBalanceAfterUnstake = VvvTokenInstance.balanceOf(sampleUser);
        vm.stopPrank();

        //50% of staked funds should be returned if withdrawing at 0% of the stake duration
        uint256 expectedUnstakeAmount = (tokenAmountToStake * expectedUnstakeNumerator) /
            expectedUnstakeDenominator;
        assertEq(
            sampleUserBalanceAfterUnstake,
            sampleUserBalanceBeforeStake - tokenAmountToStake + expectedUnstakeAmount
        );
        assertEq(VvvTokenInstance.balanceOf(DEAD_ADDRESS), tokenAmountToStake - expectedUnstakeAmount);
    }

    //tests that the Unstaked event is emitted when a user unstakes
    function testUnstakedEmit() public {
        uint256 tokenAmountToStake = 1 ether;
        uint256 stakeDurationIndex = 1;
        uint256 expectedUnstakeNumerator = 50;
        uint256 expectedUnstakeDenominator = 100;

        vm.startPrank(sampleUser, sampleUser);
        ERC20LaunchpadStakingInstance.stake(stakeDurationIndex, tokenAmountToStake);

        uint256 expectedPenalty = (tokenAmountToStake * expectedUnstakeNumerator) /
            expectedUnstakeDenominator;

        vm.expectEmit(address(ERC20LaunchpadStakingInstance));
        emit VVVERC20LaunchpadStaking.Unstake(
            sampleUser,
            stakeDurationIndex,
            tokenAmountToStake,
            expectedPenalty
        );
        ERC20LaunchpadStakingInstance.unstake(stakeDurationIndex);
        vm.stopPrank();
    }

    //tests that unstaking in a non-existent pool id causes a revert with the InvalidPoolId error
    function testUnstakeNonExistentPoolId() public {
        uint256 nonExistentPoolId = stakingDurations.length + 1;

        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVERC20LaunchpadStaking.InvalidPoolId.selector);
        ERC20LaunchpadStakingInstance.unstake(nonExistentPoolId);
        vm.stopPrank();
    }

    //tests that unstaking for a pool with no staked balance reverts with the NoStakeForPool error
    function testUnstakeNoStakeForPool() public {
        uint256 stakeDurationIndex = 1;

        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVERC20LaunchpadStaking.NoStakeForPool.selector);
        ERC20LaunchpadStakingInstance.unstake(stakeDurationIndex);
        vm.stopPrank();
    }

    //tests that calculatePenalty correctly calculates the penalty for the full stake duration
    function testCalculatePenaltyFullDuration() public {
        uint256 stakeDurationIndex = 1;
        uint256 stakeDuration = stakingDurations[stakeDurationIndex];
        uint256 referencePenalty = 0;

        VVVERC20LaunchpadStaking.StakeData memory stake = VVVERC20LaunchpadStaking.StakeData({
            durationIndex: stakeDurationIndex,
            amount: 1 ether,
            startTimestamp: block.timestamp
        });

        advanceBlockNumberAndTimestampInSeconds(stakeDuration + 1);

        uint256 calculatedPenalty = ERC20LaunchpadStakingInstance.calculatePenalty(stake);

        assertEq(calculatedPenalty, referencePenalty);
    }

    //tests that calculatePenalty correctly calculates the penalty for 1/2 stake duration
    function testCalculatePenaltyHalfDuration() public {
        uint256 stakeDurationIndex = 1;
        uint256 stakeDuration = stakingDurations[stakeDurationIndex];
        uint256 stakeDurationDivisor = 2;
        uint256 referencePenalty = 0.25 ether;

        VVVERC20LaunchpadStaking.StakeData memory stake = VVVERC20LaunchpadStaking.StakeData({
            durationIndex: stakeDurationIndex,
            amount: 1 ether,
            startTimestamp: block.timestamp
        });

        advanceBlockNumberAndTimestampInSeconds((stakeDuration / stakeDurationDivisor) + 1);

        uint256 calculatedPenalty = ERC20LaunchpadStakingInstance.calculatePenalty(stake);

        assertEq(calculatedPenalty, referencePenalty);
    }

    //tests that calculatePenalty correctly calculates the penalty for zero duration
    function testCalculatePenaltyImmediate() public {
        uint256 stakeDurationIndex = 1;
        uint256 referencePenalty = 0.5 ether;

        VVVERC20LaunchpadStaking.StakeData memory stake = VVVERC20LaunchpadStaking.StakeData({
            durationIndex: stakeDurationIndex,
            amount: 1 ether,
            startTimestamp: block.timestamp
        });

        uint256 calculatedPenalty = ERC20LaunchpadStakingInstance.calculatePenalty(stake);

        assertEq(calculatedPenalty, referencePenalty);
    }

    //tests that after staking, the stake array is read via getStakesByAddress
    function testGetStakesByAddress() public {
        uint256 tokenAmountToStake = 1 ether;

        uint256[] memory durationIndices = new uint256[](3);
        durationIndices[0] = 0;
        durationIndices[1] = 1;
        durationIndices[2] = 2;

        uint256[] memory amountsToStake = new uint256[](3);
        amountsToStake[0] = tokenAmountToStake;
        amountsToStake[1] = tokenAmountToStake + 1;
        amountsToStake[2] = tokenAmountToStake + 2;

        //stake in pool ids (stake duration indices) 0-2
        //varying amounts and timestamps slightly to confirm written values
        vm.startPrank(sampleUser, sampleUser);
        advanceBlockNumberAndTimestampInSeconds(1);
        uint256 startTimestamp = block.timestamp;
        for (uint256 i = 0; i < durationIndices.length; ++i) {
            ERC20LaunchpadStakingInstance.stake(durationIndices[i], amountsToStake[i]);
            advanceBlockNumberAndTimestampInSeconds(1);
        }
        vm.stopPrank();

        VVVERC20LaunchpadStaking.StakeData[] memory userStakes = ERC20LaunchpadStakingInstance
            .getStakesByAddress(sampleUser);

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
        ERC20LaunchpadStakingInstance.setStakingDurations(newStakingDurations);
        vm.stopPrank();

        for (uint256 i = 0; i < stakingDurations.length; i++) {
            assertEq(ERC20LaunchpadStakingInstance.stakingDurations(i), newStakingDurations[i]);
        }
    }

    //tests that a non-admin cannot set the array of staking durations
    function testNonAdminCannotSetStakingDurations() public {
        vm.startPrank(deployer, deployer);
        vm.expectRevert(VVVAuthorizationRegistryChecker.UnauthorizedCaller.selector);
        ERC20LaunchpadStakingInstance.setStakingDurations(stakingDurations);
        vm.stopPrank();
    }

    //tests that the StakingDurationsSet event is emitted when the staking durations are set
    function testStakingDurationsSetEvent() public {
        vm.startPrank(launchpadStakingManager, launchpadStakingManager);
        vm.expectEmit(address(ERC20LaunchpadStakingInstance));
        emit VVVERC20LaunchpadStaking.StakingDurationsSet(stakingDurations);
        ERC20LaunchpadStakingInstance.setStakingDurations(stakingDurations);
        vm.stopPrank();
    }

    //tests that the penalty numerator can be set by the admin
    function testAdminSetPenaltyNumerator() public {
        uint256 newPenaltyNumerator = ERC20LaunchpadStakingInstance.PENALTY_DENOMINATOR();

        vm.startPrank(launchpadStakingManager, launchpadStakingManager);
        ERC20LaunchpadStakingInstance.setPenaltyNumerator(newPenaltyNumerator);
        vm.stopPrank();

        assertEq(ERC20LaunchpadStakingInstance.penaltyNumerator(), newPenaltyNumerator);
    }

    //test that a non-admin cannot set the penalty numerator
    function testNonAdminCannotSetPenaltyNumerator() public {
        uint256 newPenaltyNumerator = 12345;
        vm.startPrank(deployer, deployer);
        vm.expectRevert(VVVAuthorizationRegistryChecker.UnauthorizedCaller.selector);
        ERC20LaunchpadStakingInstance.setPenaltyNumerator(newPenaltyNumerator);
        vm.stopPrank();
    }

    //tests that NumeratorCannotExceedDenominator is thrown correctly when attempting to set a numerator higher than the denominator
    function testSetPenaltyNumeratorLargerThanDenominator() public {
        uint256 newPenaltyNumerator = ERC20LaunchpadStakingInstance.PENALTY_DENOMINATOR() + 1;

        vm.startPrank(launchpadStakingManager, launchpadStakingManager);
        vm.expectRevert(VVVERC20LaunchpadStaking.NumeratorCannotExceedDenominator.selector);
        ERC20LaunchpadStakingInstance.setPenaltyNumerator(newPenaltyNumerator);
        vm.stopPrank();
    }

    ///tests that the PenaltyNumeratorSet event is emitted when the penalty numerator is set
    function testEmitPenaltyNumeratorSet() public {
        uint256 newPenaltyNumerator = 123;
        vm.startPrank(launchpadStakingManager, launchpadStakingManager);
        vm.expectEmit(address(ERC20LaunchpadStakingInstance));
        emit VVVERC20LaunchpadStaking.PenaltyNumeratorSet(newPenaltyNumerator);
        ERC20LaunchpadStakingInstance.setPenaltyNumerator(newPenaltyNumerator);
        vm.stopPrank();
    }
}
