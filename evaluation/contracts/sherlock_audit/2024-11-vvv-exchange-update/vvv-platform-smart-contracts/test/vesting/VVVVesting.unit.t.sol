//SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import { MockERC20 } from "contracts/mock/MockERC20.sol";
import { VVVAuthorizationRegistry } from "contracts/auth/VVVAuthorizationRegistry.sol";
import { VVVAuthorizationRegistryChecker } from "contracts/auth/VVVAuthorizationRegistryChecker.sol";
import { VVVVesting } from "contracts/vesting/VVVVesting.sol";
import { VVVVestingTestBase } from "test/vesting/VVVVestingTestBase.sol";

/**
 * @title VVVVesting Unit Tests
 * @dev use "forge test --match-contract VVVVestingUnitTests" to run tests
 * @dev use "forge coverage --match-contract VVVVesting" to run coverage
 */
contract VVVVestingUnitTests is VVVVestingTestBase {
    function setUp() public {
        vm.startPrank(deployer, deployer);

        AuthRegistry = new VVVAuthorizationRegistry(defaultAdminTransferDelay, deployer);

        VVVTokenInstance = new MockERC20(18);
        VVVVestingInstance = new VVVVesting(address(VVVTokenInstance), address(AuthRegistry));
        AuthRegistry.grantRole(vestingManagerRole, vestingManager);
        bytes4 setVestingScheduleSelector = VVVVestingInstance.setVestingSchedule.selector;
        bytes4 batchSetVestingScheduleSelector = VVVVestingInstance.batchSetVestingSchedule.selector;
        bytes4 removeVestingScheduleSelector = VVVVestingInstance.removeVestingSchedule.selector;
        bytes4 setVestedTokenSelector = VVVVestingInstance.setVestedToken.selector;
        bytes4 withdrawVvvSelector = VVVVestingInstance.withdrawVvv.selector;
        AuthRegistry.setPermission(
            address(VVVVestingInstance),
            setVestingScheduleSelector,
            vestingManagerRole
        );
        AuthRegistry.setPermission(
            address(VVVVestingInstance),
            batchSetVestingScheduleSelector,
            vestingManagerRole
        );
        AuthRegistry.setPermission(
            address(VVVVestingInstance),
            removeVestingScheduleSelector,
            vestingManagerRole
        );
        AuthRegistry.setPermission(
            address(VVVVestingInstance),
            setVestedTokenSelector,
            vestingManagerRole
        );
        AuthRegistry.setPermission(address(VVVVestingInstance), withdrawVvvSelector, vestingManagerRole);
        VVVTokenInstance.mint(address(VVVVestingInstance), 1_000_000 * 1e18); //1M tokens

        vm.stopPrank();
    }

    function testDeployment() public {
        assertTrue(address(VVVVestingInstance) != address(0));
    }

    //test deployment with zero address as vvv token address
    function testDeploymentWithZeroAddress() public {
        vm.startPrank(deployer, deployer);
        vm.expectRevert(VVVVesting.InvalidConstructorArguments.selector);
        VVVVestingInstance = new VVVVesting(address(0), address(AuthRegistry));
        vm.stopPrank();
    }

    //test that a new vesting schedule can be set and the correct values are stored/read
    function testSetNewVestingSchedule() public {
        VestingParams memory params = VestingParams({
            vestingScheduleIndex: 0,
            tokensToVestAtStart: 1_000 * 1e18, //1k tokens
            tokensToVestAfterFirstInterval: 100 * 1e18, //100 tokens
            amountWithdrawn: 0,
            scheduleStartTime: uint32(block.timestamp + 60 * 60 * 24 * 2),
            cliffEndTime: uint32(block.timestamp + 60 * 60 * 24 * 365),
            intervalLength: 12,
            maxIntervals: 100,
            growthRateProportion: 0
        });

        setVestingScheduleFromManager(
            sampleUser,
            params.vestingScheduleIndex,
            params.tokensToVestAtStart,
            params.tokensToVestAfterFirstInterval,
            params.amountWithdrawn,
            params.scheduleStartTime,
            params.cliffEndTime,
            params.intervalLength,
            params.maxIntervals,
            params.growthRateProportion
        );

        (
            uint256 _tokensToVestAtStart,
            uint256 _tokensToVestAfterFirstInterval,
            uint256 _intervalLength,
            uint256 _maxIntervals,
            uint256 _tokenAmountWithdrawn,
            uint256 _scheduleStartTime,
            uint256 _cliffEndTime,
            uint256 _growthRateProportion
        ) = VVVVestingInstance.userVestingSchedules(sampleUser, 0);

        assertTrue(_tokensToVestAtStart == params.tokensToVestAtStart);
        assertTrue(_tokensToVestAfterFirstInterval == params.tokensToVestAfterFirstInterval);
        assertTrue(_tokenAmountWithdrawn == 0);
        assertTrue(_scheduleStartTime == params.scheduleStartTime);
        assertTrue(_cliffEndTime == params.cliffEndTime);
        assertTrue(_intervalLength == params.intervalLength);
        assertTrue(_maxIntervals == params.maxIntervals);
        assertTrue(_growthRateProportion == params.growthRateProportion);
    }

    //test that a vesting schedule can be updated and the correct values are stored/read
    function testSetExistingVestingSchedule() public {
        {
            VestingParams memory params = VestingParams({
                vestingScheduleIndex: 0,
                tokensToVestAtStart: 1_000 * 1e18, //1k tokens
                tokensToVestAfterFirstInterval: 100 * 1e18, //100 tokens
                amountWithdrawn: 0,
                scheduleStartTime: uint32(block.timestamp + 60 * 60 * 24 * 2),
                cliffEndTime: uint32(block.timestamp + 60 * 60 * 24 * 365),
                intervalLength: 12,
                maxIntervals: 100,
                growthRateProportion: 0
            });

            setVestingScheduleFromManager(
                sampleUser,
                params.vestingScheduleIndex,
                params.tokensToVestAtStart,
                params.tokensToVestAfterFirstInterval,
                params.amountWithdrawn,
                params.scheduleStartTime,
                params.cliffEndTime,
                params.intervalLength,
                params.maxIntervals,
                params.growthRateProportion
            );

            (
                uint256 _tokensToVestAtStart,
                uint256 _tokensToVestAfterFirstInterval,
                uint256 _intervalLength,
                uint256 _maxIntervals,
                uint256 _tokenAmountWithdrawn,
                uint256 _scheduleStartTime,
                uint256 _cliffEndTime,
                uint256 _growthRateProportion
            ) = VVVVestingInstance.userVestingSchedules(sampleUser, 0);

            assertTrue(_tokensToVestAtStart == params.tokensToVestAtStart);
            assertTrue(_tokensToVestAfterFirstInterval == params.tokensToVestAfterFirstInterval);
            assertTrue(_tokenAmountWithdrawn == 0);
            assertTrue(_scheduleStartTime == params.scheduleStartTime);
            assertTrue(_cliffEndTime == params.cliffEndTime);
            assertTrue(_intervalLength == params.intervalLength);
            assertTrue(_maxIntervals == params.maxIntervals);
            assertTrue(_growthRateProportion == params.growthRateProportion);
        }
        {
            //update part of schedule (tokensToVestAfterStart is now 20k, postCliffDuration is now 3 years)
            VestingParams memory params = VestingParams({
                vestingScheduleIndex: 0,
                tokensToVestAtStart: 2_000 * 1e18, //2k tokens
                tokensToVestAfterFirstInterval: 200 * 1e18, //200 tokens
                amountWithdrawn: 0,
                scheduleStartTime: uint32(block.timestamp + 60 * 60 * 24 * 3), //3 days from now
                cliffEndTime: uint32(block.timestamp + 60 * 60 * 24 * 180), //180 days from scheduleStartTime
                intervalLength: 12,
                maxIntervals: 200,
                growthRateProportion: 0
            });

            setVestingScheduleFromManager(
                sampleUser,
                params.vestingScheduleIndex,
                params.tokensToVestAtStart,
                params.tokensToVestAfterFirstInterval,
                params.amountWithdrawn,
                params.scheduleStartTime,
                params.cliffEndTime,
                params.intervalLength,
                params.maxIntervals,
                params.growthRateProportion
            );

            (
                uint256 _tokensToVestAtStart,
                uint256 _tokensToVestAfterFirstInterval,
                uint256 _intervalLength,
                uint256 _maxIntervals,
                uint256 _tokenAmountWithdrawn,
                uint256 _scheduleStartTime,
                uint256 _cliffEndTime,
                uint256 _growthRateProportion
            ) = VVVVestingInstance.userVestingSchedules(sampleUser, 0);

            assertTrue(_tokensToVestAtStart == params.tokensToVestAtStart);
            assertTrue(_tokensToVestAfterFirstInterval == params.tokensToVestAfterFirstInterval);
            assertTrue(_tokenAmountWithdrawn == 0);
            assertTrue(_scheduleStartTime == params.scheduleStartTime);
            assertTrue(_cliffEndTime == params.cliffEndTime);
            assertTrue(_intervalLength == params.intervalLength);
            assertTrue(_maxIntervals == params.maxIntervals);
            assertTrue(_growthRateProportion == params.growthRateProportion);
        }
    }

    //test setting an invalid vesting schedule index
    function testSetInvalidVestingScheduleIndex() public {
        VestingParams memory params = VestingParams({
            vestingScheduleIndex: 123,
            tokensToVestAtStart: 1_000 * 1e18, //1k tokens
            tokensToVestAfterFirstInterval: 100 * 1e18, //100 tokens
            amountWithdrawn: 0,
            scheduleStartTime: uint32(block.timestamp),
            cliffEndTime: uint32(block.timestamp + 60), //1 minute cliff
            intervalLength: 12,
            maxIntervals: 100,
            growthRateProportion: 0
        });

        vm.startPrank(vestingManager, vestingManager);
        vm.expectRevert(VVVVesting.InvalidScheduleIndex.selector);
        VVVVestingInstance.setVestingSchedule(
            sampleUser,
            params.vestingScheduleIndex,
            params.tokensToVestAtStart,
            params.tokensToVestAfterFirstInterval,
            params.amountWithdrawn,
            params.scheduleStartTime,
            params.cliffEndTime,
            params.intervalLength,
            params.maxIntervals,
            params.growthRateProportion
        );
        vm.stopPrank();
    }

    //test that a vesting schedule can be removed (reset) and the correct values are stored/read
    function testRemoveVestingSchedule() public {
        uint256 vestingScheduleIndex = 0;
        {
            VestingParams memory params = VestingParams({
                vestingScheduleIndex: 0,
                tokensToVestAtStart: 1_000 * 1e18, //1k tokens
                tokensToVestAfterFirstInterval: 100 * 1e18, //100 tokens
                amountWithdrawn: 0,
                scheduleStartTime: uint32(block.timestamp + 60 * 60 * 24 * 2),
                cliffEndTime: uint32(block.timestamp + 60 * 60 * 24 * 365),
                intervalLength: 12,
                maxIntervals: 100,
                growthRateProportion: 0
            });

            setVestingScheduleFromManager(
                sampleUser,
                params.vestingScheduleIndex,
                params.tokensToVestAtStart,
                params.tokensToVestAfterFirstInterval,
                params.amountWithdrawn,
                params.scheduleStartTime,
                params.cliffEndTime,
                params.intervalLength,
                params.maxIntervals,
                params.growthRateProportion
            );

            (
                uint256 _tokensToVestAtStart,
                uint256 _tokensToVestAfterFirstInterval,
                uint256 _intervalLength,
                uint256 _maxIntervals,
                uint256 _tokenAmountWithdrawn,
                uint256 _scheduleStartTime,
                uint256 _cliffEndTime,
                uint256 _growthRateProportion
            ) = VVVVestingInstance.userVestingSchedules(sampleUser, 0);

            assertTrue(_tokensToVestAtStart == params.tokensToVestAtStart);
            assertTrue(_tokensToVestAfterFirstInterval == params.tokensToVestAfterFirstInterval);
            assertTrue(_tokenAmountWithdrawn == 0);
            assertTrue(_scheduleStartTime == params.scheduleStartTime);
            assertTrue(_cliffEndTime == params.cliffEndTime);
            assertTrue(_intervalLength == params.intervalLength);
            assertTrue(_maxIntervals == params.maxIntervals);
            assertTrue(_growthRateProportion == params.growthRateProportion);
        }

        {
            removeVestingScheduleFromManager(sampleUser, vestingScheduleIndex);
            (
                uint256 _tokensToVestAtStart2,
                uint256 _tokensToVestAfterFirstInterval2,
                uint256 _intervalLength2,
                uint256 _maxIntervals2,
                uint256 _tokenAmountWithdrawn2,
                uint256 _scheduleStartTime2,
                uint256 _cliffEndTime2,
                uint256 _growthRateProportion2
            ) = VVVVestingInstance.userVestingSchedules(sampleUser, 0);

            assertTrue(_tokensToVestAtStart2 == 0);
            assertTrue(_tokensToVestAfterFirstInterval2 == 0);
            assertTrue(_tokenAmountWithdrawn2 == 0);
            assertTrue(_scheduleStartTime2 == 0);
            assertTrue(_cliffEndTime2 == 0);
            assertTrue(_intervalLength2 == 0);
            assertTrue(_maxIntervals2 == 0);
            assertTrue(_growthRateProportion2 == 0);
        }
    }

    //test onlyAuthorized functions are not accessible by other callers
    function testRemoveVestingScheduleAsNonAdmin() public {
        //values that would work if caller was authorized
        VestingParams memory params = VestingParams({
            vestingScheduleIndex: 0,
            tokensToVestAtStart: 1_000 * 1e18, //1k tokens
            tokensToVestAfterFirstInterval: 100 * 1e18, //100 tokens
            amountWithdrawn: 0,
            scheduleStartTime: uint32(block.timestamp),
            cliffEndTime: uint32(block.timestamp + 60), //1 minute cliff
            intervalLength: 12,
            maxIntervals: 100,
            growthRateProportion: 0
        });

        setVestingScheduleFromManager(
            sampleUser,
            params.vestingScheduleIndex,
            params.tokensToVestAtStart,
            params.tokensToVestAfterFirstInterval,
            params.amountWithdrawn,
            params.scheduleStartTime,
            params.cliffEndTime,
            params.intervalLength,
            params.maxIntervals,
            params.growthRateProportion
        );

        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert();
        setVestingScheduleFromManager(
            sampleUser,
            params.vestingScheduleIndex,
            params.tokensToVestAtStart,
            params.tokensToVestAfterFirstInterval,
            params.amountWithdrawn,
            params.scheduleStartTime,
            params.cliffEndTime,
            params.intervalLength,
            params.maxIntervals,
            params.growthRateProportion
        );
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert();
        VVVVestingInstance.removeVestingSchedule(sampleUser, params.vestingScheduleIndex);
        vm.stopPrank();
    }

    //test that a user can withdraw the correct amount of tokens from a vesting schedule and the vesting contract state matches the withdrawal
    function testUserWithdrawAndVestedAmount() public {
        uint256 vestingScheduleIndex = 0;
        uint88 tokensToVestAtStart = 1_000 * 1e18; //1k tokens
        uint120 tokensToVestAfterFirstInterval = 100 * 1e18; //100 tokens
        uint128 amountWithdrawn = 0;
        uint32 scheduleStartTime = uint32(block.timestamp + 60 * 60 * 24 * 2); //2 days from now
        uint32 cliffEndTime = scheduleStartTime + 60 * 60 * 24 * 365; //1 year from scheduleStartTime
        uint32 intervalLength = 60 * 60 * 6 * 365; //3 months
        uint16 maxIntervals = 100;
        uint64 growthRateProportion = 0;

        setVestingScheduleFromManager(
            sampleUser,
            vestingScheduleIndex,
            tokensToVestAtStart,
            tokensToVestAfterFirstInterval,
            amountWithdrawn,
            scheduleStartTime,
            cliffEndTime,
            intervalLength,
            maxIntervals,
            growthRateProportion
        );

        //advance partially through the vesting schedule
        advanceBlockNumberAndTimestampInBlocks((maxIntervals * intervalLength) / 12 / 2); //seconds/(seconds per block)/fraction of postCliffDuration

        uint128 vestedAmount = uint128(
            VVVVestingInstance.getVestedAmount(sampleUser, vestingScheduleIndex)
        );
        uint256 vestingContractBalanceBeforeWithdraw = VVVTokenInstance.balanceOf(
            address(VVVVestingInstance)
        );

        withdrawVestedTokensAsUser(sampleUser, vestedAmount, sampleUser, vestingScheduleIndex);

        (, , , , uint256 _amountWithdrawn2, , , ) = VVVVestingInstance.userVestingSchedules(sampleUser, 0);
        assertTrue(_amountWithdrawn2 == vestedAmount);

        uint256 vestingContractBalanceAfterWithdraw = VVVTokenInstance.balanceOf(
            address(VVVVestingInstance)
        );
        assertTrue(
            vestingContractBalanceBeforeWithdraw == vestedAmount + vestingContractBalanceAfterWithdraw
        );
    }

    // tests the case where the contract vests more tokens than the contract token balance
    function testWithdrawMoreThanPermitted() public {
        //one more than total contract balance, relies on order of error checking in withdrawVestedTokens()
        uint256 contractBalance = VVVTokenInstance.balanceOf(address(VVVVestingInstance));

        uint256 vestingScheduleIndex = 0;
        uint88 tokensToVestAtStart = 1_000 * 1e18; //1k tokens
        uint120 tokensToVestAfterFirstInterval = uint120(contractBalance * 2);
        uint128 amountWithdrawn = 0;
        uint32 scheduleStartTime = uint32(block.timestamp + 60 * 60 * 24 * 2); //2 days from now
        uint32 cliffEndTime = scheduleStartTime + 60 * 60 * 24 * 365; //1 year from scheduleStartTime
        uint32 intervalLength = 60 * 60 * 6 * 365; //3 months
        uint16 maxIntervals = 100;
        uint64 growthRateProportion = 0;

        setVestingScheduleFromManager(
            sampleUser,
            vestingScheduleIndex,
            tokensToVestAtStart,
            tokensToVestAfterFirstInterval,
            amountWithdrawn,
            scheduleStartTime,
            cliffEndTime,
            intervalLength,
            maxIntervals,
            growthRateProportion
        );
        advanceBlockNumberAndTimestampInBlocks(maxIntervals * intervalLength); //seconds/(seconds per block) - be sure to be past 100% vesting

        uint128 vestedAmount = uint128(
            VVVVestingInstance.getVestedAmount(sampleUser, vestingScheduleIndex)
        );

        //prank to incorporate expected revert message
        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVVesting.AmountIsGreaterThanWithdrawable.selector);
        VVVVestingInstance.withdrawVestedTokens(vestedAmount + 1, sampleUser, vestingScheduleIndex);
        vm.stopPrank();
    }

    //test withdrawVestedTokens() with invalid vesting schedule index - at this point there are no schedules, so any index should fail
    function testWithdrawVestedTokensWithInvalidVestingScheduleIndex() public {
        uint256 vestingScheduleIndex = 0;
        uint128 amountToWithdraw = 0;

        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVVesting.InvalidScheduleIndex.selector);
        VVVVestingInstance.withdrawVestedTokens(amountToWithdraw, sampleUser, vestingScheduleIndex);
        vm.stopPrank();
    }

    //test withdrawVestedTokens() with a finished vesting schedule
    function testWithdrawVestedTokensWithFinishedVestingSchedule() public {
        uint256 vestingScheduleIndex = 0;
        uint88 tokensToVestAtStart = 1_000 * 1e18; //1k tokens
        uint120 tokensToVestAfterFirstInterval = 100 * 1e18; //100 tokens
        uint128 amountWithdrawn = 0;
        uint32 scheduleStartTime = uint32(block.timestamp + 60 * 60 * 24 * 2); //2 days from now
        uint32 cliffEndTime = scheduleStartTime + 60 * 60 * 24 * 365; //1 year from scheduleStartTime
        uint32 intervalLength = 60 * 60 * 6 * 365; //3 months
        uint16 maxIntervals = 100;
        uint64 growthRateProportion = 0;

        setVestingScheduleFromManager(
            sampleUser,
            vestingScheduleIndex,
            tokensToVestAtStart,
            tokensToVestAfterFirstInterval,
            amountWithdrawn,
            scheduleStartTime,
            cliffEndTime,
            intervalLength,
            maxIntervals,
            growthRateProportion
        );
        advanceBlockNumberAndTimestampInBlocks(maxIntervals * intervalLength * uint256(10)); //seconds/(seconds per block) - be sure to be past 100% vesting

        //withdraw all vested tokens after schedule is finished
        uint128 vestedAmount = uint128(
            VVVVestingInstance.getVestedAmount(sampleUser, vestingScheduleIndex)
        );
        withdrawVestedTokensAsUser(sampleUser, vestedAmount, sampleUser, vestingScheduleIndex);

        //attempt to withdraw one more token, should fail
        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVVesting.AmountIsGreaterThanWithdrawable.selector);
        VVVVestingInstance.withdrawVestedTokens(1, sampleUser, vestingScheduleIndex);
        vm.stopPrank();
    }

    //test withdrawVestedTokens() with a vesting schedule that has not yet started
    function testWithdrawVestedTokensWithVestingScheduleNotStarted() public {
        uint256 vestingScheduleIndex = 0;
        uint88 tokensToVestAtStart = 1_000 * 1e18; //1k tokens
        uint120 tokensToVestAfterFirstInterval = 100 * 1e18; //100 tokens
        uint128 amountWithdrawn = 0;
        uint32 scheduleStartTime = uint32(block.timestamp + 60 * 60 * 24 * 2); //2 days from now
        uint32 cliffEndTime = scheduleStartTime + 60 * 60 * 24 * 365; //1 year from scheduleStartTime
        uint32 intervalLength = 60 * 60 * 6 * 365; //3 months
        uint16 maxIntervals = 100;
        uint64 growthRateProportion = 0;

        setVestingScheduleFromManager(
            sampleUser,
            vestingScheduleIndex,
            tokensToVestAtStart,
            tokensToVestAfterFirstInterval,
            amountWithdrawn,
            scheduleStartTime,
            cliffEndTime,
            intervalLength,
            maxIntervals,
            growthRateProportion
        );

        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVVesting.AmountIsGreaterThanWithdrawable.selector);
        VVVVestingInstance.withdrawVestedTokens(
            (maxIntervals * intervalLength + tokensToVestAtStart),
            sampleUser,
            vestingScheduleIndex
        );
        vm.stopPrank();
    }

    //tests that an admin can set the vested token address, and than a non-admin cannot do so
    function testAdminCanSetVestedToken() public {
        address newVestedTokenAddress = 0x1234567890123456789012345678901234567890;

        // Should work
        vm.startPrank(vestingManager, vestingManager);
        VVVVestingInstance.setVestedToken(newVestedTokenAddress);
        vm.stopPrank();

        assertTrue(address(VVVVestingInstance.VVVToken()) == newVestedTokenAddress);
    }

    //test that the zero address cannot be set as the vested token
    function testZeroAddressCannotBeSetAsVestedToken() public {
        address zeroAddress = address(0);

        vm.startPrank(vestingManager, vestingManager);
        vm.expectRevert(VVVVesting.InvalidTokenAddress.selector);
        VVVVestingInstance.setVestedToken(zeroAddress);
        vm.stopPrank();
    }

    //test that a user cannot set the vested token
    function testUserCannotSetVestedToken() public {
        address newVestedTokenAddress = 0x1234567890123456789012345678901234567890;

        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVAuthorizationRegistryChecker.UnauthorizedCaller.selector);
        VVVVestingInstance.setVestedToken(newVestedTokenAddress);
        vm.stopPrank();
    }

    //tests that an admin can withdraw vvv token using 'withdrawVvv'
    function testAdminWithdrawVvv() public {
        uint256 contractBalanceBeforeWithdraw = VVVTokenInstance.balanceOf(address(VVVVestingInstance));

        vm.startPrank(vestingManager, vestingManager);
        VVVVestingInstance.withdrawVvv(contractBalanceBeforeWithdraw);
        vm.stopPrank();

        assertTrue(VVVTokenInstance.balanceOf(vestingManager) == contractBalanceBeforeWithdraw);
        assertTrue(VVVTokenInstance.balanceOf(address(VVVVestingInstance)) == 0);
    }

    //tests that a non-admin cannot withdraw vvv token using 'withdrawVvv'
    function testNonAdminRevertWithdrawVvv() public {
        uint256 contractBalanceBeforeWithdraw = VVVTokenInstance.balanceOf(address(VVVVestingInstance));

        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVAuthorizationRegistryChecker.UnauthorizedCaller.selector);
        VVVVestingInstance.withdrawVvv(contractBalanceBeforeWithdraw);
        vm.stopPrank();
    }

    //test batch-setting vesting schedules as admin with a varying vestedAddress in each vesting schedule
    function testBatchSetVestingSchedulesVaryingVestedAddress() public {
        //sample data
        uint256 numberOfVestedUsers = 2;
        uint256 growthRate = 0;
        string memory paramToVary = "vestedUser";
        VVVVesting.SetVestingScheduleParams[]
            memory setVestingScheduleParams = generateSetVestingScheduleData(
                numberOfVestedUsers,
                growthRate,
                paramToVary
            );

        //set a vesting schedule as admin
        vm.startPrank(vestingManager, vestingManager);
        VVVVestingInstance.batchSetVestingSchedule(setVestingScheduleParams);
        vm.stopPrank();

        //ensure schedules properly set
        for (uint256 i = 0; i < numberOfVestedUsers; i++) {
            (
                uint256 _tokensToVestAtStart,
                uint256 _tokensToVestAfterFirstInterval,
                uint256 _intervalLength,
                uint256 _maxIntervals,
                uint256 _tokenAmountWithdrawn,
                uint256 _scheduleStartTime,
                uint256 _cliffEndTime,
                uint256 _growthRateProportion
            ) = VVVVestingInstance.userVestingSchedules(setVestingScheduleParams[i].vestedUser, 0);

            assertTrue(
                _tokensToVestAtStart == setVestingScheduleParams[i].vestingSchedule.tokensToVestAtStart
            );
            assertTrue(
                _tokensToVestAfterFirstInterval ==
                    setVestingScheduleParams[i].vestingSchedule.tokensToVestAfterFirstInterval
            );

            assertTrue(_tokenAmountWithdrawn == 0);
            assertTrue(
                _scheduleStartTime == setVestingScheduleParams[i].vestingSchedule.scheduleStartTime
            );
            assertTrue(_cliffEndTime == setVestingScheduleParams[i].vestingSchedule.cliffEndTime);
            assertTrue(_intervalLength == setVestingScheduleParams[i].vestingSchedule.intervalLength);
            assertTrue(_maxIntervals == setVestingScheduleParams[i].vestingSchedule.maxIntervals);
            assertTrue(
                _growthRateProportion == setVestingScheduleParams[i].vestingSchedule.growthRateProportion
            );
        }
    }

    //test batch-setting vesting schedules as admin with a varying scheduleIndex in each vesting schedule
    function testBatchSetVestingSchedulesVaryingScheduleIndex() public {
        //sample data
        uint256 numberOfVestedUsers = 2;
        uint256 growthRate = 0;
        string memory paramToVary = "vestingScheduleIndex";
        VVVVesting.SetVestingScheduleParams[]
            memory setVestingScheduleParams = generateSetVestingScheduleData(
                numberOfVestedUsers,
                growthRate,
                paramToVary
            );

        //set a vesting schedule as admin
        vm.startPrank(vestingManager, vestingManager);
        VVVVestingInstance.batchSetVestingSchedule(setVestingScheduleParams);
        vm.stopPrank();

        //ensure schedules properly set
        for (uint256 i = 0; i < numberOfVestedUsers; i++) {
            (
                uint256 _tokensToVestAtStart,
                uint256 _tokensToVestAfterFirstInterval,
                uint256 _intervalLength,
                uint256 _maxIntervals,
                uint256 _tokenAmountWithdrawn,
                uint256 _scheduleStartTime,
                uint256 _cliffEndTime,
                uint256 _growthRateProportion
            ) = VVVVestingInstance.userVestingSchedules(
                    setVestingScheduleParams[i].vestedUser,
                    setVestingScheduleParams[i].vestingScheduleIndex
                );

            assertTrue(
                _tokensToVestAtStart == setVestingScheduleParams[i].vestingSchedule.tokensToVestAtStart
            );
            assertTrue(
                _tokensToVestAfterFirstInterval ==
                    setVestingScheduleParams[i].vestingSchedule.tokensToVestAfterFirstInterval
            );
            assertTrue(_tokenAmountWithdrawn == 0);
            assertTrue(
                _scheduleStartTime == setVestingScheduleParams[i].vestingSchedule.scheduleStartTime
            );
            assertTrue(_cliffEndTime == setVestingScheduleParams[i].vestingSchedule.cliffEndTime);
            assertTrue(_intervalLength == setVestingScheduleParams[i].vestingSchedule.intervalLength);
            assertTrue(_maxIntervals == setVestingScheduleParams[i].vestingSchedule.maxIntervals);
            assertTrue(
                _growthRateProportion == setVestingScheduleParams[i].vestingSchedule.growthRateProportion
            );
        }
    }

    //test that a non-admin cannot batch-set vesting schedules
    function testBatchSetVestingSchedulesUnauthorizedUser() public {
        //sample data
        uint256 numberOfVestedUsers = 1;
        uint256 growthRate = 0;
        string memory paramToVary = "vestedUser";
        VVVVesting.SetVestingScheduleParams[]
            memory setVestingScheduleParams = generateSetVestingScheduleData(
                numberOfVestedUsers,
                growthRate,
                paramToVary
            );

        //set as user (not allowed)
        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVAuthorizationRegistryChecker.UnauthorizedCaller.selector);
        VVVVestingInstance.batchSetVestingSchedule(setVestingScheduleParams);
        vm.stopPrank();
    }

    //test for remainder from division truncation, and make sure it is withdrawable after vesting schedule is finished. choosing prime amounts to make sure it'd work with any vesting schedule
    function testRemainderFromDivisionTruncationIsWithdrawable() public {
        uint256 vestingScheduleIndex = 0;
        uint88 tokensToVestAtStart = 1_000 * 1e18; //1k tokens
        uint120 tokensToVestAfterFirstInterval = 373737 * 1e16; //3737.37 tokens
        uint128 amountWithdrawn = 0;
        uint32 scheduleStartTime = uint32(block.timestamp + 4159); //4159 seconds from now
        uint32 cliffEndTime = scheduleStartTime + 60; //1 minute from scheduleStartTime
        uint32 intervalLength = 397; //397 seconds
        uint16 maxIntervals = 100;
        uint64 growthRateProportion = 0;

        uint256 numberOfIntervalsToAdvanceTimestamp = 95;

        setVestingScheduleFromManager(
            sampleUser,
            vestingScheduleIndex,
            tokensToVestAtStart,
            tokensToVestAfterFirstInterval,
            amountWithdrawn,
            scheduleStartTime,
            cliffEndTime,
            intervalLength,
            maxIntervals,
            growthRateProportion
        );

        advanceBlockNumberAndTimestampInSeconds(
            intervalLength * numberOfIntervalsToAdvanceTimestamp + cliffEndTime
        );

        uint256 vestedAmount = VVVVestingInstance.getVestedAmount(sampleUser, vestingScheduleIndex);

        assertTrue(
            vestedAmount ==
                tokensToVestAtStart +
                    VVVVestingInstance.calculateVestedAmountAtInterval(
                        tokensToVestAfterFirstInterval,
                        (95 * maxIntervals) / 100,
                        growthRateProportion
                    )
        );

        advanceBlockNumberAndTimestampInSeconds(
            intervalLength * (maxIntervals - numberOfIntervalsToAdvanceTimestamp)
        );

        uint256 vestedAmount2 = VVVVestingInstance.getVestedAmount(sampleUser, vestingScheduleIndex);
        assertTrue(
            vestedAmount2 ==
                VVVVestingInstance.calculateVestedAmountAtInterval(
                    tokensToVestAfterFirstInterval,
                    maxIntervals,
                    growthRateProportion
                ) +
                    tokensToVestAtStart
        );
    }

    //tests that the tokensToVestAtStart are available to withdraw immediately at start of vesting schedule
    function testTokensToVestAtStartAreClaimableAtVestingScheduleStart() public {
        uint256 vestingScheduleIndex = 0;
        uint88 tokensToVestAtStart = 1_000 * 1e18; //1k tokens
        uint120 tokensToVestAfterFirstInterval = 100 * 1e18; //100 tokens
        uint128 amountWithdrawn = 0;
        uint32 scheduleStartTime = uint32(block.timestamp + 60 * 60 * 24 * 2); //2 days from now
        uint32 cliffEndTime = scheduleStartTime + 60 * 60 * 24 * 365; //1 year from scheduleStartTime
        uint32 intervalLength = 60 * 60 * 6 * 365; //3 months
        uint16 maxIntervals = 100;
        uint64 growthRateProportion = 0;

        setVestingScheduleFromManager(
            sampleUser,
            vestingScheduleIndex,
            tokensToVestAtStart,
            tokensToVestAfterFirstInterval,
            amountWithdrawn,
            scheduleStartTime,
            cliffEndTime,
            intervalLength,
            maxIntervals,
            growthRateProportion
        );

        //advance to start of vesting schedule
        advanceBlockNumberAndTimestampInSeconds(scheduleStartTime);

        //read vested amount from contract
        uint256 vestedAmount = VVVVestingInstance.getVestedAmount(sampleUser, vestingScheduleIndex);

        //assert that vested amount is equal to tokensToVestAtStart
        assertTrue(vestedAmount == tokensToVestAtStart);
    }

    //test that the tokensToVestAtStart are available until cliffEndTime has elapsed
    function testTokensToVestAtStartAreClaimableAtCliffEndTime() public {
        uint256 vestingScheduleIndex = 0;
        uint88 tokensToVestAtStart = 1_000 * 1e18; //1k tokens
        uint120 tokensToVestAfterFirstInterval = 100 * 1e18; //100 tokens
        uint128 amountWithdrawn = 0;
        uint32 scheduleStartTime = uint32(block.timestamp + 60 * 60 * 24 * 2); //2 days from now
        uint32 cliffEndTime = scheduleStartTime + 60 * 60 * 24 * 365; //1 year from scheduleStartTime
        uint32 intervalLength = 60 * 60 * 6 * 365; //3 months
        uint16 maxIntervals = 100;
        uint64 growthRateProportion = 0;

        setVestingScheduleFromManager(
            sampleUser,
            vestingScheduleIndex,
            tokensToVestAtStart,
            tokensToVestAfterFirstInterval,
            amountWithdrawn,
            scheduleStartTime,
            cliffEndTime,
            intervalLength,
            maxIntervals,
            growthRateProportion
        );

        //advance to end of cliff
        advanceBlockNumberAndTimestampInSeconds(cliffEndTime);

        //read vested amount from contract
        uint256 vestedAmount = VVVVestingInstance.getVestedAmount(sampleUser, vestingScheduleIndex);

        //assert that vested amount is equal to tokensToVestAtStart
        assertTrue(vestedAmount == tokensToVestAtStart);
    }

    //test that any amount greater than tokensToVestAtStart is NOT available before cliffEndTime has elapsed
    function testClaimMoreThanTokensToVestAtStartBeforeCliffEndTime() public {
        uint256 vestingScheduleIndex = 0;
        uint88 tokensToVestAtStart = 1_000 * 1e18; //1k tokens
        uint120 tokensToVestAfterFirstInterval = 100 * 1e18; //100 tokens
        uint128 amountToWithdraw = tokensToVestAtStart + 1; //1 more than tokensToVestAtStart
        uint128 amountWithdrawn = 0;
        uint32 scheduleStartTime = uint32(block.timestamp + 60 * 60 * 24 * 2); //2 days from now
        uint32 cliffEndTime = scheduleStartTime + 60 * 60 * 24 * 365; //1 year from scheduleStartTime
        uint32 intervalLength = 60 * 60 * 6 * 365; //3 months
        uint16 maxIntervals = 100;
        uint64 growthRateProportion = 0;

        setVestingScheduleFromManager(
            sampleUser,
            vestingScheduleIndex,
            tokensToVestAtStart,
            tokensToVestAfterFirstInterval,
            amountWithdrawn,
            scheduleStartTime,
            cliffEndTime,
            intervalLength,
            maxIntervals,
            growthRateProportion
        );

        //advance to end of cliff
        advanceBlockNumberAndTimestampInSeconds(cliffEndTime);

        //attempt to withdraw more than tokensToVestAtStart
        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVVesting.AmountIsGreaterThanWithdrawable.selector);
        VVVVestingInstance.withdrawVestedTokens(amountToWithdraw, sampleUser, vestingScheduleIndex);
        vm.stopPrank();
    }

    //tests exponential vesting does not lose precision with large numbers
    function testExponentialVestingPrecisionIntegerLimits() public {
        uint256 tokensToVestAfterFirstInterval = 1_000_000_000 ether; //1 billion tokens
        uint256 numIntervals = 250;
        uint256 growthRateProportion = 34e16; //34%

        // for 1B, 250, 5800 ==> 79587295150251613031275354740710165573553129976227557061834411350789655172413
        uint256 vestedTokens = VVVVestingInstance.calculateVestedAmountAtInterval(
            tokensToVestAfterFirstInterval,
            numIntervals,
            growthRateProportion
        );

        /**
            Matlab symbolic equation yields giant fraction, so 
            based on https://www.calculator.net/big-number-calculator.html, answer is:
            175679333108460104940322512184131286963401199057761896914806.9785378930203121467,
            so nearest truncated integer is 175679333108460104940322512184131286963401199057761896914806
         */
        uint256 decimalTruncAmount = 175679333108460104940322512184131286963401199057761896914806;

        //arbitrary 0.000000000001% tolerance
        uint256 tolerance = decimalTruncAmount / 1e12;

        /**
            in this case, the error is -0.00000000000000000000064072933749166618026652451957 or about -6e-32
         */
        uint256 difference = decimalTruncAmount > vestedTokens
            ? decimalTruncAmount - vestedTokens
            : vestedTokens - decimalTruncAmount;

        assertTrue(difference <= tolerance);
    }

    /**
        tests exponential vesting is within error tolerance of reference floating point calculation for pre-seed round case:
        tokensToVestAfterFirstInterval = 5,725.241018
        growthRate = 0.005432994453
        maxIntervals = 540
        intervalLength = day
     */
    function testExponentialVestingPreSeed() public {
        uint256 tokensToVestAfterFirstInterval = 5725.24018 ether; //5725.24018 tokens
        uint256 numIntervals = 540;
        uint256 growthRateProportion = 5432994453e6; //0.005432994453 = 5432994453e6/1e18

        uint256 vestedTokens = VVVVestingInstance.calculateVestedAmountAtInterval(
            tokensToVestAfterFirstInterval,
            numIntervals,
            growthRateProportion
        );

        /**
            Matlab symbolic equation yields 18599997275290930836317870.473398
            nearest integer is 18599997275290930836317870
         */
        uint256 decimalTruncAmount = 18599997275290930836317870;

        //arbitrary 0.000000000001% tolerance
        uint256 tolerance = decimalTruncAmount / 1e12;

        //in this case decimalTruncAmount > vestedTokens
        uint256 difference = decimalTruncAmount > vestedTokens
            ? decimalTruncAmount - vestedTokens
            : vestedTokens - decimalTruncAmount;

        assertTrue(difference <= tolerance);
    }

    /**
        tests exponential vesting is within error tolerance of reference floating point calculation for seed round case of:
        tokensToVestAfterFirstInterval = 6234.604511
        growthRate = 0.007949174554
        maxIntervals = 420
        intervalLength = day
     */
    function testExponentialVestingSeed() public {
        uint256 tokensToVestAfterFirstInterval = 6234.604511 ether; //6,234.604511 tokens
        uint256 numIntervals = 420; //nice
        uint256 growthRateProportion = 7949174554e6; //0.007949174554 = 7949174554e6/1e18

        uint256 vestedTokens = VVVVestingInstance.calculateVestedAmountAtInterval(
            tokensToVestAfterFirstInterval,
            numIntervals,
            growthRateProportion
        );

        /**
            Matlab symbolic equation yields 21028569000593437859369453.658997
            nearest integer is 21028569000593437859369453
         */
        uint256 decimalTruncAmount = 21028569000593437859369453;

        //arbitrary 0.000000000001% tolerance
        uint256 tolerance = decimalTruncAmount / 1e12;

        uint256 difference = decimalTruncAmount > vestedTokens
            ? decimalTruncAmount - vestedTokens
            : vestedTokens - decimalTruncAmount;
        assertTrue(difference <= tolerance);
    }

    /**
        tests exponential vesting is within error tolerance of reference floating point calculation for future round case of:
        tokensToVestAfterFirstInterval = 200,000.00
        growthRate = 0.17650557680462953
        maxIntervals = 18
        intervalLength = month
     */
    function testExponentialVestingFuture() public {
        uint256 tokensToVestAfterFirstInterval = 200000.00 ether; //200k tokens
        uint256 numIntervals = 18;
        uint256 growthRateProportion = 176505576804629530; //0.17650557680462953 = 176505576804629530/1e18

        uint256 vestedTokens = VVVVestingInstance.calculateVestedAmountAtInterval(
            tokensToVestAfterFirstInterval,
            numIntervals,
            growthRateProportion
        );

        /**
            Matlab symbolic equation yields 20000000000636743647859458.098462
            nearest integer is 20000000000636743647859458
         */
        uint256 decimalTruncAmount = 20000000000636743647859458;

        //arbitrary 0.000000000001% tolerance
        uint256 tolerance = decimalTruncAmount / 1e12;

        uint256 difference = decimalTruncAmount > vestedTokens
            ? decimalTruncAmount - vestedTokens
            : vestedTokens - decimalTruncAmount;

        assertTrue(difference <= tolerance);
    }

    // tests that a single SetVestingSchedule event is emitted with correct parameters when withdrawVestedTokens is called
    function testEmitSetVestingScheduleOnSingleSet() public {
        uint256 vestingScheduleIndex = 0;
        uint88 tokensToVestAtStart = 1_000 * 1e18; //1k tokens
        uint120 tokensToVestAfterFirstInterval = 100 * 1e18; //100 tokens
        uint128 amountWithdrawn = 0;
        uint32 scheduleStartTime = uint32(block.timestamp + 60 * 60 * 24 * 2); //2 days from now
        uint32 cliffEndTime = scheduleStartTime + 60 * 60 * 24 * 365; //1 year from scheduleStartTime
        uint32 intervalLength = 60 * 60 * 6 * 365; //3 months
        uint16 maxIntervals = 100;
        uint64 growthRateProportion = 0;

        setVestingScheduleFromManager(
            sampleUser,
            vestingScheduleIndex,
            tokensToVestAtStart,
            tokensToVestAfterFirstInterval,
            amountWithdrawn,
            scheduleStartTime,
            cliffEndTime,
            intervalLength,
            maxIntervals,
            growthRateProportion
        );

        //advance partially through the vesting schedule
        advanceBlockNumberAndTimestampInBlocks((maxIntervals * intervalLength) / 12 / 2); //seconds/(seconds per block)/fraction of postCliffDuration

        uint128 vestedAmount = uint128(
            VVVVestingInstance.getVestedAmount(sampleUser, vestingScheduleIndex)
        );

        vm.startPrank(sampleUser, sampleUser);

        vm.expectEmit(address(VVVVestingInstance));
        emit VVVVesting.SetVestingSchedule(
            sampleUser,
            vestingScheduleIndex,
            tokensToVestAtStart,
            tokensToVestAfterFirstInterval,
            vestedAmount,
            scheduleStartTime,
            cliffEndTime,
            intervalLength,
            maxIntervals,
            growthRateProportion
        );

        VVVVestingInstance.withdrawVestedTokens(vestedAmount, sampleUser, vestingScheduleIndex);
        vm.stopPrank();
    }

    // tests that SetVestingSchedule event is emitted with correct parameters when batchSetVestingSchedule is called
    function testEmitSetVestingScheduleOnBatchSet() public {
        uint88 tokensToVestAtStart = 1_000 * 1e18; //1k tokens
        uint120 tokensToVestAfterFirstInterval = 100 * 1e18; //100 tokens
        uint128 amountWithdrawn = 0;
        uint32 scheduleStartTime = uint32(block.timestamp + 60 * 60 * 24 * 2); //2 days from now
        uint32 cliffEndTime = scheduleStartTime + 60 * 60 * 24 * 365; //1 year from scheduleStartTime
        uint32 intervalLength = 60 * 60 * 6 * 365; //3 months
        uint16 maxIntervals = 100;
        uint64 growthRateProportion = 0;

        uint256 numVestingSchedulesToSet = 11;

        //batch set vesting schedules
        VVVVesting.SetVestingScheduleParams[]
            memory setVestingScheduleParams = new VVVVesting.SetVestingScheduleParams[](
                numVestingSchedulesToSet
            );

        for (uint256 i = 0; i < numVestingSchedulesToSet; i++) {
            setVestingScheduleParams[i] = VVVVesting.SetVestingScheduleParams({
                vestedUser: sampleUser,
                vestingScheduleIndex: i,
                vestingSchedule: VVVVesting.VestingSchedule({
                    tokensToVestAtStart: uint88(tokensToVestAtStart),
                    tokensToVestAfterFirstInterval: uint120(tokensToVestAfterFirstInterval),
                    tokenAmountWithdrawn: uint128(amountWithdrawn),
                    scheduleStartTime: uint32(scheduleStartTime),
                    cliffEndTime: uint32(cliffEndTime),
                    intervalLength: uint32(intervalLength),
                    maxIntervals: uint16(maxIntervals),
                    growthRateProportion: uint16(growthRateProportion)
                })
            });
        }

        vm.startPrank(vestingManager, vestingManager);
        for (uint256 i = 0; i < numVestingSchedulesToSet; i++) {
            vm.expectEmit(address(VVVVestingInstance));
            emit VVVVesting.SetVestingSchedule(
                sampleUser,
                i,
                tokensToVestAtStart,
                tokensToVestAfterFirstInterval,
                0,
                scheduleStartTime,
                cliffEndTime,
                intervalLength,
                maxIntervals,
                growthRateProportion
            );
        }
        VVVVestingInstance.batchSetVestingSchedule(setVestingScheduleParams);
        vm.stopPrank();
    }

    // tests that a SetVestingSchedule event is emitted when removeVestingSchedule is called
    function testEmitSetVestingScheduleOnRemove() public {
        uint256 vestingScheduleIndex = 0;
        uint88 tokensToVestAtStart = 1_000 * 1e18; //1k tokens
        uint120 tokensToVestAfterFirstInterval = 100 * 1e18; //100 tokens
        uint128 amountWithdrawn = 0;
        uint32 scheduleStartTime = uint32(block.timestamp + 60 * 60 * 24 * 2); //2 days from now
        uint32 cliffEndTime = scheduleStartTime + 60 * 60 * 24 * 365; //1 year from scheduleStartTime
        uint32 intervalLength = 60 * 60 * 6 * 365; //3 months
        uint16 maxIntervals = 100;
        uint64 growthRateProportion = 0;

        setVestingScheduleFromManager(
            sampleUser,
            vestingScheduleIndex,
            tokensToVestAtStart,
            tokensToVestAfterFirstInterval,
            amountWithdrawn,
            scheduleStartTime,
            cliffEndTime,
            intervalLength,
            maxIntervals,
            growthRateProportion
        );

        vm.expectEmit(address(VVVVestingInstance));
        emit VVVVesting.RemoveVestingSchedule(sampleUser, vestingScheduleIndex);
        removeVestingScheduleFromManager(sampleUser, vestingScheduleIndex);
    }

    //tests emission of SetVestedToken when admin sets vested token address
    function testEmitSetVestedToken() public {
        vm.startPrank(vestingManager, vestingManager);
        address newToken = address(new MockERC20(18));
        vm.expectEmit(address(VVVVestingInstance));
        emit VVVVesting.SetVestedToken(newToken);
        VVVVestingInstance.setVestedToken(newToken);
        vm.stopPrank();
    }
}
