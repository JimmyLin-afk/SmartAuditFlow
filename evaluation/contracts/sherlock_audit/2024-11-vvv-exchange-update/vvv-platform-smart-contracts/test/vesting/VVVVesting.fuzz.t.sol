//SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import { Math } from "@openzeppelin/contracts/utils/math/Math.sol";
import { MockERC20 } from "contracts/mock/MockERC20.sol";
import { VVVVesting } from "contracts/vesting/VVVVesting.sol";
import { VVVVestingTestBase } from "test/vesting/VVVVestingTestBase.sol";
import { VVVAuthorizationRegistry } from "contracts/auth/VVVAuthorizationRegistry.sol";

/**
 * @title VVVVesting Fuzz Tests
 * @dev use "forge test --match-contract VVVVestingFuzzTests" to run tests
 * @dev use "forge coverage --match-contract VVVVestingFuzzTests" to run coverage
 */
contract VVVVestingFuzzTests is VVVVestingTestBase {
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

        VVVTokenInstance.mint(address(VVVVestingInstance), 1_000_000 * 1e18); //1M tokens

        vm.stopPrank();
    }

    //test setting a vesting schedule and withdrawing tokens, assert that the correct amount of tokens are withdrawn
    //fuzzes with withdraw values between 0 and (vested-withdrawn)
    function testFuzz_WithdrawVestedTokens(uint256 _tokenAmountToWithdraw, uint256 _vestingTime) public {
        uint256 tokenAmountToWithdraw = bound(_tokenAmountToWithdraw, 1, 1_000_000 * 1e18); //1M tokens

        VestingParams memory params = VestingParams({
            vestingScheduleIndex: 0,
            tokensToVestAtStart: 1_000 * 1e18, //1k tokens
            tokensToVestAfterFirstInterval: 100 * 1e18, //100 tokens
            amountWithdrawn: 0,
            scheduleStartTime: 1,
            cliffEndTime: 1 + 60, //1 minute cliff
            intervalLength: 30,
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

        uint256 vestingTime = _vestingTime > params.cliffEndTime ? _vestingTime : params.cliffEndTime;
        advanceBlockNumberAndTimestampInSeconds(vestingTime);

        uint256 vestedAmount = VVVVestingInstance.getVestedAmount(sampleUser, 0); // vestingScheduleIndex
        (, , uint128 withdrawnTokens, , , , , ) = VVVVestingInstance.userVestingSchedules(sampleUser, 0); // vestingScheduleIndex
        uint128 amountToWithdraw = uint128(
            Math.min(vestedAmount - withdrawnTokens, tokenAmountToWithdraw)
        );

        withdrawVestedTokensAsUser(sampleUser, amountToWithdraw, sampleUser, 0); // vestingScheduleIndex
        assertEq(VVVTokenInstance.balanceOf(sampleUser), amountToWithdraw);
    }

    //tests both that the correct amount of vested and withdrawn tokens are read
    //basically tests the on-contract logic against reference logic here,
    //using same contract functions but directly feeding in relevant inputs
    function testFuzz_GetVestedAmount(address _vestedUser, uint256 _vestingTime) public {
        vm.assume(_vestedUser != address(0));

        uint256 vestingScheduleIndex = 0;
        uint88 tokensToVestAtStart = 1_000 * 1e18; //1k tokens
        uint120 tokensToVestAfterFirstInterval = 100 * 1e18; //100 tokens
        uint128 amountWithdrawn = 0;
        uint32 scheduleStartTime = 1;
        uint32 cliffEndTime = scheduleStartTime + 60; //1 minute cliff
        uint32 intervalLength = 30;
        uint16 maxIntervals = 100;
        uint64 growthRateProportion = 0;

        setVestingScheduleFromManager(
            _vestedUser,
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

        //bound to some considerable amount past the end of the schedule
        uint256 vestingTime = bound(_vestingTime, cliffEndTime, intervalLength * maxIntervals * 100);
        advanceBlockNumberAndTimestampInSeconds(vestingTime);

        uint256 vestedAmount = VVVVestingInstance.getVestedAmount(_vestedUser, vestingScheduleIndex);
        uint256 elapsedIntervals = (block.timestamp - cliffEndTime) / intervalLength;

        uint256 refAccuredPostCliff = VVVVestingInstance.calculateVestedAmountAtInterval(
            tokensToVestAfterFirstInterval,
            elapsedIntervals,
            growthRateProportion
        );

        uint256 referenceVestedAmount = Math.min(
            maxIntervals * tokensToVestAfterFirstInterval,
            refAccuredPostCliff
        ) + tokensToVestAtStart;

        assertEq(vestedAmount, referenceVestedAmount);
    }
}
