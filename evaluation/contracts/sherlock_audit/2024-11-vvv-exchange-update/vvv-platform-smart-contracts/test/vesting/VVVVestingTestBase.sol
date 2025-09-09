//SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

/**
 * @title VVVVesting Test Base
 *   @dev storage, setup, and helper functions for VVVVesting tests
 */

import { MockERC20 } from "contracts/mock/MockERC20.sol";
import { Test } from "lib/forge-std/src/Test.sol"; //for stateless tests
import { VVVAuthorizationRegistry } from "contracts/auth/VVVAuthorizationRegistry.sol";
import { VVVVesting } from "contracts/vesting/VVVVesting.sol";

abstract contract VVVVestingTestBase is Test {
    VVVAuthorizationRegistry AuthRegistry;
    MockERC20 public VVVTokenInstance;
    VVVVesting public VVVVestingInstance;

    uint256 public constant DECIMALS = 18;
    uint256 public constant DENOMINATOR = 100;

    uint256 public deployerKey = 1;
    uint256 public vestingManagerKey = 2;
    uint256 public userKey = 3;
    address deployer = vm.addr(deployerKey);
    address vestingManager = vm.addr(vestingManagerKey);
    address sampleUser = vm.addr(userKey);

    uint256 blockNumber;
    uint256 blockTimestamp;

    bytes32 vestingManagerRole = keccak256("VESTING_MANAGER_ROLE");
    uint48 defaultAdminTransferDelay = 1 days;

    struct VestingParams {
        uint256 vestingScheduleIndex;
        uint88 tokensToVestAtStart;
        uint120 tokensToVestAfterFirstInterval;
        uint128 amountWithdrawn;
        uint32 scheduleStartTime;
        uint32 cliffEndTime;
        uint32 intervalLength;
        uint16 maxIntervals;
        uint64 growthRateProportion;
    }

    function advanceBlockNumberAndTimestampInBlocks(uint256 blocks) public {
        blockNumber += blocks;
        blockTimestamp += blocks * 12; //seconds per block
        vm.warp(blockTimestamp);
        vm.roll(blockNumber);
    }

    function advanceBlockNumberAndTimestampInSeconds(uint256 secondsToAdvance) public {
        blockNumber += secondsToAdvance / 12; //seconds per block
        blockTimestamp += secondsToAdvance;
        vm.warp(blockTimestamp);
        vm.roll(blockNumber);
    }

    function setVestingScheduleFromManager(
        address _user,
        uint256 _vestingScheduleIndex,
        uint88 _tokensToVestAtStart,
        uint120 _tokensToVestAfterFirstInterval,
        uint128 _amountWithdrawn,
        uint32 _scheduleStartTime,
        uint32 _cliffEndTime,
        uint32 _intervalLength,
        uint16 _maxIntervals,
        uint64 _growthRateProportion
    ) public {
        vm.startPrank(vestingManager, vestingManager);
        VVVVestingInstance.setVestingSchedule(
            _user,
            _vestingScheduleIndex,
            _tokensToVestAtStart,
            _tokensToVestAfterFirstInterval,
            _amountWithdrawn,
            _scheduleStartTime,
            _cliffEndTime,
            _intervalLength,
            _maxIntervals,
            _growthRateProportion
        );
        vm.stopPrank();
    }

    function removeVestingScheduleFromManager(address _user, uint256 _vestingScheduleIndex) public {
        vm.startPrank(vestingManager, vestingManager);
        VVVVestingInstance.removeVestingSchedule(_user, _vestingScheduleIndex);
        vm.stopPrank();
    }

    function withdrawVestedTokensAsUser(
        address _caller,
        uint128 _amount,
        address _destination,
        uint256 _vestingScheduleIndex
    ) public {
        vm.startPrank(_caller, _caller);
        VVVVestingInstance.withdrawVestedTokens(_amount, _destination, _vestingScheduleIndex);
        vm.stopPrank();
    }

    // generates a SetVestingScheduleParams array with the specified number of users and the specified parameter varied,
    // and varies vestedUser and vestingScheduleIndex because these are the factors by which the vesting schedule is identified
    function generateSetVestingScheduleData(
        uint256 _numUsers,
        uint256 _growthRateProportion,
        string memory paramToVary
    ) public view returns (VVVVesting.SetVestingScheduleParams[] memory) {
        VVVVesting.SetVestingScheduleParams[]
            memory setVestingScheduleParams = new VVVVesting.SetVestingScheduleParams[](_numUsers);

        if (keccak256(abi.encodePacked(paramToVary)) == keccak256(abi.encodePacked("vestedUser"))) {
            for (uint256 i = 0; i < _numUsers; i++) {
                setVestingScheduleParams[i].vestedUser = address(
                    uint160(uint256(keccak256(abi.encodePacked(i))))
                );
                setVestingScheduleParams[i].vestingScheduleIndex = 0;
                setVestingScheduleParams[i].vestingSchedule.tokensToVestAfterFirstInterval = uint120(
                    (i + 1) * 100 * 1e18
                ); //100 tokens
                setVestingScheduleParams[i].vestingSchedule.scheduleStartTime = uint32(
                    block.timestamp + i * 60 * 24 * 2
                ); //2 days from now
                setVestingScheduleParams[i].vestingSchedule.cliffEndTime = uint32(
                    block.timestamp + i * 60 * 24 * 30
                ); //30 days
                setVestingScheduleParams[i].vestingSchedule.intervalLength = 60 * 24 * 30; //30 days
                setVestingScheduleParams[i].vestingSchedule.maxIntervals = 10;
                setVestingScheduleParams[i].vestingSchedule.growthRateProportion = uint64(
                    _growthRateProportion
                );
            }
        } else if (
            keccak256(abi.encodePacked(paramToVary)) == keccak256(abi.encodePacked("vestingScheduleIndex"))
        ) {
            for (uint256 i = 0; i < _numUsers; i++) {
                setVestingScheduleParams[i].vestedUser = address(
                    uint160(uint256(keccak256(abi.encodePacked("vestedUser"))))
                );
                setVestingScheduleParams[i].vestingScheduleIndex = i;
                setVestingScheduleParams[i].vestingSchedule.tokensToVestAfterFirstInterval = 100 * 1e18; //100 tokens
                setVestingScheduleParams[i].vestingSchedule.scheduleStartTime =
                    uint32(block.timestamp) *
                    60 *
                    24 *
                    2; //2 days from now
                setVestingScheduleParams[i].vestingSchedule.cliffEndTime =
                    uint32(block.timestamp) +
                    uint32(60 * 24 * 30); //30 days
                setVestingScheduleParams[i].vestingSchedule.intervalLength = 60 * 24 * 30; //30 days
                setVestingScheduleParams[i].vestingSchedule.maxIntervals = 10;
                setVestingScheduleParams[i].vestingSchedule.growthRateProportion = uint64(
                    _growthRateProportion
                );
            }
        } else {
            revert("invalid paramToVary");
        }

        return setVestingScheduleParams;
    }
}
