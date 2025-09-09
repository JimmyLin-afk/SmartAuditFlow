//SPDX-License-Identifier: MIT
pragma solidity 0.8.23;

/**
 * @dev Base for testing VVVETHStaking.sol
 */

import "lib/forge-std/src/Test.sol";
import { NonReceivable } from "test/utils/NonReceivable.sol";
import { MockERC721 } from "contracts/mock/MockERC721.sol";
import { VVVAuthorizationRegistry } from "contracts/auth/VVVAuthorizationRegistry.sol";
import { VVVETHStaking } from "contracts/staking/VVVETHStaking.sol";
import { VVVERC20LaunchpadStaking } from "contracts/staking/VVVERC20LaunchpadStaking.sol";
import { VVVLaunchpadStaking } from "contracts/staking/VVVLaunchpadStaking.sol";
import { VVVS1NFTStaking } from "contracts/staking/VVVS1NFTStaking.sol";
import { VVVToken } from "contracts/tokens/VvvToken.sol";

abstract contract VVVStakingTestBase is Test {
    NonReceivable NonReceivableCaller;
    MockERC721 MockERC721Instance;
    VVVAuthorizationRegistry AuthRegistry;
    VVVToken VvvTokenInstance;
    VVVETHStaking EthStakingInstance;
    VVVERC20LaunchpadStaking ERC20LaunchpadStakingInstance;
    VVVLaunchpadStaking LaunchpadStakingInstance;
    VVVS1NFTStaking S1NFTStakingInstance;

    uint256 deployerKey = 1234;
    uint256 ethStakingManagerKey = 1235;
    uint256 launchpadStakingManagerKey = 1236;
    uint256 sampleUserKey = 1234567;

    address DEAD_ADDRESS = 0x000000000000000000000000000000000000dEaD;
    address deployer = vm.addr(deployerKey);
    address ethStakingManager = vm.addr(ethStakingManagerKey);
    address launchpadStakingManager = vm.addr(launchpadStakingManagerKey);
    address sampleUser = vm.addr(sampleUserKey);

    uint256 blockNumber;
    uint256 blockTimestamp;
    uint256 chainId = 31337; //test chain id - leave this alone

    bytes32 ethStakingManagerRole = keccak256("ETH_STAKING_MANAGER_ROLE");
    bytes32 launchpadStakingManagerRole = keccak256("LAUNCHPAD_STAKING_MANAGER_ROLE");
    uint48 defaultAdminTransferDelay = 1 days;

    uint256[] public stakingDurations;

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

    function setDefaultLaunchpadStakingDurations() public {
        stakingDurations = new uint256[](5);
        stakingDurations[0] = 30 days;
        stakingDurations[1] = 90 days;
        stakingDurations[2] = 180 days;
        stakingDurations[3] = 360 days;
        stakingDurations[4] = 720 days;
    }
}
