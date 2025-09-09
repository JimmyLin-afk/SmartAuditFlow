//SPDX-License-Identifier: MIT
pragma solidity 0.8.23;

import { MockERC721 } from "contracts/mock/MockERC721.sol";
import { VVVS1NFTStaking } from "contracts/staking/VVVS1NFTStaking.sol";
import { VVVStakingTestBase } from "test/staking/VVVStakingTestBase.sol";

/**
 * @title VVVS1NFTStaking Unit Tests
 * @dev use "forge test --match-contract VVVS1NFTStakingUnitTests" to run tests
 * @dev use "forge coverage --match-contract VVVS1NFTStaking" to run coverage
 */
contract VVVS1NFTStakingUnitTests is VVVStakingTestBase {
    function setUp() public {
        vm.startPrank(deployer, deployer);
        MockERC721Instance = new MockERC721();
        S1NFTStakingInstance = new VVVS1NFTStaking(address(MockERC721Instance));
        vm.stopPrank();
    }

    //tests deployment of the mock erc721 and s1 nft staking contracts
    function testDeployments() public {
        assertTrue(address(MockERC721Instance) != address(0));
        assertTrue(address(S1NFTStakingInstance) != address(0));
    }

    //tests that a user can stake a token with the desired duration and contract storage is updated correctly
    function testStake() public {
        VVVS1NFTStaking.StakeDuration stakeDuration = VVVS1NFTStaking.StakeDuration.DAYS_180;

        //mint a token to the user (id 1)
        vm.startPrank(sampleUser, sampleUser);
        MockERC721Instance.safeMint(sampleUser);
        MockERC721Instance.setApprovalForAll(address(S1NFTStakingInstance), true);

        //stake the token for 180 days
        S1NFTStakingInstance.stake(1, stakeDuration);
        vm.stopPrank();

        //check that the `stakes` mapping is updated correctly
        (
            uint256 tokenId,
            uint256 startTime,
            VVVS1NFTStaking.StakeDuration stakeDurationRead
        ) = S1NFTStakingInstance.stakes(sampleUser, 0);

        assertEq(tokenId, 1);
        assertEq(startTime, block.timestamp);
        assertEq(uint8(stakeDuration), uint8(stakeDurationRead));
    }

    //tests staking multiple tokens
    function testStakeMultiple() public {
        uint256 tokensToMint = 123;
        VVVS1NFTStaking.StakeDuration stakeDuration = VVVS1NFTStaking.StakeDuration.DAYS_180;

        vm.startPrank(sampleUser, sampleUser);
        for (uint256 i = 0; i < tokensToMint; i++) {
            MockERC721Instance.safeMint(sampleUser);
        }
        MockERC721Instance.setApprovalForAll(address(S1NFTStakingInstance), true);

        for (uint256 i = 0; i < tokensToMint; i++) {
            S1NFTStakingInstance.stake(i + 1, stakeDuration);
        }
        vm.stopPrank();

        for (uint256 i = 0; i < tokensToMint; i++) {
            (
                uint256 tokenId,
                uint256 startTime,
                VVVS1NFTStaking.StakeDuration stakeDurationRead
            ) = S1NFTStakingInstance.stakes(sampleUser, i);

            assertEq(tokenId, i + 1);
            assertEq(startTime, block.timestamp);
            assertEq(uint8(stakeDuration), uint8(stakeDurationRead));
        }
    }

    //tests that the Stake event is emitted with the correct data when a user stakes
    function testEmitStake() public {
        VVVS1NFTStaking.StakeDuration stakeDuration = VVVS1NFTStaking.StakeDuration.DAYS_180;

        //mint a token to the user (id 1)
        vm.startPrank(sampleUser, sampleUser);
        MockERC721Instance.safeMint(sampleUser);
        MockERC721Instance.setApprovalForAll(address(S1NFTStakingInstance), true);

        //stake the token for 180 days
        vm.expectEmit(address(S1NFTStakingInstance));
        emit VVVS1NFTStaking.Stake(sampleUser, 1, stakeDuration);
        S1NFTStakingInstance.stake(1, stakeDuration);
        vm.stopPrank();
    }

    //tests that a user can unstake
    function testUnstake() public {
        uint256 tokenId = 1;
        VVVS1NFTStaking.StakeDuration stakeDuration = VVVS1NFTStaking.StakeDuration.DAYS_180;

        //mint a token to the user (id 1)
        vm.startPrank(sampleUser, sampleUser);
        MockERC721Instance.safeMint(sampleUser);
        MockERC721Instance.setApprovalForAll(address(S1NFTStakingInstance), true);
        S1NFTStakingInstance.stake(tokenId, stakeDuration);

        //advance the staking duration chosen
        advanceBlockNumberAndTimestampInSeconds(S1NFTStakingInstance.stakeDurations(stakeDuration) + 1);

        S1NFTStakingInstance.unstake(tokenId);
        vm.stopPrank();

        //assert that the user has the token of id 1 back and their stakes is length 0 indicating the token transfer succeeded, and the stake array entry was removed
        assertTrue(MockERC721Instance.ownerOf(tokenId) == sampleUser);
        assertEq(S1NFTStakingInstance.getStakes(sampleUser).length, 0);
    }

    //test unstaking multiple tokens in a different order than that in which they were staked
    function testUnstakeMultiple() public {
        uint256 tokensToMint = 123;
        VVVS1NFTStaking.StakeDuration stakeDuration = VVVS1NFTStaking.StakeDuration.DAYS_360;

        vm.startPrank(sampleUser, sampleUser);
        for (uint256 i = 0; i < tokensToMint; i++) {
            MockERC721Instance.safeMint(sampleUser);
        }
        MockERC721Instance.setApprovalForAll(address(S1NFTStakingInstance), true);

        for (uint256 i = 0; i < tokensToMint; i++) {
            S1NFTStakingInstance.stake(i + 1, stakeDuration);
        }

        advanceBlockNumberAndTimestampInSeconds(S1NFTStakingInstance.stakeDurations(stakeDuration) + 1);

        for (uint256 i = tokensToMint; i > 0; i--) {
            S1NFTStakingInstance.unstake(i);
            assertEq(MockERC721Instance.ownerOf(i), sampleUser);
        }
        vm.stopPrank();

        assertEq(S1NFTStakingInstance.getStakes(sampleUser).length, 0);
    }

    //test that a user cannot unstake a token they didn't stake
    function testUnstakeNotStaked() public {
        uint256 tokenId = 1;
        VVVS1NFTStaking.StakeDuration stakeDuration = VVVS1NFTStaking.StakeDuration.DAYS_180;

        //mint a token to the user (id 1)
        vm.startPrank(sampleUser, sampleUser);
        MockERC721Instance.safeMint(sampleUser);
        MockERC721Instance.setApprovalForAll(address(S1NFTStakingInstance), true);
        S1NFTStakingInstance.stake(tokenId, stakeDuration);
        vm.stopPrank();

        //advance the staking duration chosen
        advanceBlockNumberAndTimestampInSeconds(S1NFTStakingInstance.stakeDurations(stakeDuration));

        vm.startPrank(deployer, deployer);
        vm.expectRevert(VVVS1NFTStaking.NotTokenOwner.selector);
        S1NFTStakingInstance.unstake(tokenId);
        vm.stopPrank();
    }

    //tests that a user cannot unstake before the end time
    function testUnstakeBeforeEndTime() public {
        uint256 tokenId = 1;
        VVVS1NFTStaking.StakeDuration stakeDuration = VVVS1NFTStaking.StakeDuration.DAYS_180;

        //mint a token to the user (id 1)
        vm.startPrank(sampleUser, sampleUser);
        MockERC721Instance.safeMint(sampleUser);
        MockERC721Instance.setApprovalForAll(address(S1NFTStakingInstance), true);
        S1NFTStakingInstance.stake(tokenId, stakeDuration);

        //advance the staking duration chosen
        advanceBlockNumberAndTimestampInSeconds(S1NFTStakingInstance.stakeDurations(stakeDuration));

        vm.expectRevert(VVVS1NFTStaking.StakeLocked.selector);
        S1NFTStakingInstance.unstake(tokenId);
        vm.stopPrank();
    }

    //tests that the Unstake event is emitted with correct data when a user unstakes
    function testEmitUnstake() public {
        uint256 tokenId = 1;
        VVVS1NFTStaking.StakeDuration stakeDuration = VVVS1NFTStaking.StakeDuration.DAYS_180;

        //mint a token to the user (id 1)
        vm.startPrank(sampleUser, sampleUser);
        MockERC721Instance.safeMint(sampleUser);
        MockERC721Instance.setApprovalForAll(address(S1NFTStakingInstance), true);
        S1NFTStakingInstance.stake(tokenId, stakeDuration);

        //advance the staking duration chosen
        advanceBlockNumberAndTimestampInSeconds(S1NFTStakingInstance.stakeDurations(stakeDuration) + 1);

        vm.expectEmit(address(S1NFTStakingInstance));
        emit VVVS1NFTStaking.Unstake(sampleUser, tokenId);
        S1NFTStakingInstance.unstake(tokenId);
        vm.stopPrank();
    }

    //tests that a user can read their stakes correctly via getStakes
    function testGetStakes() public {
        uint256 tokensToMint = 8;
        VVVS1NFTStaking.StakeDuration stakeDuration = VVVS1NFTStaking.StakeDuration.DAYS_180;

        vm.startPrank(sampleUser, sampleUser);
        for (uint256 i = 0; i < tokensToMint; i++) {
            MockERC721Instance.safeMint(sampleUser);
        }
        MockERC721Instance.setApprovalForAll(address(S1NFTStakingInstance), true);

        for (uint256 i = 0; i < tokensToMint; i++) {
            S1NFTStakingInstance.stake(i + 1, stakeDuration);
        }
        vm.stopPrank();

        VVVS1NFTStaking.StakeData[] memory stakes = S1NFTStakingInstance.getStakes(sampleUser);

        assertEq(stakes.length, tokensToMint);

        for (uint256 i = 0; i < tokensToMint; i++) {
            VVVS1NFTStaking.StakeData memory stake = stakes[i];
            assertEq(stake.tokenId, i + 1);
            assertEq(stake.startTime, block.timestamp);
            assertEq(uint8(stake.duration), uint8(stakeDuration));
        }
    }
}
