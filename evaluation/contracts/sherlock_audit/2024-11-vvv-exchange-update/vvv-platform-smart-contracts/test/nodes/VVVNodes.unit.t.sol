//SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import { Strings } from "@openzeppelin/contracts/utils/Strings.sol";
import { VVVAuthorizationRegistry } from "contracts/auth/VVVAuthorizationRegistry.sol";
import { VVVAuthorizationRegistryChecker } from "contracts/auth/VVVAuthorizationRegistryChecker.sol";
import { VVVNodes } from "contracts/nodes/VVVNodes.sol";
import { VVVNodesTestBase } from "./VVVNodesTestBase.sol";

contract VVVNodesUnitTest is VVVNodesTestBase {
    using Strings for uint256;

    function setUp() public {
        vm.startPrank(deployer, deployer);
        AuthRegistry = new VVVAuthorizationRegistry(defaultAdminTransferDelay, deployer);
        NodesInstance = new VVVNodes(address(AuthRegistry), defaultBaseURI, activationThreshold);
        vm.stopPrank();
    }

    function testDeployment() public {
        assertNotEq(address(NodesInstance), address(0));
    }

    //tests that the adminMint function correctly mints a node to the supplied destination address with the defined locked tokens
    function testAdminMint() public {
        vm.startPrank(deployer, deployer);

        uint256 unvestedAmount = (sampleLockedTokens * 60) / 100;
        uint256 lockedTransactionProcessingYield = sampleLockedTokens - unvestedAmount;
        uint256 amountToVestPerSecond = unvestedAmount / NodesInstance.VESTING_DURATION();

        vm.expectEmit(address(NodesInstance));
        emit VVVNodes.Mint(
            1,
            sampleUser,
            unvestedAmount,
            lockedTransactionProcessingYield,
            amountToVestPerSecond
        );

        NodesInstance.adminMint(sampleUser, sampleLockedTokens);
        vm.stopPrank();

        (
            uint256 unvestedAmountRead,
            uint256 vestingSinceRead,
            uint256 lockedTransactionProcessingYieldRead,
            uint256 claimableAmountRead,
            uint256 amountToVestPerSecondRead,
            uint256 stakedAmountRead
        ) = NodesInstance.tokenData(1);

        assertEq(NodesInstance.balanceOf(sampleUser), 1);
        assertEq(unvestedAmountRead, unvestedAmount);
        assertEq(vestingSinceRead, 0);
        assertEq(lockedTransactionProcessingYieldRead, lockedTransactionProcessingYield);
        assertEq(claimableAmountRead, 0);
        assertEq(amountToVestPerSecondRead, amountToVestPerSecond);
        assertEq(stakedAmountRead, 0);
    }

    //tests that an admin cannot mint above the max supply
    function testAdminMintAboveMaxSupply() public {
        vm.startPrank(deployer, deployer);

        for (uint256 i = 0; i < NodesInstance.TOTAL_SUPPLY(); ++i) {
            NodesInstance.adminMint(sampleUser, sampleLockedTokens);
        }

        vm.expectRevert(VVVNodes.MaxSupplyReached.selector);
        NodesInstance.adminMint(sampleUser, sampleLockedTokens);
        vm.stopPrank();
    }

    //tests that a non-admin cannot call adminMint
    function testNonAdminCannotAdminMint() public {
        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVAuthorizationRegistryChecker.UnauthorizedCaller.selector);
        NodesInstance.adminMint(sampleUser, uint256(1));
        vm.stopPrank();
    }

    //tests setting baseURI
    function testSetBaseURI() public {
        string memory newBaseURI = "https://example.com/token/";
        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(deployer, sampleLockedTokens);
        NodesInstance.setBaseURI(newBaseURI);
        vm.stopPrank();
        assertEq(NodesInstance.baseURI(), newBaseURI);
    }

    //tests that a new base URI reflects in the ERC721 function which reads an individual token's URI
    function testReadTokenURIAfterBaseURIUpdate() public {
        uint256 tokenId = 1;
        string memory newBaseURI = "https://example.com/token/";

        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(sampleUser, sampleLockedTokens);

        string memory tokenURIdefaultBase = NodesInstance.tokenURI(tokenId);
        string memory concatRefDefaultURI = string(abi.encodePacked(defaultBaseURI, tokenId.toString()));
        assertEq(tokenURIdefaultBase, concatRefDefaultURI);

        NodesInstance.setBaseURI(newBaseURI);
        vm.stopPrank();

        string memory tokenURINewBase = NodesInstance.tokenURI(tokenId);
        string memory concatRefNewBaseURI = string(abi.encodePacked(newBaseURI, tokenId.toString()));
        assertEq(tokenURINewBase, concatRefNewBaseURI);
    }

    //tests that a non-admin cannot set the tokenURI
    function testNonAdminCannotSetBaseURI() public {
        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVAuthorizationRegistryChecker.UnauthorizedCaller.selector);
        NodesInstance.setBaseURI("https://example.com/token/");
        vm.stopPrank();
    }

    //tests that a node owner can stake $VVV via their node s.t. the node is activated
    function testStakeActivate() public {
        vm.deal(sampleUser, activationThreshold);

        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(sampleUser, sampleLockedTokens); //mints tokenId 1
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        NodesInstance.stake{ value: activationThreshold }(1);
        vm.stopPrank();

        bool isActive = NodesInstance.isNodeActive(1);
        (, , , , , uint256 stakedAmount) = NodesInstance.tokenData(1);
        assertTrue(isActive);
        assertEq(stakedAmount, activationThreshold);
        assertEq(address(NodesInstance).balance, activationThreshold);
    }

    //tests that a node owner can stake sub-activation threshold amount
    function testStakeSubActivationThreshold() public {
        vm.deal(sampleUser, activationThreshold);

        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(sampleUser, sampleLockedTokens); //mints tokenId 1
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        NodesInstance.stake{ value: activationThreshold - 1 }(1);
        vm.stopPrank();

        bool isActive = NodesInstance.isNodeActive(1);
        assertFalse(isActive);
    }

    //tests that an attempt to stake 0 $VVV will revert
    function testStakeZeroMsgValue() public {
        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(sampleUser, sampleLockedTokens); //mints tokenId 1
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVNodes.ZeroTokenTransfer.selector);
        NodesInstance.stake{ value: 0 }(1);
        vm.stopPrank();
    }

    //test that a user's attempt to stake in an unowned node reverts
    function testStakeUnownedToken() public {
        vm.deal(sampleUser, activationThreshold);

        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(deployer, sampleLockedTokens); //mints tokenId 1
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVNodes.CallerIsNotTokenOwner.selector);
        NodesInstance.stake{ value: activationThreshold }(1);
        vm.stopPrank();
    }

    //tests that a node owner can unstake $VVV via their node
    function testUnstake() public {
        vm.deal(sampleUser, activationThreshold);
        uint256 tokenId = 1;

        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(sampleUser, sampleLockedTokens); //mints tokenId 1
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        NodesInstance.stake{ value: activationThreshold }(tokenId);

        bool isActiveBeforeUnstake = NodesInstance.isNodeActive(tokenId);

        uint256 userBalanceBeforeUnstake = sampleUser.balance;
        NodesInstance.unstake(tokenId, activationThreshold);
        vm.stopPrank();

        bool isActiveAfterUnstake = NodesInstance.isNodeActive(tokenId);

        //the node is active after staking and inactive after unstaking
        assertTrue(isActiveBeforeUnstake);
        assertFalse(isActiveAfterUnstake);

        //the same staked amount is now back in the user's wallet + some dust due to 1s of vesting
        assertGt(sampleUser.balance, userBalanceBeforeUnstake);
    }

    //test that a user's attempt to unstake from an unowned node reverts
    function testUnstakeUnownedToken() public {
        vm.deal(sampleUser, activationThreshold);
        uint256 tokenId = 1;

        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(sampleUser, sampleLockedTokens); //mints tokenId 1
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        NodesInstance.stake{ value: activationThreshold }(tokenId);
        vm.stopPrank();

        vm.startPrank(deployer, deployer);
        vm.expectRevert(VVVNodes.CallerIsNotTokenOwner.selector);
        NodesInstance.unstake(tokenId, activationThreshold);
        vm.stopPrank();
    }

    //tests that unstaking s.t. the node is below the activation threshold triggers the actions associated with deactivation
    function testUnstakeDeactivatesNode() public {
        vm.deal(sampleUser, activationThreshold);
        uint256 tokenId = 1;

        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(sampleUser, sampleLockedTokens); //mints tokenId 1
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        NodesInstance.stake{ value: activationThreshold }(tokenId);

        //full unvested amount
        (
            uint256 unvestedAmountPreDeactivation,
            uint256 vestingSince,
            ,
            ,
            uint256 amountToVestPerSecond,

        ) = NodesInstance.tokenData(tokenId);

        //advance enough to accrue some vested tokens
        advanceBlockNumberAndTimestampInSeconds(2 weeks);

        //unstaking 1 token should deactivate the node
        NodesInstance.unstake(tokenId, 1);
        vm.stopPrank();

        //post-deactivation unvested amount is slightly less, the difference has become claimable
        (uint256 unvestedAmountPostDeactivation, , , uint256 claimableAmount, , ) = NodesInstance
            .tokenData(tokenId);

        uint256 expectedVestedAmount = (block.timestamp - vestingSince) * amountToVestPerSecond; //2 weeks of vesting

        //change in unvested amount is same as amount made claimable during deactivation
        assertEq(unvestedAmountPreDeactivation - unvestedAmountPostDeactivation, claimableAmount);

        //assert that the claimable amount is the amount expected to be vested after 2 weeks
        assertEq(expectedVestedAmount, claimableAmount);
    }

    //tests claim, entire amount accrued during vesting period should be claimable by a user. utilizes placeholder logic in mint() to set TokenData
    function testClaim() public {
        vm.deal(sampleUser, activationThreshold);
        vm.deal(address(NodesInstance), type(uint128).max);

        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(sampleUser, sampleLockedTokens); //mints token of ID 1
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        uint256 userTokenId = 1;

        uint256 vestingDuration = secondsInTwoYears;
        uint256 refTotalVestedTokens = (sampleLockedTokens * 60) / 100;

        //check pre-vesting unvested tokens and owner wallet balance for userTokenId
        (uint256 unvestedAmountPreClaim, , , , , ) = NodesInstance.tokenData(userTokenId);

        //stake the activation threshold to activate the node
        NodesInstance.stake{ value: activationThreshold }(userTokenId);

        advanceBlockNumberAndTimestampInSeconds(vestingDuration * 2);

        NodesInstance.claim(userTokenId);

        //check post-vesting unvested tokens and owner wallet balance for userTokenId
        (uint256 unvestedAmountPostClaim, , , , , ) = NodesInstance.tokenData(userTokenId);
        uint256 balancePostClaim = sampleUser.balance;
        vm.stopPrank();

        assertEq(unvestedAmountPreClaim, refTotalVestedTokens);
        assertEq(unvestedAmountPostClaim, 0);
        assertEq(refTotalVestedTokens, balancePostClaim);
    }

    //tests that token claims are accurate when only a partial vesting period has elapsed
    function testClaimPartial() public {
        vm.deal(sampleUser, activationThreshold);
        vm.deal(address(NodesInstance), type(uint128).max);

        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(sampleUser, sampleLockedTokens); //mints token of ID 1
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        uint256 userTokenId = 1;

        uint256 vestingDuration = secondsInTwoYears;

        //stake the activation threshold to activate the node
        NodesInstance.stake{ value: activationThreshold }(userTokenId);

        uint256 balancePostStake = sampleUser.balance;

        advanceBlockNumberAndTimestampInSeconds((vestingDuration / 2) + 1);

        NodesInstance.claim(userTokenId);

        //check post-vesting unvested tokens and owner wallet balance for userTokenId
        (uint256 unvestedAmountPostClaim, , , , uint256 vestedPerSecond, ) = NodesInstance.tokenData(
            userTokenId
        );
        uint256 balancePostClaim = sampleUser.balance;
        vm.stopPrank();

        uint256 refTotalVestedTokens = (vestedPerSecond * NodesInstance.VESTING_DURATION()) / 2;

        //assert that the change in vested tokens is the change in user balance,
        //assert that the change in unvested tokens corresponds to the change in vested tokens
        assertEq(refTotalVestedTokens, balancePostClaim - balancePostStake);
        assertEq(((sampleLockedTokens * 60) / 100) - refTotalVestedTokens, unvestedAmountPostClaim);
    }

    //tests claim with a tokenId that is not owned by the caller
    function testClaimNotOwned() public {
        vm.deal(sampleUser, activationThreshold);
        vm.deal(address(NodesInstance), type(uint128).max);

        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(sampleUser, sampleLockedTokens); //mints token of ID 1
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        uint256 userTokenId = 1;

        uint256 vestingDuration = secondsInTwoYears;

        //stake the activation threshold to activate the node
        NodesInstance.stake{ value: activationThreshold }(userTokenId);
        vm.stopPrank();

        advanceBlockNumberAndTimestampInSeconds(vestingDuration * 2);

        vm.startPrank(deployer, deployer);
        vm.expectRevert(VVVNodes.CallerIsNotTokenOwner.selector);
        NodesInstance.claim(userTokenId);
        vm.stopPrank();
    }

    //tests claim of zero $VVV (can't be defined as a claim amount, but if a user claims with a token they own the instant after activating, this error will be thrown)
    function testClaimZero() public {
        vm.deal(sampleUser, activationThreshold);
        vm.deal(address(NodesInstance), type(uint128).max);

        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(sampleUser, sampleLockedTokens); //mints token of ID 1
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        uint256 userTokenId = 1;
        uint256 vestingDuration = secondsInTwoYears;

        //stake the activation threshold to activate the node
        NodesInstance.stake{ value: activationThreshold }(userTokenId);

        advanceBlockNumberAndTimestampInSeconds(vestingDuration * 2);
        NodesInstance.claim(userTokenId);

        vm.expectRevert(VVVNodes.NoClaimableTokens.selector);
        NodesInstance.claim(userTokenId);
        vm.stopPrank();
    }

    //tests that batchClaim can claim $VVV for multiple nodes
    function testBatchClaim() public {
        vm.deal(address(NodesInstance), type(uint128).max);

        uint256 nodesToMint = 12;
        vm.deal(sampleUser, activationThreshold * nodesToMint);

        uint256 vestingDuration = secondsInTwoYears;
        uint256[] memory tokenIds = new uint256[](nodesToMint);

        //mint nodesToMint nodes and stake activationThreshold for each. i is tokenId.
        for (uint256 i = 1; i <= nodesToMint; ++i) {
            vm.startPrank(deployer, deployer);
            NodesInstance.adminMint(sampleUser, sampleLockedTokens);
            vm.stopPrank();

            //stake the activation threshold to activate the node
            vm.startPrank(sampleUser, sampleUser);
            NodesInstance.stake{ value: activationThreshold }(i);
            vm.stopPrank();
        }

        //wait for the vesting period to pass
        advanceBlockNumberAndTimestampInSeconds(vestingDuration + 1);

        //calls batchClaim to claim for both nodes
        for (uint256 i = 0; i < nodesToMint; ++i) {
            tokenIds[i] = i + 1;
        }
        vm.startPrank(sampleUser, sampleUser);
        NodesInstance.batchClaim(tokenIds);
        vm.stopPrank();

        //assert that the user has correct amount of $VVV claimed from all nodes
        (, , , , uint256 vestedPerSecond, ) = NodesInstance.tokenData(1);
        uint256 referenceClaimedAmount = nodesToMint * vestedPerSecond * NodesInstance.VESTING_DURATION();

        assertEq(sampleUser.balance, referenceClaimedAmount);
    }

    //tests batchClaim where a user does not own one of the nodes
    function testBatchClaimUnownedNode() public {
        vm.deal(address(NodesInstance), type(uint128).max);
        vm.deal(deployer, type(uint256).max);

        uint256 nodesToMint = 12;
        vm.deal(sampleUser, activationThreshold * nodesToMint);

        uint256 vestingDuration = secondsInTwoYears;
        uint256[] memory tokenIds = new uint256[](nodesToMint);

        //mint nodesToMint nodes and stake activationThreshold for each. i is tokenId.
        for (uint256 i = 1; i <= nodesToMint; ++i) {
            vm.startPrank(deployer, deployer);
            NodesInstance.adminMint(sampleUser, sampleLockedTokens);
            vm.stopPrank();

            //stake the activation threshold to activate the node
            vm.startPrank(sampleUser, sampleUser);
            NodesInstance.stake{ value: activationThreshold }(i);
            vm.stopPrank();
        }

        //set soulbound to false to allow transfer
        vm.startPrank(deployer, deployer);
        NodesInstance.setSoulbound(false);
        vm.stopPrank();

        //transfer one token from sampleUser to deployer and activate it
        vm.startPrank(sampleUser, sampleUser);
        NodesInstance.transferFrom(sampleUser, deployer, 1);
        vm.stopPrank();

        vm.startPrank(deployer, deployer);
        NodesInstance.stake{ value: activationThreshold }(1);
        vm.stopPrank();

        //wait for the vesting period to pass
        advanceBlockNumberAndTimestampInSeconds(vestingDuration + 1);

        //calls batchClaim to claim for both nodes
        for (uint256 i = 0; i < nodesToMint; ++i) {
            tokenIds[i] = i + 1;
        }

        vm.expectRevert(VVVNodes.CallerIsNotTokenOwner.selector);
        NodesInstance.batchClaim(tokenIds);
        vm.stopPrank();
    }

    //tests that a user cannot batch claim when one of the nodes has zero claimable tokens
    function testBatchClaimZero() public {
        vm.deal(address(NodesInstance), type(uint128).max);
        vm.deal(deployer, type(uint256).max);

        uint256 nodesToMint = 12;
        vm.deal(sampleUser, activationThreshold * nodesToMint);

        uint256 vestingDuration = secondsInTwoYears;
        uint256[] memory tokenIds = new uint256[](nodesToMint);

        //mint nodesToMint nodes and stake activationThreshold for each. i is tokenId.
        for (uint256 i = 1; i <= nodesToMint; ++i) {
            vm.startPrank(deployer, deployer);
            NodesInstance.adminMint(sampleUser, sampleLockedTokens);
            vm.stopPrank();

            //stake the activation threshold to activate the node
            if (i != 2) {
                vm.startPrank(sampleUser, sampleUser);
                NodesInstance.stake{ value: activationThreshold }(i);
                vm.stopPrank();
            }
        }

        //wait for the vesting period to pass
        advanceBlockNumberAndTimestampInSeconds(vestingDuration + 1);

        //calls batchClaim to claim for both nodes
        for (uint256 i = 0; i < nodesToMint; ++i) {
            tokenIds[i] = i + 1;
        }

        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVNodes.NoClaimableTokens.selector);
        NodesInstance.batchClaim(tokenIds);
        vm.stopPrank();
    }

    //tests that an admin can deposit launchpad yield for max supply of nodes
    function testDepositLaunchpadYield() public {
        uint256 nodesToMint = NodesInstance.TOTAL_SUPPLY();
        uint256[] memory tokenIds = new uint256[](nodesToMint);
        uint256[] memory amounts = new uint256[](nodesToMint);
        address[] memory nodeOwners = new address[](nodesToMint);
        uint256 amountsSum;

        vm.startPrank(deployer, deployer);
        for (uint256 i = 0; i < nodesToMint; ++i) {
            address thisNodeOwner = address(uint160(uint256(keccak256(abi.encodePacked(i)))));
            NodesInstance.adminMint(thisNodeOwner, sampleLockedTokens);

            tokenIds[i] = i + 1;
            amounts[i] = (uint256(keccak256(abi.encodePacked(i))) % 10 ether) + 1;
            amountsSum += amounts[i];
            nodeOwners[i] = thisNodeOwner;
        }

        vm.deal(deployer, amountsSum);

        NodesInstance.depositLaunchpadYield{ value: amountsSum }(tokenIds, amounts);
        vm.stopPrank();

        for (uint256 i = 0; i < nodesToMint; ++i) {
            (, , , uint256 claimableAmount, , ) = NodesInstance.tokenData(i + 1);
            assertEq(claimableAmount, amounts[i]);
        }
    }

    //tests that a non-admin can't call depositLaunchpadYield
    function testNonAdminCannotDepositLaunchpadYield() public {
        uint256 nodesToMint = 1;
        uint256[] memory tokenIds = new uint256[](nodesToMint);
        uint256[] memory amounts = new uint256[](nodesToMint);

        vm.startPrank(sampleUser, sampleUser);
        //permission errors will be first-encountered, so no need to have working values, arrays, etc.
        vm.expectRevert(VVVAuthorizationRegistryChecker.UnauthorizedCaller.selector);
        NodesInstance.depositLaunchpadYield(tokenIds, amounts);
        vm.stopPrank();
    }

    //tests that if an admin attempts to deposit launchpad yield with unequal array lengths, the call reverts
    function testDepositLaunchpadYieldUnequalArrayLengths() public {
        uint256 nodesToMint = 1;
        uint256[] memory tokenIds = new uint256[](nodesToMint);
        uint256[] memory amounts = new uint256[](nodesToMint + 1);

        vm.startPrank(deployer, deployer);
        vm.expectRevert(VVVNodes.ArrayLengthMismatch.selector);
        NodesInstance.depositLaunchpadYield(tokenIds, amounts);
        vm.stopPrank();
    }

    //tests that if an admin attempts to deposit launchpad yield with zero msg.value, the call reverts
    function testDepositLaunchpadYieldZeroValue() public {
        uint256 nodesToMint = 1;
        uint256[] memory tokenIds = new uint256[](nodesToMint);
        uint256[] memory amounts = new uint256[](nodesToMint);

        vm.startPrank(deployer, deployer);
        vm.expectRevert(VVVNodes.ZeroTokenTransfer.selector);
        NodesInstance.depositLaunchpadYield(tokenIds, amounts);
        vm.stopPrank();
    }

    //tests that if an admin attempts to deposit launchpad yield for an unminted token id, the call reverts
    function testDepositLaunchpadYieldUnmintedTokenId() public {
        //one less than the total supply now
        uint256 nodesToMint = NodesInstance.TOTAL_SUPPLY();

        uint256[] memory tokenIds = new uint256[](nodesToMint);
        uint256[] memory amounts = new uint256[](nodesToMint);
        address[] memory nodeOwners = new address[](nodesToMint);
        uint256 amountsSum;

        vm.startPrank(deployer, deployer);
        for (uint256 i = 0; i < nodesToMint; ++i) {
            address thisNodeOwner = address(uint160(uint256(keccak256(abi.encodePacked(i)))));

            //don't mint the last token, despite assigning more claimable tokens
            if (i != nodesToMint - 1) {
                NodesInstance.adminMint(thisNodeOwner, sampleLockedTokens);
            }

            tokenIds[i] = i + 1;
            amounts[i] = (uint256(keccak256(abi.encodePacked(i))) % 10 ether) + 1;
            amountsSum += amounts[i];
            nodeOwners[i] = thisNodeOwner;
        }

        vm.deal(deployer, amountsSum);

        vm.expectRevert(abi.encodeWithSelector(VVVNodes.UnmintedTokenId.selector, nodesToMint));
        NodesInstance.depositLaunchpadYield{ value: amountsSum }(tokenIds, amounts);
        vm.stopPrank();
    }

    //tests that if an admin attempts to deposit launchpad yield with msg.value != sum of amounts input, the call reverts
    function testDepositLaunchpadYieldUnequalValue() public {
        uint256 nodesToMint = 1;
        uint256[] memory tokenIds = new uint256[](nodesToMint);
        uint256[] memory amounts = new uint256[](nodesToMint);
        tokenIds[0] = 1;

        vm.deal(deployer, 1);

        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(sampleUser, sampleLockedTokens);
        vm.expectRevert(VVVNodes.MsgValueDistAmountMismatch.selector);
        NodesInstance.depositLaunchpadYield{ value: 1 }(tokenIds, amounts);
        vm.stopPrank();
    }

    //tests that an admin can unlock transaction processing yield
    function testUnlockTransactionProcessingYield() public {
        uint256 nodesToMint = NodesInstance.TOTAL_SUPPLY();
        uint256[] memory tokenIds = new uint256[](nodesToMint);
        address[] memory nodeOwners = new address[](nodesToMint);

        vm.startPrank(deployer, deployer);
        for (uint256 i = 0; i < nodesToMint; ++i) {
            address thisNodeOwner = address(uint160(uint256(keccak256(abi.encodePacked(i)))));
            NodesInstance.adminMint(thisNodeOwner, sampleLockedTokens);

            tokenIds[i] = i + 1;
            nodeOwners[i] = thisNodeOwner;
        }

        //the locked amount per node is the same, and assigned at mint, so is equal to the amount to unlock per node if unlocking the full amount
        (, , uint256 amountToUnlockPerNode, , , ) = NodesInstance.tokenData(1);

        //deal the amount to unlock per node * nodesToMint to the contract to ensure all can claim
        vm.deal(address(NodesInstance), amountToUnlockPerNode * nodesToMint);

        vm.expectEmit(address(NodesInstance));
        for (uint256 i = 0; i < nodesToMint; ++i) {
            emit VVVNodes.UnlockTransactionProcessingYield(i + 1, amountToUnlockPerNode);
        }
        NodesInstance.unlockTransactionProcessingYield(tokenIds, amountToUnlockPerNode);
        vm.stopPrank();

        for (uint256 i = 0; i < nodesToMint; ++i) {
            (, , uint256 lockedTransactionProcessingYield, uint256 claimableAmount, , ) = NodesInstance
                .tokenData(i + 1);
            assertEq(claimableAmount, amountToUnlockPerNode);
            assertEq(lockedTransactionProcessingYield, 0);
        }
    }

    //tests that a non-admin cannot unlock transaction processing yield
    function testNonAdminCannotUnlockTransactionProcessingYield() public {
        uint256 amountToUnlockPerNode;
        uint256 nodesToMint = 1;
        uint256[] memory tokenIds = new uint256[](nodesToMint);

        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVAuthorizationRegistryChecker.UnauthorizedCaller.selector);
        NodesInstance.unlockTransactionProcessingYield(tokenIds, amountToUnlockPerNode);
        vm.stopPrank();
    }

    //tests that tx processing yield cannot be unlocked for an unminted token id
    function testUnlockTransactionProcessingYieldUnmintedTokenId() public {
        uint256 amountToUnlockPerNode;
        uint256 nodesToMint = 1;
        uint256[] memory tokenIds = new uint256[](nodesToMint);
        tokenIds[0] = 1;

        vm.startPrank(deployer, deployer);
        vm.expectRevert(abi.encodeWithSelector(VVVNodes.UnmintedTokenId.selector, nodesToMint));
        NodesInstance.unlockTransactionProcessingYield(tokenIds, amountToUnlockPerNode);
        vm.stopPrank();
    }

    //tests that unlockTransactionProcessingYield if trying to unlock yield in a
    //token with 0 unlockable yield
    function testUnlockTransactionProcessingYieldZeroUnlockable() public {
        uint256 nodesToMint = 1;
        uint256[] memory tokenIds = new uint256[](nodesToMint);
        tokenIds[0] = 1;

        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(sampleUser, sampleLockedTokens);

        //unlock all transaction processing yield
        (, , uint256 amountToUnlock, , , ) = NodesInstance.tokenData(1);
        NodesInstance.unlockTransactionProcessingYield(tokenIds, amountToUnlock);

        vm.expectRevert(abi.encodeWithSelector(VVVNodes.NoRemainingUnlockableYield.selector, tokenIds[0]));
        //attempt to unlock one more token of yield
        NodesInstance.unlockTransactionProcessingYield(tokenIds, 1);
        vm.stopPrank();
    }

    //tests that a view function can output whether a token of tokenId is active
    function testCheckNodeIsActive() public {
        vm.deal(sampleUser, activationThreshold);
        uint256 tokenId = 1;

        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(sampleUser, sampleLockedTokens);
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        NodesInstance.stake{ value: activationThreshold }(tokenId);
        vm.stopPrank();

        assertTrue(NodesInstance.isNodeActive(tokenId));
    }

    //tests that minting works, but transfers do not when tokens are soulbound
    function testTransferFailsWhenSoulbound() public {
        uint256 tokenId = 1;

        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(sampleUser, sampleLockedTokens);
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVNodes.NodesAreSoulbound.selector);
        NodesInstance.transferFrom(sampleUser, deployer, tokenId);
        vm.stopPrank();
    }

    //tests that after setting soulbound=false, transfers work normally
    function testTransferSucceedsAfterSoulboundSetToFalse() public {
        uint256 tokenId = 1;

        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(sampleUser, sampleLockedTokens);
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVNodes.NodesAreSoulbound.selector);
        NodesInstance.transferFrom(sampleUser, deployer, tokenId);
        vm.stopPrank();

        vm.startPrank(deployer, deployer);
        NodesInstance.setSoulbound(false);
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        NodesInstance.transferFrom(sampleUser, deployer, tokenId);
        vm.stopPrank();

        assertEq(NodesInstance.balanceOf(deployer), 1);
    }

    //tests that an admin can set activationThreshold
    function testSetActivationThreshold() public {
        uint256 currentActivationThreshold = NodesInstance.activationThreshold();
        uint256 newActivationThreshold = currentActivationThreshold + 1;

        vm.startPrank(deployer, deployer);
        NodesInstance.setActivationThreshold(newActivationThreshold);
        vm.stopPrank();
        assertEq(NodesInstance.activationThreshold(), newActivationThreshold);
    }

    //tests that a non-admin cannot set activation threshold
    function testNonAdminCannotSetActivationThreshold() public {
        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVAuthorizationRegistryChecker.UnauthorizedCaller.selector);
        NodesInstance.setActivationThreshold(activationThreshold + 1);
        vm.stopPrank();
    }

    //tests that an active node becomes inactive when an admin increases activationThreshold
    function testNodeInactivationOnThresholdIncrease() public {
        vm.deal(sampleUser, activationThreshold);
        uint256 tokenId = 1;

        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(sampleUser, sampleLockedTokens);
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        bool isActiveBeforeStake = NodesInstance.isNodeActive(tokenId);
        NodesInstance.stake{ value: activationThreshold }(tokenId);
        bool isActiveAfterStake = NodesInstance.isNodeActive(tokenId);
        vm.stopPrank();

        vm.startPrank(deployer, deployer);
        NodesInstance.setActivationThreshold(activationThreshold + 1);
        vm.stopPrank();

        bool isActiveAfterThresholdChange = NodesInstance.isNodeActive(tokenId);

        assertFalse(isActiveBeforeStake);
        assertTrue(isActiveAfterStake);
        assertFalse(isActiveAfterThresholdChange);
    }

    //tests that a threshold decrease which would activate a previously inactive node
    function testNodeActivationOnThresholdDecrease() public {
        vm.deal(sampleUser, activationThreshold);
        uint256 tokenId = 1;
        uint256 newThreshold = activationThreshold - 1;

        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(sampleUser, sampleLockedTokens);
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        bool isActiveBeforeStake = NodesInstance.isNodeActive(tokenId);
        //stake one less than the activation threshold so the node does not activate
        NodesInstance.stake{ value: newThreshold }(tokenId);
        bool isActiveAfterStake = NodesInstance.isNodeActive(tokenId);
        vm.stopPrank();

        vm.startPrank(deployer, deployer);
        NodesInstance.setActivationThreshold(newThreshold);
        vm.stopPrank();

        bool isActiveAfterThresholdChange = NodesInstance.isNodeActive(tokenId);

        assertFalse(isActiveBeforeStake);
        assertFalse(isActiveAfterStake);
        assertTrue(isActiveAfterThresholdChange);
    }

    //tests that a defined gas limit is not exceeded when calling setActivationThreshold in the worst possible conditions
    function testGasConsumedBySetActivationThreshold() public {
        uint256 vestingDuration = secondsInTwoYears;
        uint256 totalGasLimit = type(uint256).max;
        vm.deal(sampleUser, type(uint256).max);

        //mint 5000 nodes
        uint256 nodesToMint = 5000;
        vm.startPrank(deployer, deployer);
        for (uint256 i = 0; i < nodesToMint; i++) {
            NodesInstance.adminMint(sampleUser, sampleLockedTokens);
        }

        //activate all nodes
        vm.startPrank(sampleUser);
        for (uint256 i = 0; i < nodesToMint; i++) {
            NodesInstance.stake{ value: activationThreshold }(i + 1);
        }
        vm.stopPrank();

        //wait for all tokens to vest
        advanceBlockNumberAndTimestampInSeconds(vestingDuration * 2);

        vm.startPrank(deployer, deployer);
        uint256 gasLeftBeforeThresholdChange = gasleft();
        NodesInstance.setActivationThreshold(type(uint256).max);
        uint256 gasLeftAfterThresholdChange = gasleft();
        vm.stopPrank();

        uint256 gasUsed = gasLeftBeforeThresholdChange - gasLeftAfterThresholdChange;

        assertLt(gasUsed, totalGasLimit);
    }

    //tests that an admin can set maxLaunchpadStakeAmount
    function testSetMaxLaunchpadStakeAmount() public {
        uint256 currentMaxLaunchpadStakeAmount = NodesInstance.maxLaunchpadStakeAmount();
        uint256 newMaxLaunchpadStakeAmount = currentMaxLaunchpadStakeAmount + 1;

        vm.startPrank(deployer, deployer);
        NodesInstance.setMaxLaunchpadStakeAmount(newMaxLaunchpadStakeAmount);
        vm.stopPrank();
        assertEq(NodesInstance.maxLaunchpadStakeAmount(), newMaxLaunchpadStakeAmount);
    }

    //tests that a non-admin cannot set maxLaunchpadStakeAmount
    function testNonAdminCannotSetMaxLaunchpadStakeAmount() public {
        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVAuthorizationRegistryChecker.UnauthorizedCaller.selector);
        NodesInstance.setMaxLaunchpadStakeAmount(activationThreshold + 1);
        vm.stopPrank();
    }

    //tests that an admin can set soulbound
    function testSetSoulbound() public {
        bool currentSoulbound = NodesInstance.soulbound();
        bool newSoulbound = !currentSoulbound;

        vm.startPrank(deployer, deployer);
        NodesInstance.setSoulbound(newSoulbound);
        vm.stopPrank();
        assertEq(NodesInstance.soulbound(), newSoulbound);
    }

    //tests that a non-admin cannot set soulbound
    function testNonAdminCannotSetSoulbound() public {
        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVAuthorizationRegistryChecker.UnauthorizedCaller.selector);
        NodesInstance.setSoulbound(true);
        vm.stopPrank();
    }

    //tests that an admin can withdraw $VVV from the contract
    function testWithdrawVVVVNative() public {
        uint256 amountToWithdraw = 100 * 1e18;
        vm.deal(address(NodesInstance), amountToWithdraw);

        vm.startPrank(deployer, deployer);
        NodesInstance.withdraw(amountToWithdraw);
        vm.stopPrank();

        assertEq(address(NodesInstance).balance, 0);
    }

    //tests that a non-admin cannot withdraw $VVV
    function testNonAdminCannotWithdrawVVV() public {
        uint256 amountToWithdraw = 100 * 1e18;
        vm.deal(address(NodesInstance), amountToWithdraw);

        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVAuthorizationRegistryChecker.UnauthorizedCaller.selector);
        NodesInstance.withdraw(1);
        vm.stopPrank();
    }

    //tests that the SetActivationThreshold event is emitted when the activation threshold is set
    function testSetActivationThresholdEvent() public {
        uint256 currentActivationThreshold = NodesInstance.activationThreshold();
        uint256 newActivationThreshold = currentActivationThreshold + 1;

        vm.startPrank(deployer, deployer);
        vm.expectEmit(address(NodesInstance));
        emit VVVNodes.SetActivationThreshold(newActivationThreshold);
        NodesInstance.setActivationThreshold(newActivationThreshold);
        vm.stopPrank();

        assertEq(NodesInstance.activationThreshold(), newActivationThreshold);
    }

    //tests that the Stake event is emitted when $VVV is staked
    function testEmitStakeEvent() public {
        vm.deal(sampleUser, activationThreshold);

        uint256 tokenId = 1;

        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(sampleUser, sampleLockedTokens); //mints tokenId 1
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        vm.expectEmit(address(NodesInstance));
        emit VVVNodes.Stake(tokenId, activationThreshold);
        NodesInstance.stake{ value: activationThreshold }(tokenId);
        vm.stopPrank();
    }

    //tests that the Unstake event is emitted when $VVV is unstaked
    function testEmitUnstakeEvent() public {
        vm.deal(sampleUser, activationThreshold);

        uint256 tokenId = 1;
        uint256 amountToUnstake = 1;

        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(sampleUser, sampleLockedTokens); //mints tokenId 1
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        NodesInstance.stake{ value: activationThreshold }(tokenId);
        vm.expectEmit(address(NodesInstance));
        emit VVVNodes.Unstake(tokenId, amountToUnstake);
        NodesInstance.unstake(tokenId, amountToUnstake);
        vm.stopPrank();
    }

    //tests that VestingSinceUpdated is emitted properly if a node claims accrued yield
    function testEmitVestingSinceUpdated_OnClaim() public {
        vm.deal(sampleUser, activationThreshold);
        vm.deal(address(NodesInstance), type(uint128).max);

        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(sampleUser, sampleLockedTokens); //mints token of ID 1
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        uint256 userTokenId = 1;

        uint256 vestingDuration = secondsInTwoYears;

        //stake the activation threshold to activate the node
        NodesInstance.stake{ value: activationThreshold }(userTokenId);

        advanceBlockNumberAndTimestampInSeconds(vestingDuration * 2);

        vm.expectEmit(address(NodesInstance));
        emit VVVNodes.VestingSinceUpdated(userTokenId, block.timestamp);
        NodesInstance.claim(userTokenId);
    }

    //tests that VestingSinceUpdated is emitted properly if a stake activates a node
    function testEmitVestingSinceUpdated_StakeActivatesNode() public {
        vm.deal(sampleUser, activationThreshold);
        uint256 tokenId = 1;

        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(sampleUser, sampleLockedTokens); //mints tokenId 1
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        vm.expectEmit(address(NodesInstance));
        emit VVVNodes.VestingSinceUpdated(tokenId, block.timestamp);
        NodesInstance.stake{ value: activationThreshold }(1);
        vm.stopPrank();

        assertTrue(NodesInstance.isNodeActive(tokenId));
    }

    //tests that VestingSinceUpdated is emitted properly if a threshold change activates nodes
    function testEmitVestingSinceUpdated_ThresholdChangeActivatesNode() public {
        uint256 tokensToMint = 50;
        vm.deal(sampleUser, tokensToMint * activationThreshold);
        uint256 newThreshold = activationThreshold - 1;

        vm.startPrank(deployer, deployer);
        for (uint256 i = 0; i < tokensToMint; i++) {
            NodesInstance.adminMint(sampleUser, sampleLockedTokens);
        }
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        for (uint256 i = 0; i < tokensToMint; i++) {
            assertFalse(NodesInstance.isNodeActive(i + 1));
        }

        //stake one less than the activation threshold so the node does not activate
        for (uint256 i = 0; i < tokensToMint; i++) {
            NodesInstance.stake{ value: newThreshold }(i + 1);
            assertFalse(NodesInstance.isNodeActive(i + 1));
        }
        vm.stopPrank();

        vm.startPrank(deployer, deployer);
        for (uint256 i = 0; i < tokensToMint; i++) {
            vm.expectEmit(address(NodesInstance));
            emit VVVNodes.VestingSinceUpdated(i + 1, block.timestamp);
        }
        vm.expectEmit(address(NodesInstance));
        emit VVVNodes.SetActivationThreshold(newThreshold);
        NodesInstance.setActivationThreshold(newThreshold);
        vm.stopPrank();

        for (uint256 i = 0; i < tokensToMint; i++) {
            assertTrue(NodesInstance.isNodeActive(i + 1));
        }
    }

    //tests that the Claim event is emitted when accrued yield is claimed
    function testEmitClaim() public {
        vm.deal(sampleUser, activationThreshold);
        vm.deal(address(NodesInstance), type(uint128).max);

        vm.startPrank(deployer, deployer);
        NodesInstance.adminMint(sampleUser, sampleLockedTokens); //mints token of ID 1
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        uint256 userTokenId = 1;
        uint256 refTotalVestedTokens = (sampleLockedTokens * 60) / 100;
        uint256 vestingDuration = secondsInTwoYears;

        //stake the activation threshold to activate the node
        NodesInstance.stake{ value: activationThreshold }(userTokenId);

        advanceBlockNumberAndTimestampInSeconds(vestingDuration * 2);

        vm.expectEmit(address(NodesInstance));
        emit VVVNodes.Claim(userTokenId, refTotalVestedTokens);
        NodesInstance.claim(userTokenId);
    }

    //tests that the DepositLaunchpadYield is emitted when the launchpad yield is deposited
    function testEmitDepositLaunchpadYield() public {
        uint256 nodesToMint = NodesInstance.TOTAL_SUPPLY();
        uint256[] memory tokenIds = new uint256[](nodesToMint);
        uint256[] memory amounts = new uint256[](nodesToMint);
        uint256 amountsSum;

        vm.startPrank(deployer, deployer);
        for (uint256 i = 0; i < nodesToMint; ++i) {
            address thisNodeOwner = address(uint160(uint256(keccak256(abi.encodePacked(i)))));
            NodesInstance.adminMint(thisNodeOwner, sampleLockedTokens);

            tokenIds[i] = i + 1;
            amounts[i] = (uint256(keccak256(abi.encodePacked(i))) % 10 ether) + 1;
            amountsSum += amounts[i];
        }

        vm.deal(deployer, amountsSum);
        for (uint256 i = 0; i < nodesToMint; ++i) {
            vm.expectEmit(address(NodesInstance));
            emit VVVNodes.DepositLaunchpadYield(tokenIds[i], amounts[i]);
        }
        NodesInstance.depositLaunchpadYield{ value: amountsSum }(tokenIds, amounts);
    }
}
