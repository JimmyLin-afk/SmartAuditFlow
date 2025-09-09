//SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import { MockERC20 } from "contracts/mock/MockERC20.sol";
import { Ownable } from "@openzeppelin/contracts/access/Ownable.sol";
import { VVVAuthorizationRegistry } from "contracts/auth/VVVAuthorizationRegistry.sol";
import { VVVAuthorizationRegistryChecker } from "contracts/auth/VVVAuthorizationRegistryChecker.sol";
import { VVVVCInvestmentLedger } from "contracts/vc/VVVVCInvestmentLedger.sol";
import { VVVVCTokenDistributor } from "contracts/vc/VVVVCTokenDistributor.sol";
import { VVVVCTestBase } from "test/vc/VVVVCTestBase.sol";

/**
 * @title VVVVCTokenDistributor Unit Tests
 * @dev use "forge test --match-contract VVVVCTokenDistributorUnitTests" to run tests
 * @dev use "forge coverage --match-contract VVVVCTokenDistributor" to run coverage
 */
contract VVVVCTokenDistributorUnitTests is VVVVCTestBase {
    function setUp() public {
        vm.startPrank(deployer, deployer);

        AuthRegistry = new VVVAuthorizationRegistry(defaultAdminTransferDelay, deployer);

        TokenDistributorInstance = new VVVVCTokenDistributor(
            testSigner,
            environmentTag,
            address(AuthRegistry)
        );

        AuthRegistry.grantRole(tokenDistributorManagerRole, tokenDistributorManager);
        bytes4 setClaimPausedSelector = TokenDistributorInstance.setClaimIsPaused.selector;
        AuthRegistry.setPermission(
            address(TokenDistributorInstance),
            setClaimPausedSelector,
            tokenDistributorManagerRole
        );

        distributorDomainSeparator = TokenDistributorInstance.DOMAIN_SEPARATOR();
        claimTypehash = TokenDistributorInstance.CLAIM_TYPEHASH();

        ProjectTokenInstance = new MockERC20(18);

        vm.stopPrank();

        //supply the proxy wallets with the project token to be withdrawn from them by investors
        //from each wallet, approve the token distributor to withdraw the project token
        for (uint256 i = 0; i < projectTokenProxyWallets.length; i++) {
            ProjectTokenInstance.mint(projectTokenProxyWallets[i], projectTokenAmountToProxyWallet);
            vm.startPrank(projectTokenProxyWallets[i], projectTokenProxyWallets[i]);
            ProjectTokenInstance.approve(
                address(TokenDistributorInstance),
                projectTokenAmountToProxyWallet
            );
            vm.stopPrank();
        }
    }

    function testDeployment() public {
        assertTrue(address(TokenDistributorInstance) != address(0));
    }

    // ensure claims are not paused by default
    function testClaimsAreNotPausedByDefault() public {
        assertFalse(TokenDistributorInstance.claimIsPaused());
    }

    function testValidateSignature() public {
        VVVVCTokenDistributor.ClaimParams memory claimParams = generateClaimParamsWithSignature(
            sampleKycAddress,
            projectTokenProxyWallets,
            sampleTokenAmountsToClaim
        );

        assertTrue(TokenDistributorInstance.isSignatureValid(claimParams));
    }

    function testInvalidateSignature() public {
        VVVVCTokenDistributor.ClaimParams memory claimParams = generateClaimParamsWithSignature(
            sampleKycAddress,
            projectTokenProxyWallets,
            sampleTokenAmountsToClaim
        );

        claimParams.projectTokenProxyWallets[0] = address(0);

        assertFalse(TokenDistributorInstance.isSignatureValid(claimParams));
    }

    //test that claiming for a single round works
    function testClaimSingleRound() public {
        address[] memory thisProjectTokenProxyWallets = new address[](1);
        uint256[] memory thisTokenAmountsToClaim = new uint256[](1);

        thisProjectTokenProxyWallets[0] = projectTokenProxyWallets[0];

        uint256 claimAmount = sampleTokenAmountsToClaim[0];
        thisTokenAmountsToClaim[0] = claimAmount;

        VVVVCTokenDistributor.ClaimParams memory claimParams = generateClaimParamsWithSignature(
            sampleKycAddress,
            thisProjectTokenProxyWallets,
            thisTokenAmountsToClaim
        );

        claimAsUser(sampleKycAddress, claimParams);
        assertTrue(ProjectTokenInstance.balanceOf(sampleKycAddress) == claimAmount);
    }

    // test that zero tokens can be claimed
    function testClaimZeroAmount() public {
        address[] memory thisProjectTokenProxyWallets = new address[](1);
        uint256[] memory thisTokenAmountsToClaim = new uint256[](1);

        thisProjectTokenProxyWallets[0] = projectTokenProxyWallets[0];

        uint256 claimAmount = 0;
        thisTokenAmountsToClaim[0] = claimAmount;

        //claim for the same single round
        VVVVCTokenDistributor.ClaimParams memory claimParams = generateClaimParamsWithSignature(
            sampleKycAddress,
            thisProjectTokenProxyWallets,
            thisTokenAmountsToClaim
        );

        claimAsUser(sampleKycAddress, claimParams);
        assertTrue(ProjectTokenInstance.balanceOf(sampleKycAddress) == claimAmount);
    }

    // test claiming in multiple rounds
    function testClaimMultipleRounds() public {
        VVVVCTokenDistributor.ClaimParams memory claimParams = generateClaimParamsWithSignature(
            sampleKycAddress,
            projectTokenProxyWallets,
            sampleTokenAmountsToClaim
        );

        claimAsUser(sampleKycAddress, claimParams);
        assertTrue(ProjectTokenInstance.balanceOf(sampleKycAddress) == sum(sampleTokenAmountsToClaim));
    }

    //test that claims will succeed after unpausing claims
    function testClaimSuccessAfterUnpause() public {
        vm.startPrank(tokenDistributorManager);
        TokenDistributorInstance.setClaimIsPaused(true);
        vm.stopPrank();

        VVVVCTokenDistributor.ClaimParams memory claimParams = generateClaimParamsWithSignature(
            sampleUser,
            projectTokenProxyWallets,
            sampleTokenAmountsToClaim
        );

        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVVCTokenDistributor.ClaimIsPaused.selector);
        TokenDistributorInstance.claim(claimParams);
        vm.stopPrank();

        vm.startPrank(tokenDistributorManager);
        TokenDistributorInstance.setClaimIsPaused(false);
        vm.stopPrank();

        claimAsUser(sampleUser, claimParams);
        assertTrue(ProjectTokenInstance.balanceOf(sampleUser) == sum(sampleTokenAmountsToClaim));
    }

    // tests any claim where the signature includes a parameter value that invalidates it
    function testClaimWithInvalidSignature() public {
        VVVVCTokenDistributor.ClaimParams memory claimParams = generateClaimParamsWithSignature(
            sampleKycAddress,
            projectTokenProxyWallets,
            sampleTokenAmountsToClaim
        );

        claimParams.projectTokenAddress = address(0);
        vm.expectRevert(VVVVCTokenDistributor.InvalidSignature.selector);
        claimAsUser(sampleUser, claimParams);
    }

    // tests that invalid nonce causes revert with InvalidNonce error
    function testClaimWithInvalidNonce() public {
        VVVVCTokenDistributor.ClaimParams memory claimParams = generateClaimParamsWithSignature(
            sampleKycAddress,
            projectTokenProxyWallets,
            sampleTokenAmountsToClaim
        );

        claimParams.nonce = 0;
        vm.expectRevert(VVVVCTokenDistributor.InvalidNonce.selector);
        claimAsUser(sampleUser, claimParams);
    }

    // tests that the ArrayLengthMismatch error is thrown when the lengths of the projectTokenProxyWallets and tokenAmountsToClaim arrays do not match
    function testClaimArrayLengthMismatch() public {
        address[] memory shorterProxyWalletArray = new address[](1);
        shorterProxyWalletArray[0] = projectTokenProxyWallets[0];

        VVVVCTokenDistributor.ClaimParams memory claimParams = generateClaimParamsWithSignature(
            sampleKycAddress,
            shorterProxyWalletArray,
            sampleTokenAmountsToClaim
        );

        vm.expectRevert(VVVVCTokenDistributor.ArrayLengthMismatch.selector);
        claimAsUser(sampleUser, claimParams);
    }

    // that calling claim when claimIsPaused is true causes revert ClaimIsPaused
    function testClaimWhenPaused() public {
        VVVVCTokenDistributor.ClaimParams memory claimParams = generateClaimParamsWithSignature(
            sampleUser,
            projectTokenProxyWallets,
            sampleTokenAmountsToClaim
        );

        vm.startPrank(tokenDistributorManager);
        TokenDistributorInstance.setClaimIsPaused(true);
        vm.stopPrank();

        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVVCTokenDistributor.ClaimIsPaused.selector);
        TokenDistributorInstance.claim(claimParams);
        vm.stopPrank();
    }

    // Test that VCClaim is correctly emitted when project tokens are claimed
    function testEmitVCClaim() public {
        VVVVCTokenDistributor.ClaimParams memory claimParams = generateClaimParamsWithSignature(
            sampleKycAddress,
            projectTokenProxyWallets,
            sampleTokenAmountsToClaim
        );

        vm.startPrank(sampleUser, sampleUser);
        vm.expectEmit(address(TokenDistributorInstance));
        emit VVVVCTokenDistributor.VCClaim(
            sampleKycAddress,
            address(ProjectTokenInstance),
            projectTokenProxyWallets,
            sampleTokenAmountsToClaim,
            claimParams.nonce
        );
        TokenDistributorInstance.claim(claimParams);
        vm.stopPrank();
    }

    // test that a caller given the required role can pause and unpause claims
    function testAuthorizedCanPauseClaims() public {
        vm.startPrank(tokenDistributorManager);
        TokenDistributorInstance.setClaimIsPaused(true);
        vm.stopPrank();
        assertTrue(TokenDistributorInstance.claimIsPaused());
    }

    function testAuthorizedCanUnpauseClaims() public {
        vm.startPrank(tokenDistributorManager);
        TokenDistributorInstance.setClaimIsPaused(false);
        vm.stopPrank();
        assertFalse(TokenDistributorInstance.claimIsPaused());
    }

    // test that a caller that is not given the required role cannot pause or unpause claims
    function testUnauthorizedCannotCallSetClaimIsPaused() public {
        vm.startPrank(sampleUser, sampleUser);
        vm.expectRevert(VVVAuthorizationRegistryChecker.UnauthorizedCaller.selector);
        TokenDistributorInstance.setClaimIsPaused(false);
        vm.stopPrank();
    }
}
