//SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import { MockERC20 } from "contracts/mock/MockERC20.sol";
import { IERC20Errors } from "@openzeppelin/contracts/interfaces/draft-IERC6093.sol";
import { IERC20 } from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import { Ownable } from "@openzeppelin/contracts/access/Ownable.sol";
import { VVVAuthorizationRegistry } from "contracts/auth/VVVAuthorizationRegistry.sol";
import { VVVAuthorizationRegistryChecker } from "contracts/auth/VVVAuthorizationRegistryChecker.sol";
import { VVVVCInvestmentLedger } from "contracts/vc/VVVVCInvestmentLedger.sol";
import { VVVVCTokenDistributor } from "contracts/vc/VVVVCTokenDistributor.sol";
import { VVVVCTestBase } from "test/vc/VVVVCTestBase.sol";

/**
 * @title VVVVCTokenDistributor Fuzz Tests
 * @dev use "forge test --match-contract VVVVCTokenDistributorFuzzTests" to run tests
 * @dev use "forge coverage --match-contract VVVVCTokenDistributor" to run coverage
 */
contract VVVVCTokenDistributorFuzzTests is VVVVCTestBase {
    function setUp() public {
        vm.startPrank(deployer, deployer);

        //supply the proxy wallets with the project token to be withdrawn from them by investors
        ProjectTokenInstance = new MockERC20(18);
        for (uint256 i = 0; i < projectTokenProxyWallets.length; i++) {
            ProjectTokenInstance.mint(projectTokenProxyWallets[i], projectTokenAmountToProxyWallet);
        }

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

        vm.stopPrank();
    }

    function testFuzz_ClaimSuccess(
        address _callerAddress,
        address _kycAddress,
        uint256 _seed,
        uint256 _length
    ) public {
        vm.assume(_callerAddress != address(0));
        vm.assume(_kycAddress != address(0));
        vm.assume(_seed != 0);
        vm.assume(_length != 0);

        uint256 maxLength = 100;
        uint256 arrayLength = bound(_length, 1, maxLength);

        address[] memory projectTokenProxyWallets = new address[](arrayLength);
        uint256[] memory tokenAmountsToClaim = new uint256[](arrayLength);

        uint256 totalClaimAmount = 0;

        for (uint256 i = 0; i < arrayLength; i++) {
            projectTokenProxyWallets[i] = address(
                uint160(uint256(keccak256(abi.encodePacked(_callerAddress, i))))
            );

            tokenAmountsToClaim[i] = bound(_seed, 0, 1000 * 1e18);
            totalClaimAmount += tokenAmountsToClaim[i];

            // Mint tokens to the proxy wallet and approve the distributor
            ProjectTokenInstance.mint(projectTokenProxyWallets[i], tokenAmountsToClaim[i]);
            vm.prank(projectTokenProxyWallets[i]);
            ProjectTokenInstance.approve(address(TokenDistributorInstance), tokenAmountsToClaim[i]);
        }

        uint256 balanceBefore = ProjectTokenInstance.balanceOf(_callerAddress);

        VVVVCTokenDistributor.ClaimParams memory claimParams = generateClaimParamsWithSignature(
            _kycAddress,
            projectTokenProxyWallets,
            tokenAmountsToClaim
        );

        // Attempt to claim
        claimAsUser(_callerAddress, claimParams);

        // Check if the total claimed amount matches expected
        assertTrue(ProjectTokenInstance.balanceOf(_callerAddress) == balanceBefore + totalClaimAmount);
    }

    // Ensures no claim can be made without either a valid signature, valid nonce, or sufficient token balance
    function testFuzz_ClaimRevert(
        address _callerAddress,
        address _kycAddress,
        uint256 _seed,
        uint256 _length,
        uint8 _testCase
    ) public {
        vm.assume(_callerAddress != address(0));
        vm.assume(_kycAddress != address(0));
        vm.assume(_seed != 0);

        uint256 lengthLimit = 100;
        uint256 arrayLength = bound(_length, 1, lengthLimit);

        address[] memory projectTokenProxyWallets = new address[](arrayLength);
        uint256[] memory tokenAmountsToClaim = new uint256[](arrayLength);

        for (uint256 i = 0; i < arrayLength; i++) {
            projectTokenProxyWallets[i] = address(
                uint160(uint256(keccak256(abi.encodePacked(_callerAddress, i))))
            );
            tokenAmountsToClaim[i] = bound(_seed, 1, 1000 * 1e18);
            vm.startPrank(projectTokenProxyWallets[i]);
            ProjectTokenInstance.mint(projectTokenProxyWallets[i], tokenAmountsToClaim[i]);
            ProjectTokenInstance.approve(address(TokenDistributorInstance), type(uint256).max);
            vm.stopPrank();
        }

        VVVVCTokenDistributor.ClaimParams memory claimParams = generateClaimParamsWithSignature(
            _kycAddress,
            projectTokenProxyWallets,
            tokenAmountsToClaim
        );

        uint256 testCase = _testCase % 3;

        if (testCase == 0) {
            // Invalid signature
            claimParams.tokenAmountsToClaim = new uint256[](projectTokenProxyWallets.length);
            vm.expectRevert(VVVVCTokenDistributor.InvalidSignature.selector);
        } else if (testCase == 1) {
            // Invalid nonce
            claimParams.nonce = 0;
            vm.expectRevert(VVVVCTokenDistributor.InvalidNonce.selector);
        } else {
            // Insufficient token balance
            address thisProxyWallet = projectTokenProxyWallets[0];
            uint256 defecit = 1;
            uint256 balance = ProjectTokenInstance.balanceOf(thisProxyWallet);
            vm.prank(thisProxyWallet);
            ProjectTokenInstance.transfer(address(0xdead), defecit);
            vm.expectRevert(
                abi.encodeWithSelector(
                    IERC20Errors.ERC20InsufficientBalance.selector,
                    thisProxyWallet,
                    balance - defecit,
                    balance
                )
            );
        }

        // Attempt to claim
        claimAsUser(_callerAddress, claimParams);
    }
}
