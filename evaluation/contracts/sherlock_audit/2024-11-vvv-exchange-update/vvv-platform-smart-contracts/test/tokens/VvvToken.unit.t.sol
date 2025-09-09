//SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import { VVVTokenTestBase } from "./VvvTokenTestBase.sol";
import { VVVToken } from "contracts/tokens/VvvToken.sol";
import { VVVAuthorizationRegistry } from "contracts/auth/VVVAuthorizationRegistry.sol";
import { VVVAuthorizationRegistryChecker } from "contracts/auth/VVVAuthorizationRegistryChecker.sol";

contract VVVTokenUInitTests is VVVTokenTestBase {
    function setUp() public {
        vm.startPrank(deployer, deployer);

        AuthRegistry = new VVVAuthorizationRegistry(defaultAdminTransferDelay, deployer);
        vvvToken = new VVVToken(cap, initialSupply, address(AuthRegistry));

        //set auth registry permissions for tokenMintManager (TOKEN_MINT_MANAGER_ROLE)
        AuthRegistry.grantRole(tokenMintManagerRole, tokenMintManager);
        bytes4 mintSelector = vvvToken.mint.selector;
        AuthRegistry.setPermission(address(vvvToken), mintSelector, tokenMintManagerRole);

        vm.stopPrank();
    }

    /**
     * @dev deployment test
     */
    function testDeployment() public {
        assertTrue(address(vvvToken) != address(0));
    }

    /**
     * @dev test initial supply
     */
    function testInitialSupply() public {
        uint256 expected = initialSupply;
        uint256 actual = vvvToken.totalSupply();

        assertEq(actual, expected);
    }

    /**
     * @dev test cap
     */
    function testCap() public {
        uint256 expected = cap;
        uint256 actual = vvvToken.cap();

        assertEq(actual, expected);
    }

    /**
     * @dev test minting with tokenMintManager role
     */
    function testMint() public {
        uint256 amount = 1000;
        uint256 expected = vvvToken.totalSupply() + amount;

        vm.startPrank(tokenMintManager, tokenMintManager);
        vvvToken.mint(address(this), amount);
        vm.stopPrank();

        uint256 actual = vvvToken.totalSupply();

        assertEq(actual, expected);
    }

    /**
     * @dev test unauthorized minting is not possible
     */
    function testMintUnauthorized() public {
        uint256 amount = 1000;

        vm.startPrank(deployer, deployer);
        vm.expectRevert(VVVAuthorizationRegistryChecker.UnauthorizedCaller.selector);
        vvvToken.mint(address(this), amount);
        vm.stopPrank();
    }
}
