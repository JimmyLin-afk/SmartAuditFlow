///SPDX-License-Identifier: MIT
pragma solidity 0.8.23;

import "lib/forge-std/src/Test.sol";
import { VVVAuthorizationRegistry } from "contracts/auth/VVVAuthorizationRegistry.sol";
import { VVVAuthorizationRegistryChecker } from "contracts/auth/VVVAuthorizationRegistryChecker.sol";
import { MockERC20 } from "contracts/mock/MockERC20_AuthRegistry.sol";

contract VVVAuthorizationRegistryTests is Test {
    VVVAuthorizationRegistry registry;
    MockERC20 mockToken;

    uint256 deployerKey = 1234;
    uint256 managerKey = 12345;
    address deployer = vm.addr(deployerKey);
    address manager = vm.addr(managerKey);

    uint48 defaultAdminTransferDelay = 3 days;
    address defaultAdmin = deployer;

    uint256 blockNumber;
    uint256 blockTimestamp;
    uint256 chainId = 31337; //test chain id - leave this alone

    function setUp() public {
        vm.startPrank(deployer, deployer);
        registry = new VVVAuthorizationRegistry(defaultAdminTransferDelay, defaultAdmin);
        mockToken = new MockERC20(18, address(registry));
        vm.stopPrank();
    }

    //helper
    function advanceBlockNumberAndTimestampInSeconds(uint256 secondsToAdvance) public {
        blockNumber += secondsToAdvance / 12; //seconds per block
        blockTimestamp += secondsToAdvance;
        vm.warp(blockTimestamp);
        vm.roll(blockNumber);
    }

    //unit tests
    //ensures contract is deployed as expected
    function testDeployment() public {
        assertTrue(address(registry) != address(0));
        assertTrue(registry.defaultAdmin() == defaultAdmin);
        assertTrue(registry.defaultAdminDelay() == defaultAdminTransferDelay);
    }

    //tests that DEFAULT_ADMIN_ROLE is transferrable per 2-step process of AccessControlDefaultAdminRules
    function testTransferDefaultAdminRole() public {
        vm.startPrank(deployer, deployer);
        registry.beginDefaultAdminTransfer(manager);
        vm.stopPrank();

        //advance delay time + 2 to accomodate time > delaytime (not >=), and the test starting at t=0
        advanceBlockNumberAndTimestampInSeconds(defaultAdminTransferDelay + 2);

        vm.startPrank(manager, manager);
        registry.acceptDefaultAdminTransfer();
        vm.stopPrank();

        assertTrue(registry.defaultAdmin() == manager);
    }

    //tests that DEFAULT_ADMIN_ROLE cannot be transferred without the delay between steps having passed
    function testCannotTransferDefaultAdminRoleBeforeDelay() public {
        vm.startPrank(deployer, deployer);
        registry.beginDefaultAdminTransfer(manager);
        vm.stopPrank();

        //advance delay time +1, 1 second before eligibility to accept transfer of DEFAULT_ADMIN_ROLE
        advanceBlockNumberAndTimestampInSeconds(defaultAdminTransferDelay + 1);

        vm.startPrank(manager, manager);
        vm.expectRevert(
            abi.encodeWithSignature("AccessControlEnforcedDefaultAdminDelay(uint48)", block.timestamp)
        );
        registry.acceptDefaultAdminTransfer();
        vm.stopPrank();
    }

    //ensures that the DEFAULT_ADMIN_ROLE can set permissions
    function testDefaultAdminSetPermission() public {
        bytes4 selector = bytes4(keccak256("test()"));
        bytes32 role = bytes32(keccak256("TEST_ROLE"));
        address contractToCall = address(this);

        vm.startPrank(deployer, deployer);
        registry.setPermission(contractToCall, selector, role);
        vm.stopPrank();

        bytes24 key = bytes24(keccak256(abi.encodePacked(contractToCall, selector)));
        assertTrue(registry.permissions(key) == role);
    }

    //ensures that the DEFAULT_ADMIN_ROLE can revoke a previously set permission by setting the role for a key to bytes32(0)
    function testDefaultAdminRevokePermission() public {
        vm.startPrank(deployer, deployer);
        bytes4 selector = bytes4(keccak256("test()"));
        bytes32 role = bytes32(keccak256("TEST_ROLE"));
        address contractToCall = address(this);

        registry.setPermission(contractToCall, selector, role);

        bytes24 key = bytes24(keccak256(abi.encodePacked(contractToCall, selector)));

        registry.setPermission(contractToCall, selector, bytes32(0));

        assertTrue(registry.permissions(key) == bytes32(0));

        vm.stopPrank();
    }

    //tests that permission cannot be set without DEFAULT_ADMIN_ROLE
    function testNonDefaultAdminGrantPermission() public {
        bytes4 selector = bytes4(keccak256("test()"));
        bytes32 role = bytes32(keccak256("TEST_ROLE"));
        address contractToCall = address(mockToken);

        vm.startPrank(manager, manager);
        vm.expectRevert(
            abi.encodeWithSignature(
                "AccessControlUnauthorizedAccount(address,bytes32)",
                manager,
                registry.DEFAULT_ADMIN_ROLE()
            )
        );
        registry.setPermission(contractToCall, selector, role);
        vm.stopPrank();
    }

    //tests that permission cannot be removed without DEFAULT_ADMIN_ROLE
    //in this context, role removal is done by setting the role to bytes32(0)
    function testNonDefaultAdminRevokePermission() public {
        bytes4 selector = bytes4(keccak256("test()"));
        bytes32 role = bytes32(0); //restoring default mapping value, only DEFAULT_ADMIN_ROLE has authorization for this key
        address contractToCall = address(mockToken);

        vm.startPrank(manager, manager);
        vm.expectRevert(
            abi.encodeWithSignature(
                "AccessControlUnauthorizedAccount(address,bytes32)",
                manager,
                registry.DEFAULT_ADMIN_ROLE()
            )
        );
        registry.setPermission(contractToCall, selector, role);
        vm.stopPrank();
    }

    //test the isAuthorized function directly to ensure MockERC20_AuthRegistry is not misusing the isAuthorized modifier
    function testIsAuthorized() public {
        vm.startPrank(deployer, deployer);
        //create permission for role
        bytes4 selector = mockToken.mintWithAuth.selector;
        bytes32 role = bytes32(keccak256("TEST_ROLE"));
        address contractToCall = address(mockToken);
        registry.setPermission(contractToCall, selector, role);
        //assign role to non-super-admin (non-deployer) address
        registry.grantRole(role, manager);
        vm.stopPrank();

        vm.startPrank(manager, manager);
        assertTrue(registry.isAuthorized(contractToCall, selector, manager));
        vm.stopPrank();
    }

    //test that isAuthorized returns false if the caller does not have the required role
    function testNotAuthorized() public {
        vm.startPrank(deployer, deployer);
        //create permission for role
        bytes4 selector = mockToken.mintWithAuth.selector;
        bytes32 role = bytes32(keccak256("TEST_ROLE"));
        address contractToCall = address(mockToken);
        registry.setPermission(contractToCall, selector, role);
        //no role granted to manager here, unlike testIsAuthorized
        vm.stopPrank();

        vm.startPrank(manager, manager);
        assertTrue(!registry.isAuthorized(contractToCall, selector, manager));
        vm.stopPrank();
    }

    //test that the onlyAuthorized modifier allows a user with a given role to call a function
    function testPermitFunctionCall() public {
        vm.startPrank(deployer, deployer);
        //create permission for role
        bytes4 selector = mockToken.mintWithAuth.selector;
        bytes32 role = bytes32(keccak256("TEST_ROLE"));
        address contractToCall = address(mockToken);
        registry.setPermission(contractToCall, selector, role);
        //assign role to non-super-admin (non-deployer) address
        registry.grantRole(role, manager);
        vm.stopPrank();

        vm.startPrank(manager, manager);
        mockToken.mintWithAuth(msg.sender, 0);
        vm.stopPrank();
    }

    //test that the same function would not be callable without having been granted the role to call it
    function testCallAuthFunctionWithoutRole() public {
        vm.startPrank(deployer, deployer);
        //create permission for role
        bytes4 selector = mockToken.mintWithAuth.selector;
        bytes32 role = bytes32(keccak256("TEST_ROLE"));
        address contractToCall = address(mockToken);
        registry.setPermission(contractToCall, selector, role);

        //no role granted here, unlike testPermitFunctionCall
        vm.stopPrank();

        // VVVAuthorizationRegistryChecker.UnauthorizedCaller error expected
        vm.startPrank(manager, manager);
        vm.expectRevert(VVVAuthorizationRegistryChecker.UnauthorizedCaller.selector);
        mockToken.mintWithAuth(msg.sender, 0);
        vm.stopPrank();
    }
}
