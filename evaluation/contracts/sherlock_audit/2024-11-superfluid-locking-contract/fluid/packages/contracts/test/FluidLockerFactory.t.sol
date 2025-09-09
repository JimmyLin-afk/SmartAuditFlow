// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import { SFTest } from "./SFTest.t.sol";

import {
    ISuperToken,
    ISuperfluidPool
} from "@superfluid-finance/ethereum-contracts/contracts/interfaces/superfluid/ISuperfluid.sol";
import { SuperTokenV1Library } from "@superfluid-finance/ethereum-contracts/contracts/apps/SuperTokenV1Library.sol";
import { IFluidLockerFactory } from "../src/FluidLockerFactory.sol";

using SuperTokenV1Library for ISuperToken;

contract FluidLockerFactoryTest is SFTest {
    function setUp() public override {
        super.setUp();
    }

    function testCreateLockerContract(address _user) external {
        vm.assume(_user != address(0));

        assertEq(_fluidLockerFactory.getLockerAddress(_user), address(0), "locker should not exists");

        vm.prank(_user);
        address userLockerAddress = _fluidLockerFactory.createLockerContract();

        assertEq(_fluidLockerFactory.getLockerAddress(_user), userLockerAddress, "locker should exists");

        vm.prank(_user);
        vm.expectRevert();
        _fluidLockerFactory.createLockerContract();
    }

    function testSetGovernor(address _newGovernor) external {
        address currentGovernor = _fluidLockerFactory.governor();
        vm.assume(_newGovernor != currentGovernor);
        vm.assume(_newGovernor != address(0));

        vm.prank(_newGovernor);
        vm.expectRevert(IFluidLockerFactory.NOT_GOVERNOR.selector);
        _fluidLockerFactory.setGovernor(_newGovernor);

        vm.prank(currentGovernor);
        _fluidLockerFactory.setGovernor(_newGovernor);

        assertEq(_fluidLockerFactory.governor(), _newGovernor, "governor not updated");
    }

    function testGetUserLocker(address user, address nonUser) external {
        vm.assume(user != nonUser);

        vm.prank(user);
        address userLockerAddress = _fluidLockerFactory.createLockerContract();

        (bool isCreated, address lockerAddressResult) = _fluidLockerFactory.getUserLocker(user);

        assertEq(lockerAddressResult, userLockerAddress, "incorrect address");
        assertEq(isCreated, true, "locker should be created");

        (isCreated, lockerAddressResult) = _fluidLockerFactory.getUserLocker(nonUser);
        assertEq(lockerAddressResult, address(0), "should be the zero-address");
        assertEq(isCreated, false, "locker should not be created");
    }

    function testGetLockerBeaconImplementation() external view {
        assertEq(_fluidLockerFactory.getLockerBeaconImplementation(), address(_fluidLockerLogic));
    }
}
