pragma solidity ^0.8.0;
// SPDX-License-Identifier: MIT

import "./Testable.sol";
import "./ProtocolFee_Override.sol";

import { User } from "./Test_Helpers.sol";

contract ProtocolFee_Test is Testable {
    ProtocolFee_Override protocolFee;

    User borrower;

    uint16 initialFee = 12;

    function setUp() public {
        protocolFee = new ProtocolFee_Override();
    }

    function test_initialize() public {
        protocolFee.initialize(initialFee);

        assertEq(
            protocolFee.protocolFee(),
            initialFee,
            "did not initialize fee properly"
        );
    }

    function test_setProtocolFee() public {
        protocolFee.initialize(initialFee);

        protocolFee.setProtocolFee(5);

        assertEq(
            protocolFee.protocolFee(),
            5,
            "did not initialize fee properly"
        );
    }

    function test_initialize_separately() public {
        protocolFee.protocolFeeInit(initialFee);

        assertEq(
            protocolFee.protocolFee(),
            initialFee,
            "did not initialize fee properly"
        );
    }

    function test_setProtocolFee_while_not_initializing() public {
        protocolFee.initialize(initialFee);

        vm.expectRevert("Initializable: contract is already initialized");

        protocolFee.initialize(initialFee);
    }

    function test_setProtocolFee_twice() public {
        protocolFee.initialize(initialFee);

        protocolFee.setProtocolFee(5);
        protocolFee.setProtocolFee(5);

        assertEq(
            protocolFee.protocolFee(),
            5,
            "did not initialize fee properly"
        );
    }

    function test_setProtocolFee_invalid() public {
        protocolFee.initialize(initialFee);

        vm.prank(address(borrower), address(borrower));

        vm.expectRevert("Ownable: caller is not the owner");

        protocolFee.setProtocolFee(5);
    }
}
