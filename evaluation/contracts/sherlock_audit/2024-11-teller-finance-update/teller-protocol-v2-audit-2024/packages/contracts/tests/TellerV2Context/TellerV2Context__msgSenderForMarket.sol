// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/utils/Address.sol";

import { Testable } from "../Testable.sol";
import { TellerV2Context_Override, TellerV2Context } from "./TellerV2Context_Override.sol";

contract TellerV2Context__msgSenderForMarket is Testable {
    using Address for address;

    TellerV2Context_Override private context;

    function setUp() public {
        context = new TellerV2Context_Override(address(0), address(111));
    }

    function test_Return_actual_caller_for_untrusted_forwarders() public {
        address expectedSender = address(this);
        address sender = context.external__msgSenderForMarket(7);

        assertEq(sender, expectedSender, "sender should be the actual caller");
    }

    function test_Revert_for_trusted_but_unapproved_forwarders() public {
        uint256 marketId = 7;

        context.mock_setTrustedMarketForwarder(marketId, address(this));

        vm.expectRevert("Sender must approve market forwarder");
        context.external__msgSenderForMarket(marketId);
    }

    function test_Return_appended_20_bytes_from_calldata() public {
        uint256 marketId = 7;
        address expectedSender = address(4567890);

        context.mock_setTrustedMarketForwarder(marketId, address(this));
        context.mock_setApprovedMarketForwarder(
            marketId,
            address(this),
            expectedSender,
            true
        );

        bytes memory data = abi.encodeWithSignature(
            "external__msgSenderForMarket(uint256)",
            marketId
        );
        bytes memory res = address(context).functionCall(
            abi.encodePacked(data, expectedSender)
        );

        address sender = abi.decode(res, (address));
        assertEq(
            sender,
            expectedSender,
            "sender should be the appended 20 bytes"
        );
    }
}
