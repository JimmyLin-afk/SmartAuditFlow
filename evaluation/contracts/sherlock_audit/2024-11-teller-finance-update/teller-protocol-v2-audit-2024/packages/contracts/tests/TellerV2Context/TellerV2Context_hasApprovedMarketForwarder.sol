// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { Testable } from "../Testable.sol";
import { TellerV2Context_Override, TellerV2Context } from "./TellerV2Context_Override.sol";

contract TellerV2Context_hasApprovedMarketForwarder is Testable {
    TellerV2Context_Override private context;

    function setUp() public {
        context = new TellerV2Context_Override(address(0), address(111));
    }

    function test_False_for_untrusted_forwarders() public {
        uint256 marketId = 7;
        address stubbedMarketForwarder = address(123);

        context.mock_setApprovedMarketForwarder(
            marketId,
            stubbedMarketForwarder,
            address(this),
            true
        );

        assertFalse(
            context.hasApprovedMarketForwarder(
                marketId,
                stubbedMarketForwarder,
                address(this)
            ),
            "by default forwarder should not be approved"
        );
    }

    function test_False_for_unapproved_forwarders() public {
        uint256 marketId = 7;
        address stubbedMarketForwarder = address(123);

        context.mock_setTrustedMarketForwarder(
            marketId,
            stubbedMarketForwarder
        );

        assertFalse(
            context.hasApprovedMarketForwarder(
                marketId,
                stubbedMarketForwarder,
                address(this)
            ),
            "by default forwarder should not be approved"
        );
    }

    function test_True_for_trusted_and_approved_forwarders() public {
        uint256 marketId = 7;
        address stubbedMarketForwarder = address(123);

        context.mock_setTrustedMarketForwarder(
            marketId,
            stubbedMarketForwarder
        );
        context.mock_setApprovedMarketForwarder(
            marketId,
            stubbedMarketForwarder,
            address(this),
            true
        );

        assertTrue(
            context.hasApprovedMarketForwarder(
                marketId,
                stubbedMarketForwarder,
                address(this)
            ),
            "forwarder should be approved"
        );
    }
}
