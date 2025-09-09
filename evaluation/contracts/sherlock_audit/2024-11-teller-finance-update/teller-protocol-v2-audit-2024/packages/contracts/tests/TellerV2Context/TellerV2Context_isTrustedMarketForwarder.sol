// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { Testable } from "../Testable.sol";
import { TellerV2Context_Override } from "./TellerV2Context_Override.sol";

contract TellerV2Context_isTrustedMarketForwarder is Testable {
    address private lenderCommitmentForwarder = address(111);

    TellerV2Context_Override private context;

    function setUp() public {
        context = new TellerV2Context_Override(
            address(0),
            lenderCommitmentForwarder
        );
    }

    function test_Lender_Commitment_Forwarder_trusted_by_all_markets() public {
        bool isTrusted1 = context.isTrustedMarketForwarder(
            1,
            lenderCommitmentForwarder
        );
        bool isTrusted7 = context.isTrustedMarketForwarder(
            7,
            lenderCommitmentForwarder
        );
        bool isTrusted34 = context.isTrustedMarketForwarder(
            34,
            lenderCommitmentForwarder
        );
        bool isTrusted89 = context.isTrustedMarketForwarder(
            89,
            lenderCommitmentForwarder
        );
        assertTrue(
            isTrusted1 && isTrusted7 && isTrusted34 && isTrusted89,
            "lenderCommitmentForwarder should be a trusted forwarder for all markets"
        );
    }

    function test_Custom_Market_Forwarder_trusted_for_market() public {
        address stubbedMarketForwarder = address(123);
        assertFalse(
            context.isTrustedMarketForwarder(7, stubbedMarketForwarder),
            "by default an address should not be a trusted forwarder"
        );

        context.mock_setTrustedMarketForwarder(7, stubbedMarketForwarder);

        assertTrue(
            context.isTrustedMarketForwarder(7, stubbedMarketForwarder),
            "address should be a trusted forwarder for the market after setting"
        );
    }
}
