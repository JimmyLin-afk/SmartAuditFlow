// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { Testable } from "../Testable.sol";
import { TellerV2Context_Override, TellerV2Context } from "./TellerV2Context_Override.sol";

contract TellerV2Context_renounceMarketForwarder is Testable {
    TellerV2Context_Override private context;

    event MarketForwarderRenounced(
        uint256 indexed marketId,
        address indexed forwarder,
        address sender
    );

    function setUp() public {
        context = new TellerV2Context_Override(address(0), address(111));
    }

    function test_Successfully_renounce_trusted_market_forwarder() public {
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

        assertEq(
            context.hasApprovedMarketForwarder(
                marketId,
                stubbedMarketForwarder,
                address(this)
            ),
            true,
            "Forwarder should be approved"
        );

        context.renounceMarketForwarder(marketId, stubbedMarketForwarder);

        assertEq(
            context.hasApprovedMarketForwarder(
                marketId,
                stubbedMarketForwarder,
                address(this)
            ),
            false,
            "Forwarder should not be approved"
        );
    }
}
