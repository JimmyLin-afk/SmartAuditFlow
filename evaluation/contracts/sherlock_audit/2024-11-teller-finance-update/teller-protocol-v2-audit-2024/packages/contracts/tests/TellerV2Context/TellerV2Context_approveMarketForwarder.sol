// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { Testable } from "../Testable.sol";
import { TellerV2Context_Override, TellerV2Context } from "./TellerV2Context_Override.sol";

contract TellerV2Context_approveMarketForwarder is Testable {
    TellerV2Context_Override private context;

    function setUp() public {
        context = new TellerV2Context_Override(address(0), address(111));
    }

    function test_Fail_to_approve_untrusted_forwarder() public {
        uint256 marketId = 7;
        address stubbedMarketForwarder = address(123);

        vm.expectRevert("Forwarder must be trusted by the market");
        context.approveMarketForwarder(marketId, stubbedMarketForwarder);
    }

    event MarketForwarderApproved(
        uint256 indexed marketId,
        address indexed forwarder,
        address sender
    );

    function test_Successfully_approve_trusted_market_forwarder() public {
        uint256 marketId = 7;
        address stubbedMarketForwarder = address(123);
        context.mock_setTrustedMarketForwarder(
            marketId,
            stubbedMarketForwarder
        );

        //vm.expectEmit(address(context));
        //emit MarketForwarderApproved(marketId, stubbedMarketForwarder, address(this));

        context.approveMarketForwarder(marketId, stubbedMarketForwarder);

        assertEq(
            context.hasApprovedMarketForwarder(
                marketId,
                stubbedMarketForwarder,
                address(this)
            ),
            true,
            "Forwarder should be approved"
        );
    }
}
