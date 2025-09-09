// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { Testable } from "../Testable.sol";
import { TellerV2Context_Override, TellerV2Context } from "./TellerV2Context_Override.sol";

contract TellerV2Context_setTrustedMarketForwarder is Testable {
    address private lenderCommitmentForwarder = address(111);

    MockMarketRegistry private marketRegistry;
    TellerV2Context_Override private context;

    function setUp() public {
        marketRegistry = new MockMarketRegistry();
        context = new TellerV2Context_Override(
            address(marketRegistry),
            lenderCommitmentForwarder
        );
    }

    function test_Non_market_owner_fails_to_set_trusted_forwarder() public {
        vm.expectRevert("Caller must be the market owner");
        context.setTrustedMarketForwarder(7, address(123));
    }

    event TrustedMarketForwarderSet(
        uint256 indexed marketId,
        address forwarder,
        address sender
    );

    function test_Market_owner_can_set_trusted_forwarder() public {
        uint256 marketId = 7;
        address newMarketForwarder = address(123);

        marketRegistry.mock_setMarketOwner(marketId, address(this));

        //vm.expectEmit(address(context));
        //emit TrustedMarketForwarderSet(marketId, newMarketForwarder, address(this));

        context.setTrustedMarketForwarder(marketId, newMarketForwarder);

        assertEq(
            context.isTrustedMarketForwarder(marketId, newMarketForwarder),
            true,
            "Trusted forwarder should be set"
        );
    }
}

contract MockMarketRegistry {
    mapping(uint256 => address) private marketOwners;

    function mock_setMarketOwner(uint256 _marketId, address _owner) public {
        marketOwners[_marketId] = _owner;
    }

    function getMarketOwner(uint256 _marketId) external view returns (address) {
        return marketOwners[_marketId];
    }
}
