// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "hardhat/console.sol";

import "@openzeppelin/contracts/utils/Address.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

import { Testable } from "./Testable.sol";
import { TellerV2Context } from "../contracts/TellerV2Context.sol";
import { IMarketRegistry } from "../contracts/interfaces/IMarketRegistry.sol";
import { TellerV2MarketForwarder_G1 } from "../contracts/TellerV2MarketForwarder_G1.sol";

import { User } from "./Test_Helpers.sol";

import "../contracts/mock/MarketRegistryMock.sol";

contract MarketForwarder_Test is Testable, TellerV2MarketForwarder_G1 {
    MarketForwarderTester private tellerV2Mock;

    MarketRegistryMock mockMarketRegistry;

    uint256 private marketId;
    MarketForwarderUser private marketOwner;
    MarketForwarderUser private user1;
    MarketForwarderUser private user2;

    constructor()
        TellerV2MarketForwarder_G1(
            address(new MarketForwarderTester()),
            address(new MarketRegistryMock())
        )
    {}

    function setUp() public {
        mockMarketRegistry = MarketRegistryMock(address(getMarketRegistry()));
        tellerV2Mock = MarketForwarderTester(address(getTellerV2()));

        marketOwner = new MarketForwarderUser(address(tellerV2Mock));
        user1 = new MarketForwarderUser(address(tellerV2Mock));
        user2 = new MarketForwarderUser(address(tellerV2Mock));

        tellerV2Mock.__setMarketRegistry(address(mockMarketRegistry));

        mockMarketRegistry.setMarketOwner(address(marketOwner));

        delete marketId;
    }

    function setTrustedMarketForwarder_before() public {
        marketOwner.setTrustedMarketForwarder(marketId, address(this));
    }

    function test_setTrustedMarketForwarder() public {
        setTrustedMarketForwarder_before();
        assertEq(
            tellerV2Mock.isTrustedMarketForwarder(marketId, address(this)),
            true,
            "Trusted forwarder was not set"
        );
    }

    function approveMarketForwarder_before() public {
        setTrustedMarketForwarder_before();

        user1.approveMarketForwarder(marketId, address(this));
        user2.approveMarketForwarder(marketId, address(this));
    }

    function test_approveMarketForwarder() public {
        approveMarketForwarder_before();
        assertEq(
            tellerV2Mock.hasApprovedMarketForwarder(
                marketId,
                address(this),
                address(user1)
            ),
            true,
            "Borrower did not set market forwarder approval"
        );

        assertEq(
            tellerV2Mock.hasApprovedMarketForwarder(
                marketId,
                address(this),
                address(user2)
            ),
            true,
            "Lender did not set market forwarder approval"
        );
    }

    function forwardUserCall_before() public {
        approveMarketForwarder_before();
    }

    function test_forwardUserCall() public {
        forwardUserCall_before();

        address expectedSender = address(user1);
        address sender = abi.decode(
            _forwardCall(
                abi.encodeWithSelector(
                    MarketForwarderTester.getSenderForMarket.selector,
                    marketId
                ),
                expectedSender
            ),
            (address)
        );
        assertEq(
            sender,
            expectedSender,
            "Sender address for market does not match expected"
        );

        bytes memory expectedData = abi.encodeWithSelector(
            MarketForwarderTester.getDataForMarket.selector,
            marketId
        );
        bytes memory data = abi.decode(
            _forwardCall(expectedData, expectedSender),
            (bytes)
        );
        assertEq0(
            data,
            expectedData,
            "Function calldata for market does not match expected"
        );
    }
}

//This should use the user helper !!
contract MarketForwarderUser is User {
    constructor(address _tellerV2) User(_tellerV2) {}
}

//Move to a helper
//this is a tellerV2 mock
contract MarketForwarderTester is TellerV2Context {
    constructor() TellerV2Context(address(0)) {}

    function __setMarketRegistry(address _marketRegistry) external {
        marketRegistry = IMarketRegistry(_marketRegistry);
    }

    function getSenderForMarket(uint256 _marketId)
        external
        view
        returns (address)
    {
        return _msgSenderForMarket(_marketId);
    }

    function getDataForMarket(uint256 _marketId)
        external
        view
        returns (bytes calldata)
    {
        return _msgDataForMarket(_marketId);
    }
}
