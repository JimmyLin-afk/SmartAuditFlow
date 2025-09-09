// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/utils/Address.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

import "@openzeppelin/contracts/token/ERC721/IERC721.sol";

import "../contracts/TellerV2MarketForwarder_G1.sol";
import { Testable } from "./Testable.sol";
import { LenderManager } from "../contracts/LenderManager.sol";

import "../contracts/mock/MarketRegistryMock.sol";

import { User } from "./Test_Helpers.sol";

import { TellerV2Context } from "../contracts/TellerV2Context.sol";

contract LenderManager_Test is Testable, LenderManager {
    LenderManagerUser private marketOwner;
    LenderManagerUser private lender;
    LenderManagerUser private borrower;

    LenderCommitmentTester mockTellerV2;
    MarketRegistryMock mockMarketRegistry;

    bool mockedHasMarketVerification;

    constructor()
        LenderManager(
            //address(0),
            new MarketRegistryMock()
        )
    {}

    function setUp() public {
        mockTellerV2 = new LenderCommitmentTester();

        marketOwner = new LenderManagerUser(
            address(mockTellerV2),
            address(this)
        );
        borrower = new LenderManagerUser(address(mockTellerV2), address(this));
        lender = new LenderManagerUser(address(mockTellerV2), address(this));

        mockMarketRegistry = new MarketRegistryMock();

        mockMarketRegistry.setMarketOwner(address(marketOwner));

        delete mockedHasMarketVerification;
    }

    function test_registerLoan() public {
        mockedHasMarketVerification = true;

        uint256 bidId = 2;

        super.registerLoan(bidId, address(lender));

        assertEq(
            super._exists(bidId),
            true,
            "Loan registration did not mint nft"
        );
    }

    function test_transferFrom() public {
        mockedHasMarketVerification = true;

        uint256 bidId = 2;

        super._mint(address(lender), bidId);

        lender.transferLoan(bidId, address(borrower));

        assertEq(
            super.ownerOf(bidId),
            address(borrower),
            "Loan nft was not transferred"
        );
    }

    function test_transferFromToInvalidRecipient() public {
        mockedHasMarketVerification = true;

        uint256 bidId = 2;
        super._mint(address(lender), bidId);

        mockedHasMarketVerification = false;

        bool transferFailed;

        //I want to expect this to fail !!
        try lender.transferLoan(bidId, address(address(borrower))) {} catch {
            transferFailed = true;
        }

        assertEq(transferFailed, true, "Loan transfer should have failed");

        assertEq(
            super.ownerOf(bidId),
            address(lender),
            "Loan nft is no longer owned by lender"
        );
    }

    //override
    function _hasMarketVerification(address _lender, uint256 _bidId)
        internal
        view
        override
        returns (bool)
    {
        return mockedHasMarketVerification;
    }

    //should be able to test the negative case-- use foundry
    function _checkOwner() internal view override {
        // do nothing
    }
}

contract LenderManagerUser is User {
    address lenderManager;

    constructor(address _tellerV2, address _lenderManager) User(_tellerV2) {
        lenderManager = _lenderManager;
    }

    function transferLoan(uint256 bidId, address to) public {
        IERC721(lenderManager).transferFrom(address(this), to, bidId);
    }

    function onERC721Received(
        address operator,
        address from,
        uint256 tokenId,
        bytes calldata data
    ) external returns (bytes4) {
        return
            bytes4(
                keccak256("onERC721Received(address,address,uint256,bytes)")
            );
    }
}

//Move to a helper  or change it
contract LenderCommitmentTester is TellerV2Context {
    constructor() TellerV2Context(address(0)) {}

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
