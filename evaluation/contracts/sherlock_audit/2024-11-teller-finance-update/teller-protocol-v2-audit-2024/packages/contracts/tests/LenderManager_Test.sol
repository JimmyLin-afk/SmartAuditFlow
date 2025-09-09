// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/utils/Address.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

import "@openzeppelin/contracts/token/ERC721/IERC721.sol";

import "../contracts/TellerV2MarketForwarder_G1.sol";
import { Testable } from "./Testable.sol";
import { LenderManager } from "../contracts/LenderManager.sol";
import { LenderManager_Override } from "./LenderManager_Override.sol";

import "../contracts/mock/MarketRegistryMock.sol";

import { User } from "./Test_Helpers.sol";

import { TellerV2Context } from "../contracts/TellerV2Context.sol";

contract LenderManager_Test is Testable {
    LenderManagerUser private marketOwner;
    LenderManagerUser private lender;
    LenderManagerUser private borrower;

    LenderCommitmentTester mockTellerV2;
    MarketRegistryMock mockMarketRegistry;

    LenderManager_Override lenderManager;

    constructor() {}

    function setUp() public {
        mockTellerV2 = new LenderCommitmentTester();

        mockMarketRegistry = new MarketRegistryMock();

        lenderManager = new LenderManager_Override(address(mockMarketRegistry));

        borrower = new LenderManagerUser(
            address(mockTellerV2),
            address(lenderManager)
        );
        lender = new LenderManagerUser(
            address(mockTellerV2),
            address(lenderManager)
        );
        marketOwner = new LenderManagerUser(
            address(mockTellerV2),
            address(lenderManager)
        );

        mockMarketRegistry.setMarketOwner(address(marketOwner));
    }

    function test_initialize() public {
        lenderManager.initialize();
    }

    function test_getLoanMarketId() public {
        lenderManager.initialize();

        uint256 marketId = lenderManager._getLoanMarketIdSuper(1);

        assertEq(
            marketId,
            getLoanMarketId(marketId),
            "Market id is not correct"
        );
    }

    function test_hasMarketVerification() public {
        lenderManager.initialize();

        bool hasMarketVerification = lenderManager._hasMarketVerificationSuper(
            address(lender),
            1
        );

        assertEq(
            hasMarketVerification,
            true,
            "Market verification is not correct"
        );
    }

    function test_baseURI() public {
        lenderManager.initialize();

        string memory baseURI = lenderManager._baseURISuper();

        assertEq(baseURI, "", "Base URI is not correct");
    }

    function test_mint() public {
        lenderManager.setHasMarketVerification(true);

        uint256 bidId = 2;

        lenderManager.mint(address(lender), bidId);

        assertEq(
            lenderManager.exists(bidId),
            true,
            "Loan registration did not mint nft"
        );
    }

    function test_mintToInvalidRecipient() public {
        lenderManager.setHasMarketVerification(false);

        uint256 bidId = 2;

        bool mintFailed;

        vm.expectRevert("Not approved by market");

        lenderManager.mint(address(address(lender)), bidId);
    }

    function test_registerLoan() public {
        lenderManager.initialize();
        lenderManager.setHasMarketVerification(true);

        uint256 bidId = 2;

        lenderManager.registerLoan(bidId, address(lender));

        assertEq(
            lenderManager.exists(bidId),
            true,
            "Loan registration did not mint nft"
        );
    }

    function test_registerLoan_invalid_owner() public {
        lenderManager.setHasMarketVerification(true);

        uint256 bidId = 2;

        vm.prank(address(borrower));
        vm.expectRevert("Ownable: caller is not the owner");

        lenderManager.registerLoan(bidId, address(lender));
    }

    function test_transferFrom() public {
        lenderManager.setHasMarketVerification(true);

        uint256 bidId = 2;

        lenderManager.mint(address(lender), bidId);

        lender.transferLoan(bidId, address(borrower));

        assertEq(
            lenderManager.ownerOf(bidId),
            address(borrower),
            "Loan nft was not transferred"
        );
    }

    function test_transferFromToInvalidRecipient() public {
        lenderManager.setHasMarketVerification(true);

        uint256 bidId = 2;
        lenderManager.mint(address(lender), bidId);

        lenderManager.setHasMarketVerification(false);

        bool transferFailed;

        //I want to expect this to fail !!
        try lender.transferLoan(bidId, address(address(borrower))) {} catch {
            transferFailed = true;
        }

        assertEq(transferFailed, true, "Loan transfer should have failed");

        assertEq(
            lenderManager.ownerOf(bidId),
            address(lender),
            "Loan nft is no longer owned by lender"
        );
    }

    // Overrides

    function getLoanMarketId(uint256 bidId) public view returns (uint256) {
        return 42;
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
