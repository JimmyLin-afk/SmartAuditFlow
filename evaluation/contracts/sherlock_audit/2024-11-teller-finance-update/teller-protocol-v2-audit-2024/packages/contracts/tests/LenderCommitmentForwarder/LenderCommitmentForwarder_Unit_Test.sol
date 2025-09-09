// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/utils/Address.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "../../contracts/TellerV2MarketForwarder_G1.sol";

import "../tokens/TestERC20Token.sol";
import "../tokens/TestERC721Token.sol";
import "../tokens/TestERC1155Token.sol";
import "../../contracts/TellerV2Context.sol";

import { Testable } from "../Testable.sol";

import "../../contracts/interfaces/ILenderCommitmentForwarder.sol";
import { LenderCommitmentForwarder_G2 } from "../../contracts/LenderCommitmentForwarder/LenderCommitmentForwarder_G2.sol";

import { Collateral, CollateralType } from "../../contracts/interfaces/escrow/ICollateralEscrowV1.sol";

import { User } from "../Test_Helpers.sol";

import "../../contracts/mock/MarketRegistryMock.sol";

import { LenderCommitmentForwarder_Override } from "./LenderCommitmentForwarder_Override.sol";

contract LenderCommitmentForwarder_Test is Testable {
    LenderCommitmentForwarderTest_TellerV2Mock private tellerV2Mock;
    MarketRegistryMock mockMarketRegistry;

    LenderCommitmentUser private marketOwner;
    LenderCommitmentUser private lender;
    LenderCommitmentUser private borrower;

    address[] emptyArray;
    address[] borrowersArray;

    TestERC20Token principalToken;
    uint8 constant principalTokenDecimals = 18;

    TestERC20Token collateralToken;
    uint8 constant collateralTokenDecimals = 6;

    TestERC721Token erc721Token;
    TestERC1155Token erc1155Token;

    LenderCommitmentForwarder_Override lenderCommitmentForwarder;

    uint256 maxPrincipal;
    uint32 expiration;
    uint32 maxDuration;
    uint16 minInterestRate;
    // address collateralTokenAddress;
    uint256 collateralTokenId;
    uint256 maxPrincipalPerCollateralAmount;
    ILenderCommitmentForwarder.CommitmentCollateralType collateralTokenType;

    uint256 marketId;

    //  address principalTokenAddress;

    /*
 
           
            FNDA:0,LenderCommitmentForwarder.getCommitmentBorrowers
            FNDA:0,LenderCommitmentForwarder._getEscrowCollateralType



    */

    constructor() {}

    function setUp() public {
        tellerV2Mock = new LenderCommitmentForwarderTest_TellerV2Mock();
        mockMarketRegistry = new MarketRegistryMock();

        lenderCommitmentForwarder = new LenderCommitmentForwarder_Override(
            address(tellerV2Mock),
            address(mockMarketRegistry)
        );

        marketOwner = new LenderCommitmentUser(
            address(tellerV2Mock),
            address(lenderCommitmentForwarder)
        );
        borrower = new LenderCommitmentUser(
            address(tellerV2Mock),
            address(lenderCommitmentForwarder)
        );
        lender = new LenderCommitmentUser(
            address(tellerV2Mock),
            address(lenderCommitmentForwarder)
        );

        tellerV2Mock.__setMarketRegistry(address(mockMarketRegistry));
        mockMarketRegistry.setMarketOwner(address(marketOwner));

        //tokenAddress = address(0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174);
        marketId = 2;
        maxPrincipal = 100000000000000000000;
        maxPrincipalPerCollateralAmount = 100;
        maxDuration = 2480000;
        minInterestRate = 3000;
        expiration = uint32(block.timestamp) + uint32(64000);

        marketOwner.setTrustedMarketForwarder(
            marketId,
            address(lenderCommitmentForwarder)
        );
        lender.approveMarketForwarder(
            marketId,
            address(lenderCommitmentForwarder)
        );

        borrowersArray = new address[](1);
        borrowersArray[0] = address(borrower);

        principalToken = new TestERC20Token(
            "Test Wrapped ETH",
            "TWETH",
            0,
            principalTokenDecimals
        );

        collateralToken = new TestERC20Token(
            "Test USDC",
            "TUSDC",
            0,
            collateralTokenDecimals
        );

        erc721Token = new TestERC721Token("ERC721", "ERC721");

        erc1155Token = new TestERC1155Token("Test 1155");
    }

    function test_createCommitment() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: maxPrincipalPerCollateralAmount,
                collateralTokenType: collateralTokenType,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        uint256 c_id = lender._createCommitment(c, emptyArray);

        assertEq(
            lenderCommitmentForwarder.getCommitmentLender(c_id),
            address(lender),
            "unexpected lender for created commitment"
        );
    }

    function test_createCommitment_invalid_lender() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: maxPrincipalPerCollateralAmount,
                collateralTokenType: collateralTokenType,
                lender: address(borrower),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        vm.expectRevert("unauthorized commitment creator");

        lender._createCommitment(c, emptyArray);
    }

    function test_createCommitment_invalid_principal() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: 0,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: maxPrincipalPerCollateralAmount,
                collateralTokenType: collateralTokenType,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        vm.expectRevert("commitment principal allocation 0");

        lender._createCommitment(c, emptyArray);
    }

    function test_createCommitment_expired() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: 0,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: maxPrincipalPerCollateralAmount,
                collateralTokenType: collateralTokenType,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        vm.expectRevert("expired commitment");

        lender._createCommitment(c, emptyArray);
    }

    function test_createCommitment_collateralType() public {}

    function test_updateCommitment() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: maxPrincipalPerCollateralAmount,
                collateralTokenType: collateralTokenType,
                lender: address(lender),
                marketId: 99,
                principalTokenAddress: address(principalToken)
            });

        lenderCommitmentForwarder.setCommitment(0, c);

        vm.prank(address(lender));
        lenderCommitmentForwarder.updateCommitment(0, c);

        assertEq(
            lenderCommitmentForwarder.getCommitmentMarketId(0),
            c.marketId,
            "unexpected marketId after update"
        );
    }

    function test_updateCommitment_reject_change_lender() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: maxPrincipalPerCollateralAmount,
                collateralTokenType: collateralTokenType,
                lender: address(lender),
                marketId: 99,
                principalTokenAddress: address(principalToken)
            });

        lenderCommitmentForwarder.setCommitment(0, c);
        vm.expectRevert("Commitment lender cannot be updated.");

        c.lender = address(borrower);

        vm.prank(address(lender));
        lenderCommitmentForwarder.updateCommitment(0, c);
    }

    function test_updateCommitment_invalid_lender() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: 0,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: maxPrincipalPerCollateralAmount,
                collateralTokenType: collateralTokenType,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        vm.expectRevert("unauthorized commitment lender");

        vm.prank(address(lender));
        lenderCommitmentForwarder.updateCommitment(99, c);
    }

    function test_updateCommitment_prevent_update_principal_token() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: maxPrincipalPerCollateralAmount,
                collateralTokenType: collateralTokenType,
                lender: address(lender),
                marketId: 99,
                principalTokenAddress: address(principalToken)
            });

        lenderCommitmentForwarder.setCommitment(0, c);

        c.principalTokenAddress = address(collateralToken);

        vm.expectRevert("Principal token address cannot be updated.");
        vm.prank(address(lender));
        lenderCommitmentForwarder.updateCommitment(0, c);
    }

    function test_updateCommitment_prevent_update_market_id() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: maxPrincipalPerCollateralAmount,
                collateralTokenType: collateralTokenType,
                lender: address(lender),
                marketId: 99,
                principalTokenAddress: address(principalToken)
            });

        lenderCommitmentForwarder.setCommitment(0, c);

        c.marketId = 100;

        vm.expectRevert("Market Id cannot be updated.");
        vm.prank(address(lender));
        lenderCommitmentForwarder.updateCommitment(0, c);
    }

    function test_addCommitmentBorrowers() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: maxPrincipalPerCollateralAmount,
                collateralTokenType: collateralTokenType,
                lender: address(lender),
                marketId: 99,
                principalTokenAddress: address(principalToken)
            });

        lenderCommitmentForwarder.setCommitment(0, c);

        address[] memory newBorrowers = new address[](1);
        newBorrowers[0] = address(1);

        vm.prank(address(lender));
        lenderCommitmentForwarder.addCommitmentBorrowers(0, newBorrowers);

        //check an assertion
        assertEq(
            lenderCommitmentForwarder.getCommitmentBorrowers(0).length,
            1,
            "unexpected borrower count after update"
        );
    }

    function test_addCommitmentBorrowers_cannot_update_empty() public {
        vm.expectRevert("unauthorized commitment lender");
        lenderCommitmentForwarder.addCommitmentBorrowers(0, emptyArray);
    }

    function test_uaddCommitmentBorrowers_unauthorized() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: maxPrincipalPerCollateralAmount,
                collateralTokenType: collateralTokenType,
                lender: address(lender),
                marketId: 99,
                principalTokenAddress: address(principalToken)
            });

        lenderCommitmentForwarder.setCommitment(0, c);

        vm.expectRevert("unauthorized commitment lender");
        lenderCommitmentForwarder.addCommitmentBorrowers(0, emptyArray);
    }

    function test_deleteCommitment() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: maxPrincipalPerCollateralAmount,
                collateralTokenType: collateralTokenType,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        lenderCommitmentForwarder.setCommitment(0, c);

        lender._deleteCommitment(0);

        assertEq(
            lenderCommitmentForwarder.getCommitmentLender(0),
            address(0),
            "commitment data was not deleted"
        );
    }

    function test_deleteCommitment_unauthorized() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: maxPrincipalPerCollateralAmount,
                collateralTokenType: collateralTokenType,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        lenderCommitmentForwarder.setCommitment(0, c);

        vm.expectRevert("unauthorized commitment lender");

        lenderCommitmentForwarder.deleteCommitment(0);
    }

    function test_validateCommitment() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: maxPrincipalPerCollateralAmount,
                collateralTokenType: collateralTokenType,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        uint256 commitmentId = 0;

        lenderCommitmentForwarder.setCommitment(commitmentId, c);

        lenderCommitmentForwarder.validateCommitmentSuper(commitmentId);
    }

    function test_validateCommitment_expired() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: maxPrincipalPerCollateralAmount,
                collateralTokenType: collateralTokenType,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        uint256 commitmentId = 0;

        lenderCommitmentForwarder.setCommitment(commitmentId, c);

        vm.warp(expiration + 1);

        vm.expectRevert("expired commitment");
        lenderCommitmentForwarder.validateCommitmentSuper(commitmentId);
    }

    function test_validateCommitment_zero_principal_allocation() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: 0,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: maxPrincipalPerCollateralAmount,
                collateralTokenType: collateralTokenType,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        uint256 commitmentId = 0;

        lenderCommitmentForwarder.setCommitment(commitmentId, c);

        vm.expectRevert("commitment principal allocation 0");
        lenderCommitmentForwarder.validateCommitmentSuper(commitmentId);
    }

    function test_validateCommitment_zero_collateral_ratio() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: 0,
                collateralTokenType: ILenderCommitmentForwarder
                    .CommitmentCollateralType
                    .ERC20,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        uint256 commitmentId = 0;

        lenderCommitmentForwarder.setCommitment(commitmentId, c);

        vm.expectRevert("commitment collateral ratio 0");
        lenderCommitmentForwarder.validateCommitmentSuper(commitmentId);
    }

    function test_validateCommitment_erc20_with_token_id() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: 66,
                maxPrincipalPerCollateralAmount: maxPrincipalPerCollateralAmount,
                collateralTokenType: ILenderCommitmentForwarder
                    .CommitmentCollateralType
                    .ERC20,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        uint256 commitmentId = 0;

        lenderCommitmentForwarder.setCommitment(commitmentId, c);

        vm.expectRevert("commitment collateral token id must be 0 for ERC20");
        lenderCommitmentForwarder.validateCommitmentSuper(commitmentId);
    }

    function test_acceptCommitment() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: maxPrincipalPerCollateralAmount,
                collateralTokenType: collateralTokenType,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        uint256 commitmentId = 0;

        lenderCommitmentForwarder.setCommitment(commitmentId, c);

        uint256 bidId = borrower._acceptCommitment(
            commitmentId,
            maxPrincipal - 100, //principal
            maxPrincipal, //collateralAmount
            0, //collateralTokenId
            address(collateralToken),
            minInterestRate,
            maxDuration
        );

        assertEq(
            lenderCommitmentForwarder.acceptBidWasCalled(),
            true,
            "Expect accept bid called after exercise"
        );
    }

    function test_acceptCommitment_exact_principal() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: maxPrincipalPerCollateralAmount,
                collateralTokenType: collateralTokenType,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        uint256 commitmentId = 0;

        lenderCommitmentForwarder.setCommitment(commitmentId, c);

        uint256 principalAmount = maxPrincipal;
        uint256 collateralAmount = 1000;
        uint16 interestRate = minInterestRate;
        uint32 loanDuration = maxDuration;

        // vm.expectRevert("collateral token mismatch");
        lenderCommitmentForwarder.acceptCommitment(
            commitmentId,
            principalAmount,
            collateralAmount,
            collateralTokenId,
            address(collateralToken),
            interestRate,
            loanDuration
        );

        assertEq(
            lenderCommitmentForwarder.getCommitmentMaxPrincipal(commitmentId),
            maxPrincipal,
            "Max principal changed"
        );
    }

    function test_acceptCommitment_merkle_proof() public {
        //https://github.com/OpenZeppelin/openzeppelin-contracts/blob/master/contracts/utils/cryptography/MerkleProof.sol

        uint256 tokenIdLeaf = 1;
        bytes32 merkleLeaf = keccak256(abi.encodePacked(tokenIdLeaf)); //  0xc89efdaa54c0f20c7adf612882df0950f5a951637e0307cdcb4c672f298b8bc6;
        bytes32 merkleRoot = merkleLeaf;
        bytes32[] memory merkleProof = new bytes32[](0);

        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: uint256(merkleRoot),
                maxPrincipalPerCollateralAmount: maxPrincipal * 1e18,
                collateralTokenType: ILenderCommitmentForwarder
                    .CommitmentCollateralType
                    .ERC721_MERKLE_PROOF,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        uint256 commitmentId = 0;

        lenderCommitmentForwarder.setCommitment(commitmentId, c);

        uint256 principalAmount = maxPrincipal;
        uint256 collateralAmount = 1;
        uint16 interestRate = minInterestRate;
        uint32 loanDuration = maxDuration;

        collateralTokenId = tokenIdLeaf;

        // vm.expectRevert("collateral token mismatch");
        lenderCommitmentForwarder.acceptCommitmentWithProof(
            commitmentId,
            principalAmount,
            collateralAmount,
            collateralTokenId,
            address(collateralToken),
            interestRate,
            loanDuration,
            merkleProof
        );

        assertEq(
            lenderCommitmentForwarder.getCommitmentMaxPrincipal(commitmentId),
            maxPrincipal,
            "Max principal changed"
        );

        assertEq(
            lenderCommitmentForwarder.getCommitmentAcceptedPrincipal(
                commitmentId
            ),
            principalAmount,
            "Incorrect accepted principal"
        );
    }

    function test_acceptCommitment_merkle_proof_two() public {
        //https://github.com/OpenZeppelin/openzeppelin-contracts/blob/master/contracts/utils/cryptography/MerkleProof.sol

        uint256 tokenIdLeafA = 1;
        uint256 tokenIdLeafB = 3;

        bytes32 merkleLeafA = keccak256(abi.encodePacked(tokenIdLeafA)); //  0xc89efdaa54c0f20c7adf612882df0950f5a951637e0307cdcb4c672f298b8bc6;
        bytes32 merkleLeafB = keccak256(abi.encodePacked(tokenIdLeafB));

        //a merkle root is simply the hash of the hashes of the leaves in the layer above, where the leaves are always sorted alphanumerically.
        //it so happens that the hash of (1) is less than the hash of (3) so we can compute the merkle root manually like this without a sorting function:
        bytes32 merkleRoot = keccak256(
            abi.encodePacked(merkleLeafA, merkleLeafB)
        );

        bytes32[] memory merkleProof = new bytes32[](1);
        merkleProof[0] = merkleLeafB;

        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: uint256(merkleRoot),
                maxPrincipalPerCollateralAmount: maxPrincipal * 1e18,
                collateralTokenType: ILenderCommitmentForwarder
                    .CommitmentCollateralType
                    .ERC721_MERKLE_PROOF,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        uint256 commitmentId = 0;

        lenderCommitmentForwarder.setCommitment(commitmentId, c);

        uint256 principalAmount = maxPrincipal;
        uint256 collateralAmount = 1;
        uint16 interestRate = minInterestRate;
        uint32 loanDuration = maxDuration;

        collateralTokenId = tokenIdLeafA;

        // vm.expectRevert("collateral token mismatch");
        lenderCommitmentForwarder.acceptCommitmentWithProof(
            commitmentId,
            principalAmount,
            collateralAmount,
            collateralTokenId,
            address(collateralToken),
            interestRate,
            loanDuration,
            merkleProof
        );

        assertEq(
            lenderCommitmentForwarder.getCommitmentMaxPrincipal(commitmentId),
            maxPrincipal,
            "Max principal changed"
        );

        assertEq(
            lenderCommitmentForwarder.getCommitmentAcceptedPrincipal(
                commitmentId
            ),
            principalAmount,
            "Incorrect accepted principal"
        );
    }

    function test_acceptCommitment_merkle_proof_invalid_proof() public {
        uint256 tokenIdLeaf = 1;
        bytes32 merkleLeaf = keccak256(abi.encodePacked(tokenIdLeaf)); //  0xc89efdaa54c0f20c7adf612882df0950f5a951637e0307cdcb4c672f298b8bc6;
        bytes32 merkleRoot = keccak256(abi.encodePacked(uint256(2)));
        bytes32[] memory merkleProof = new bytes32[](0);

        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: uint256(merkleRoot),
                maxPrincipalPerCollateralAmount: maxPrincipal * 1e18,
                collateralTokenType: ILenderCommitmentForwarder
                    .CommitmentCollateralType
                    .ERC721_MERKLE_PROOF,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        uint256 commitmentId = 0;

        lenderCommitmentForwarder.setCommitment(commitmentId, c);

        uint256 principalAmount = maxPrincipal;
        uint256 collateralAmount = 1;
        uint16 interestRate = minInterestRate;
        uint32 loanDuration = maxDuration;

        collateralTokenId = tokenIdLeaf;

        vm.expectRevert("Invalid proof");
        lenderCommitmentForwarder.acceptCommitmentWithProof(
            commitmentId,
            principalAmount,
            collateralAmount,
            collateralTokenId,
            address(collateralToken),
            interestRate,
            loanDuration,
            merkleProof
        );
    }

    function test_acceptCommitment_merkle_proof_invalid_type() public {
        uint256 tokenIdLeaf = 1;
        bytes32 merkleLeaf = keccak256(abi.encodePacked(tokenIdLeaf)); //  0xc89efdaa54c0f20c7adf612882df0950f5a951637e0307cdcb4c672f298b8bc6;
        bytes32 merkleRoot = merkleLeaf;
        bytes32[] memory merkleProof = new bytes32[](0);

        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: uint256(merkleRoot),
                maxPrincipalPerCollateralAmount: maxPrincipal * 1e18,
                collateralTokenType: ILenderCommitmentForwarder
                    .CommitmentCollateralType
                    .ERC721,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        uint256 commitmentId = 0;

        lenderCommitmentForwarder.setCommitment(commitmentId, c);

        uint256 principalAmount = maxPrincipal;
        uint256 collateralAmount = 1;
        uint16 interestRate = minInterestRate;
        uint32 loanDuration = maxDuration;

        collateralTokenId = tokenIdLeaf;

        vm.expectRevert("Invalid commitment collateral type");
        lenderCommitmentForwarder.acceptCommitmentWithProof(
            commitmentId,
            principalAmount,
            collateralAmount,
            collateralTokenId,
            address(collateralToken),
            interestRate,
            loanDuration,
            merkleProof
        );
    }

    function test_acceptCommitment_mismatch_collateral_token() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: maxPrincipalPerCollateralAmount,
                collateralTokenType: collateralTokenType,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        uint256 commitmentId = 0;

        lenderCommitmentForwarder.setCommitment(commitmentId, c);

        uint256 principalAmount = maxPrincipal;
        uint256 collateralAmount = 1000;
        uint16 interestRate = minInterestRate;
        uint32 loanDuration = maxDuration;

        vm.expectRevert("Mismatching collateral token");
        lenderCommitmentForwarder.acceptCommitment(
            commitmentId,
            principalAmount,
            collateralAmount,
            collateralTokenId,
            address(principalToken),
            interestRate,
            loanDuration
        );
    }

    function test_acceptCommitment_invalid_interest_rate() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: maxPrincipalPerCollateralAmount,
                collateralTokenType: collateralTokenType,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        uint256 commitmentId = 0;

        lenderCommitmentForwarder.setCommitment(commitmentId, c);

        uint256 principalAmount = maxPrincipal;
        uint256 collateralAmount = 1000;
        uint16 interestRate = 0;
        uint32 loanDuration = maxDuration;

        vm.expectRevert("Invalid interest rate");
        lenderCommitmentForwarder.acceptCommitment(
            commitmentId,
            principalAmount,
            collateralAmount,
            collateralTokenId,
            address(collateralToken),
            interestRate,
            loanDuration
        );
    }

    function test_acceptCommitment_invalid_duration() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: maxPrincipalPerCollateralAmount,
                collateralTokenType: collateralTokenType,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        uint256 commitmentId = 0;

        lenderCommitmentForwarder.setCommitment(commitmentId, c);

        uint256 principalAmount = maxPrincipal;
        uint256 collateralAmount = 1000;
        uint16 interestRate = minInterestRate;
        uint32 loanDuration = maxDuration + 100;

        vm.expectRevert("Invalid loan max duration");
        lenderCommitmentForwarder.acceptCommitment(
            commitmentId,
            principalAmount,
            collateralAmount,
            collateralTokenId,
            address(collateralToken),
            interestRate,
            loanDuration
        );
    }

    function test_acceptCommitment_invalid_commitment_borrower() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: maxPrincipalPerCollateralAmount,
                collateralTokenType: collateralTokenType,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        uint256 commitmentId = 0;

        lenderCommitmentForwarder.setCommitment(commitmentId, c);

        uint256 principalAmount = maxPrincipal;
        uint256 collateralAmount = 1000;
        uint16 interestRate = minInterestRate;
        uint32 loanDuration = maxDuration;

        address[] memory newBorrowers = new address[](1);
        newBorrowers[0] = address(1);

        vm.prank(address(lender));
        lenderCommitmentForwarder.addCommitmentBorrowers(
            commitmentId,
            newBorrowers
        );

        vm.expectRevert("unauthorized commitment borrower");
        lenderCommitmentForwarder.acceptCommitment(
            commitmentId,
            principalAmount,
            collateralAmount,
            collateralTokenId,
            address(collateralToken),
            interestRate,
            loanDuration
        );
    }

    function test_acceptCommitment_insufficient_commitment_allocation() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: maxPrincipalPerCollateralAmount,
                collateralTokenType: collateralTokenType,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        uint256 commitmentId = 0;

        lenderCommitmentForwarder.setCommitment(commitmentId, c);

        uint256 principalAmount = maxPrincipal + 100;
        uint256 collateralAmount = 1000;
        uint16 interestRate = minInterestRate;
        uint32 loanDuration = maxDuration;

        vm.expectRevert(
            abi.encodeWithSelector(
                LenderCommitmentForwarder_G2
                    .InsufficientCommitmentAllocation
                    .selector,
                c.maxPrincipal,
                principalAmount
            )
        );
        lenderCommitmentForwarder.acceptCommitment(
            commitmentId,
            principalAmount,
            collateralAmount,
            collateralTokenId,
            address(collateralToken),
            interestRate,
            loanDuration
        );
    }

    function test_acceptCommitment_insufficient_borrower_collateral() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: 10000,
                collateralTokenType: ILenderCommitmentForwarder
                    .CommitmentCollateralType
                    .ERC20,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        uint256 commitmentId = 0;

        lenderCommitmentForwarder.setCommitment(commitmentId, c);

        uint256 principalAmount = maxPrincipal;
        uint256 collateralAmount = 0;
        uint16 interestRate = minInterestRate;
        uint32 loanDuration = maxDuration;

        uint256 requiredCollateralAmount = lenderCommitmentForwarder
            .getRequiredCollateral(
                principalAmount,
                c.maxPrincipalPerCollateralAmount,
                c.collateralTokenType,
                c.collateralTokenAddress,
                c.principalTokenAddress
            );

        vm.expectRevert(
            abi.encodeWithSelector(
                LenderCommitmentForwarder_G2
                    .InsufficientBorrowerCollateral
                    .selector,
                requiredCollateralAmount,
                collateralAmount
            )
        );
        lenderCommitmentForwarder.acceptCommitment(
            commitmentId,
            principalAmount,
            collateralAmount,
            collateralTokenId,
            address(collateralToken),
            interestRate,
            loanDuration
        );
    }

    function test_acceptCommitment_invalid_721_collateral_amount() public {
        ILenderCommitmentForwarder.Commitment memory c = ILenderCommitmentForwarder
            .Commitment({
                maxPrincipal: 100,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(erc721Token),
                collateralTokenId: 0,
                maxPrincipalPerCollateralAmount: 100 * 1e18, //expand by token decimals
                collateralTokenType: ILenderCommitmentForwarder
                    .CommitmentCollateralType
                    .ERC721,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        uint256 commitmentId = 0;

        lenderCommitmentForwarder.setCommitment(commitmentId, c);

        uint256 principalAmount = c.maxPrincipal;
        uint256 collateralAmount = 12;
        uint16 interestRate = minInterestRate;
        uint32 loanDuration = maxDuration;

        vm.expectRevert("invalid commitment collateral amount for ERC721");
        lenderCommitmentForwarder.acceptCommitment(
            commitmentId,
            principalAmount,
            collateralAmount,
            collateralTokenId,
            address(erc721Token),
            interestRate,
            loanDuration
        );
    }

    function test_acceptCommitment_invalid_collateral_id() public {}

    function test_getRequiredCollateral_erc20() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: 10000,
                collateralTokenType: ILenderCommitmentForwarder
                    .CommitmentCollateralType
                    .ERC20,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        uint256 commitmentId = 0;

        lenderCommitmentForwarder.setCommitment(commitmentId, c);

        uint256 principalAmount = maxPrincipal;
        uint256 collateralAmount = 0;
        uint16 interestRate = minInterestRate;
        uint32 loanDuration = maxDuration;

        uint256 requiredCollateral = lenderCommitmentForwarder
            .getRequiredCollateral(
                principalAmount,
                c.maxPrincipalPerCollateralAmount,
                c.collateralTokenType,
                c.collateralTokenAddress,
                c.principalTokenAddress
            );

        assertEq(requiredCollateral, 1e40, "unexpected required collateral");
    }

    function test_getRequiredCollateral_type_none() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: maxPrincipal,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: collateralTokenId,
                maxPrincipalPerCollateralAmount: 10000,
                collateralTokenType: ILenderCommitmentForwarder
                    .CommitmentCollateralType
                    .NONE,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        uint256 commitmentId = 0;

        lenderCommitmentForwarder.setCommitment(commitmentId, c);

        uint256 principalAmount = maxPrincipal;
        uint256 collateralAmount = 0;
        uint16 interestRate = minInterestRate;
        uint32 loanDuration = maxDuration;

        uint256 requiredCollateral = lenderCommitmentForwarder
            .getRequiredCollateral(
                principalAmount,
                c.maxPrincipalPerCollateralAmount,
                c.collateralTokenType,
                c.collateralTokenAddress,
                c.principalTokenAddress
            );

        assertEq(requiredCollateral, 0, "unexpected required collateral");
    }

    function test_getRequiredCollateral_type_erc721() public {
        ILenderCommitmentForwarder.Commitment
            memory c = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: 100,
                expiration: expiration,
                maxDuration: maxDuration,
                minInterestRate: minInterestRate,
                collateralTokenAddress: address(erc721Token),
                collateralTokenId: 0,
                maxPrincipalPerCollateralAmount: 100 * 1e18,
                collateralTokenType: ILenderCommitmentForwarder
                    .CommitmentCollateralType
                    .ERC721,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: address(principalToken)
            });

        uint256 commitmentId = 0;

        lenderCommitmentForwarder.setCommitment(commitmentId, c);

        uint256 principalAmount = c.maxPrincipal;

        uint256 requiredCollateral = lenderCommitmentForwarder
            .getRequiredCollateral(
                principalAmount,
                c.maxPrincipalPerCollateralAmount,
                c.collateralTokenType,
                c.collateralTokenAddress,
                c.principalTokenAddress
            );

        assertEq(requiredCollateral, 1, "unexpected required collateral");
    }

    function test_getEscrowCollateralType_erc20() public {
        CollateralType cType = lenderCommitmentForwarder
            ._getEscrowCollateralTypeSuper(
                ILenderCommitmentForwarder.CommitmentCollateralType.ERC20
            );

        assertEq(
            uint16(cType),
            uint16(CollateralType.ERC20),
            "unexpected collateral type"
        );
    }

    function test_getEscrowCollateralType_erc721() public {
        CollateralType cType = lenderCommitmentForwarder
            ._getEscrowCollateralTypeSuper(
                ILenderCommitmentForwarder.CommitmentCollateralType.ERC721
            );

        assertEq(
            uint16(cType),
            uint16(CollateralType.ERC721),
            "unexpected collateral type"
        );
    }

    function test_getEscrowCollateralType_erc1155() public {
        CollateralType cType = lenderCommitmentForwarder
            ._getEscrowCollateralTypeSuper(
                ILenderCommitmentForwarder.CommitmentCollateralType.ERC1155
            );

        assertEq(
            uint16(cType),
            uint16(CollateralType.ERC1155),
            "unexpected collateral type"
        );
    }

    function test_getEscrowCollateralType_unknown() public {
        vm.expectRevert("Unknown Collateral Type");
        CollateralType cType = lenderCommitmentForwarder
            ._getEscrowCollateralTypeSuper(
                ILenderCommitmentForwarder.CommitmentCollateralType.NONE
            );

        ///assertEq(uint16(cType), uint16(CollateralType.NONE), "unexpected collateral type");
    }

    /*
        Overrider methods for exercise 
    */
    /*

    function _submitBid(CreateLoanArgs memory, address)
        internal
        override
        returns (uint256 bidId)
    {
        submitBidWasCalled = true;
        return 1;
    }

    function _submitBidWithCollateral(
        CreateLoanArgs memory,
        Collateral[] memory,
        address
    ) internal override returns (uint256 bidId) {
        submitBidWithCollateralWasCalled = true;
        return 1;
    }

    function _acceptBid(uint256, address) internal override returns (bool) {
        acceptBidWasCalled = true;

        assertEq(
            submitBidWithCollateralWasCalled,
            true,
            "Submit bid must be called before accept bid"
        );

        return true;
    }

    */
}

contract LenderCommitmentUser is User {
    LenderCommitmentForwarder_G2 public immutable commitmentForwarder;

    constructor(address _tellerV2, address _commitmentForwarder)
        User(_tellerV2)
    {
        commitmentForwarder = LenderCommitmentForwarder_G2(
            _commitmentForwarder
        );
    }

    function _createCommitment(
        ILenderCommitmentForwarder.Commitment calldata _commitment,
        address[] calldata borrowerAddressList
    ) public returns (uint256) {
        return
            commitmentForwarder.createCommitment(
                _commitment,
                borrowerAddressList
            );
    }

    function _acceptCommitment(
        uint256 commitmentId,
        uint256 principal,
        uint256 collateralAmount,
        uint256 collateralTokenId,
        address collateralTokenAddress,
        uint16 interestRate,
        uint32 loanDuration
    ) public returns (uint256) {
        return
            commitmentForwarder.acceptCommitment(
                commitmentId,
                principal,
                collateralAmount,
                collateralTokenId,
                collateralTokenAddress,
                interestRate,
                loanDuration
            );
    }

    function _deleteCommitment(uint256 _commitmentId) public {
        commitmentForwarder.deleteCommitment(_commitmentId);
    }
}

//Move to a helper file !
contract LenderCommitmentForwarderTest_TellerV2Mock is TellerV2Context {
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
