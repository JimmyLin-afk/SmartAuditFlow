// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { StdStorage, stdStorage } from "forge-std/StdStorage.sol";
import { Testable } from "../Testable.sol";
import { TellerV2_Override } from "./TellerV2_Override.sol";

import { Bid, BidState, Collateral, Payment, LoanDetails, Terms, ActionNotAllowed } from "../../contracts/TellerV2.sol";
import { PaymentType, PaymentCycleType } from "../../contracts/libraries/V2Calculations.sol";

import { ReputationManagerMock } from "../../contracts/mock/ReputationManagerMock.sol";
import { CollateralManagerMock } from "../../contracts/mock/CollateralManagerMock.sol";
import { LenderManagerMock } from "../../contracts/mock/LenderManagerMock.sol";
import { MarketRegistryMock } from "../../contracts/mock/MarketRegistryMock.sol";

import { ProtocolPausingManager } from "../../contracts/pausing/ProtocolPausingManager.sol";


import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

import "../tokens/TestERC20Token.sol";

import "lib/forge-std/src/console.sol";

contract TellerV2_bids_test is Testable {
    using stdStorage for StdStorage;

    TellerV2_Override tellerV2;

    TestERC20Token lendingToken;

    TestERC20Token lendingTokenZeroDecimals;

    User borrower;
    User lender;
    User receiver;

    User marketOwner;

    User feeRecipient;

    MarketRegistryMock marketRegistryMock;

    ReputationManagerMock reputationManagerMock;
    CollateralManagerMock collateralManagerMock;
    LenderManagerMock lenderManagerMock;

    ProtocolPausingManager protocolPausingManager;

    uint256 marketplaceId = 100;

    //have to copy and paste events in here to expectEmit
    event SubmittedBid(
        uint256 indexed bidId,
        address indexed borrower,
        address receiver,
        bytes32 indexed metadataURI
    );

    function setUp() public {
        tellerV2 = new TellerV2_Override();

        marketRegistryMock = new MarketRegistryMock();
        reputationManagerMock = new ReputationManagerMock();
        collateralManagerMock = new CollateralManagerMock();
        lenderManagerMock = new LenderManagerMock();
        protocolPausingManager = new ProtocolPausingManager();
        protocolPausingManager.initialize();

        tellerV2.setProtocolPausingManagerSuper(address(protocolPausingManager));
      

        borrower = new User();
        lender = new User();
        receiver = new User();

        marketOwner = new User();
        feeRecipient = new User();

        lendingToken = new TestERC20Token("Wrapped Ether", "WETH", 1e30, 18);
        lendingTokenZeroDecimals = new TestERC20Token(
            "Wrapped Ether",
            "WETH",
            1e16,
            0
        );

        //stdstore.target(address(tellerV2)).sig("marketRegistry()").checked_write(address(0x1234));
    }

    function setMockBid(uint256 bidId) public {
        tellerV2.mock_setBid(
            bidId,
            Bid({
                borrower: address(borrower),
                lender: address(lender),
                receiver: address(receiver),
                marketplaceId: marketplaceId,
                _metadataURI: "0x1234",
                loanDetails: LoanDetails({
                    lendingToken: lendingToken,
                    principal: 100,
                    timestamp: 100,
                    acceptedTimestamp: 100,
                    lastRepaidTimestamp: 100,
                    loanDuration: 5000,
                    totalRepaid: Payment({ principal: 100, interest: 5 })
                }),
                terms: Terms({
                    paymentCycleAmount: 10,
                    paymentCycle: 2000,
                    APR: 10
                }),
                state: BidState.PENDING,
                paymentType: PaymentType.EMI
            })
        );
    }

    /*

    todo 

    FNDA:0,TellerV2.cancelBid
    FNDA:0,TellerV2._cancelBid

    FNDA:0,TellerV2.marketOwnerCancelBid
  
   
    FNDA:0,TellerV2.repayLoanMinimum
    FNDA:0,TellerV2.repayLoan
     FNDA:0,TellerV2.claimLoanNFT
   
    FNDA:0,TellerV2.liquidateLoanFull


    */

    function test_submit_bid_internal() public {
        tellerV2.setMarketRegistrySuper(address(marketRegistryMock));

        vm.expectEmit(false, false, false, false);

        emit SubmittedBid(
            0,
            address(this),
            address(this),
            keccak256(abi.encodePacked(""))
        );

        uint256 bidId = tellerV2._submitBidSuper(
            address(lendingToken), // lending token
            1, // market ID
            100, // principal
            365 days, // duration
            20_00, // interest rate
            "", // metadata URI
            address(this) // receiver
        );
    }

    function test_submit_bid_internal_fails_when_market_closed() public {
        tellerV2.setMarketRegistrySuper(address(marketRegistryMock));

        marketRegistryMock.mock_setGlobalMarketsClosed(true);

        vm.expectRevert("Mkt C");

        tellerV2._submitBidSuper(
            address(lendingToken), // lending token
            1, // market ID
            100, // principal
            365 days, // duration
            20_00, // interest rate
            "", // metadata URI
            address(this) // receiver
        );
    }

    function test_submit_bid_internal_fails_when_borrower_not_verified()
        public
    {
        tellerV2.setMarketRegistrySuper(address(marketRegistryMock));

        marketRegistryMock.mock_setBorrowerIsVerified(false);

        vm.expectRevert("Borrower NV");

        tellerV2._submitBidSuper(
            address(lendingToken), // lending token
            1, // market ID
            100, // principal
            365 days, // duration
            20_00, // interest rate
            "", // metadata URI
            address(this) // receiver
        );
    }

    function test_submit_bid_without_collateral() public {
         //tellerV2.setProtocolPausingManagerSuper(address(protocolPausingManager));
       
        tellerV2.submitBid(
            address(1), // lending token
            1, // market ID
            100, // principal
            365 days, // duration
            20_00, // interest rate
            "", // metadata URI
            address(this) // receiver
        );

        assertTrue(tellerV2.submitBidWasCalled(), "Submit bid was not called");
    }

    function test_submit_bid_with_collateral() public {
        //  tellerV2.setProtocolPausingManagerSuper(address(protocolPausingManager));
       
        tellerV2.setCollateralManagerSuper(address(collateralManagerMock));

        Collateral[] memory collateral = new Collateral[](1);

        tellerV2.submitBid(
            address(1), // lending token
            1, // market ID
            100, // principal
            365 days, // duration
            20_00, // interest rate
            "", // metadata URI
            address(this), // receiver
            collateral // collateral
        );

        assertTrue(tellerV2.submitBidWasCalled(), "Submit bid was not called");
    }

    function test_submit_bid_reverts_when_protocol_IS_paused() public {
        tellerV2.setProtocolPausingManagerSuper(address(protocolPausingManager));
        protocolPausingManager.pauseProtocol();

        vm.expectRevert("Protocol paused");
        tellerV2.submitBid(
            address(1), // lending token
            1, // market ID
            100, // principal
            365 days, // duration
            20_00, // interest rate
            "", // metadata URI
            address(this) // receiver
        );
    }

    function test_submit_bid_Reverts_when_protocol_IS_paused__with_collateral()
        public
    {
        tellerV2.setProtocolPausingManagerSuper(address(protocolPausingManager));
        protocolPausingManager.pauseProtocol();

        Collateral[] memory collateral = new Collateral[](1);

        vm.expectRevert("Protocol paused");
        tellerV2.submitBid(
            address(1), // lending token
            1, // market ID
            100, // principal
            365 days, // duration
            20_00, // interest rate
            "", // metadata URI
            address(this), // receiver
            collateral // collateral
        );
    }

    function test_submit_bid_reverts_when_collateral_invalid() public {
        Collateral[] memory collateral = new Collateral[](1);

        tellerV2.setCollateralManagerSuper(address(collateralManagerMock));

        collateralManagerMock.forceSetCommitCollateralValidation(false);

        vm.expectRevert("C bal NV");
        tellerV2.submitBid(
            address(1), // lending token
            1, // market ID
            100, // principal
            365 days, // duration
            20_00, // interest rate
            "", // metadata URI
            address(this), // receiver
            collateral // collateral
        );
    }

    function test_cancel_bid() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        tellerV2.mock_setBidState(bidId, BidState.PENDING);

        //need to mock as owner
        tellerV2.setMockMsgSenderForMarket(address(borrower));
        vm.prank(address(borrower));

        tellerV2.cancelBid(bidId);

        assertTrue(
            tellerV2.cancelBidWasCalled(),
            "Cancel bid internal was not called"
        );
    }

    function test_cancel_bid_invalid_owner() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        tellerV2.setMockMsgSenderForMarket(address(lender));
        tellerV2.mock_setBidState(bidId, BidState.PENDING);

        vm.expectRevert(
            abi.encodeWithSelector(
                ActionNotAllowed.selector,
                bidId,
                "CB",
                "Not bid owner"
            )
        );
        tellerV2.cancelBid(bidId);
    }

    function test_cancel_bid_internal_not_pending() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);

        //how can i specify the expected revert message??
        vm.expectRevert();
        tellerV2._cancelBidSuper(bidId);
    }

    function test_cancel_bid_internal() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        tellerV2.mock_setBidState(bidId, BidState.PENDING);

        tellerV2._cancelBidSuper(bidId);

        BidState state = tellerV2.getBidState(bidId);

        require(state == BidState.CANCELLED, "bid was not cancelled");
    }

    function test_market_owner_cancel_bid() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        //need to mock set market owner

        tellerV2.setMarketRegistrySuper(address(marketRegistryMock));
        marketRegistryMock.setMarketOwner(address(marketOwner));

        //why doesnt this work ?
        vm.prank(address(marketOwner));

        tellerV2.marketOwnerCancelBid(bidId);

        assertTrue(
            tellerV2.cancelBidWasCalled(),
            "Cancel bid internal was not called"
        );
    }

    function test_market_owner_cancel_bid_invalid_owner() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        vm.expectRevert();

        tellerV2.marketOwnerCancelBid(bidId);
    }

    function test_lender_accept_bid() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        tellerV2.mock_initialize(); //set address this as owner

        lendingToken.approve(address(tellerV2), 1e20);

        //make address (this) be the one that makes the payment
        tellerV2.setMockMsgSenderForMarket(address(this));

        tellerV2.mock_setBidState(bidId, BidState.PENDING);

        tellerV2.setMarketRegistrySuper(address(marketRegistryMock));
        marketRegistryMock.setMarketFeeRecipient(address(feeRecipient));

        tellerV2.setCollateralManagerSuper(address(collateralManagerMock));

        tellerV2.lenderAcceptBid(bidId);

        assertTrue(
            collateralManagerMock.deployAndDepositWasCalled(),
            "deploy and deposit was not called"
        );
    }

    function test_lender_accept_bid_invalid_state() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);

        tellerV2.setMarketRegistrySuper(address(marketRegistryMock));

        tellerV2.setCollateralManagerSuper(address(collateralManagerMock));

        vm.expectRevert(
            abi.encodeWithSelector(
                ActionNotAllowed.selector,
                bidId,
                "lab",
                "Bid not pending"
            )
        );

        tellerV2.lenderAcceptBid(bidId);
    }

    function test_lender_accept_bid_when_paused() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        tellerV2.mock_initialize(); //set address this as owner

        lendingToken.approve(address(tellerV2), 1e20);

        //make address (this) be the one that makes the payment
        tellerV2.setMockMsgSenderForMarket(address(this));

        tellerV2.mock_setBidState(bidId, BidState.PENDING);

        tellerV2.setMarketRegistrySuper(address(marketRegistryMock));
        marketRegistryMock.setMarketFeeRecipient(address(feeRecipient));

        tellerV2.setCollateralManagerSuper(address(collateralManagerMock));



        tellerV2.setProtocolPausingManagerSuper(address(protocolPausingManager));
        protocolPausingManager.pauseProtocol();
        //tellerV2.pauseProtocol();

        vm.expectRevert("Protocol paused");

        tellerV2.lenderAcceptBid(bidId);
    }

    function test_lender_accept_bid_when_not_verified() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        tellerV2.mock_initialize(); //set address this as owner

        lendingToken.approve(address(tellerV2), 1e20);

        //make address (this) be the one that makes the payment
        tellerV2.setMockMsgSenderForMarket(address(this));

        tellerV2.mock_setBidState(bidId, BidState.PENDING);

        tellerV2.setMarketRegistrySuper(address(marketRegistryMock));
        marketRegistryMock.setMarketFeeRecipient(address(feeRecipient));

        tellerV2.setCollateralManagerSuper(address(collateralManagerMock));

        marketRegistryMock.mock_setLenderIsVerified(false);

        vm.expectRevert(); //NV 

        tellerV2.lenderAcceptBid(bidId);
    }

    function test_lender_accept_bid_when_market_closed() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        tellerV2.mock_initialize(); //set address this as owner

        lendingToken.approve(address(tellerV2), 1e20);

        //make address (this) be the one that makes the payment
        tellerV2.setMockMsgSenderForMarket(address(this));

        tellerV2.mock_setBidState(bidId, BidState.PENDING);

        tellerV2.setMarketRegistrySuper(address(marketRegistryMock));
        marketRegistryMock.setMarketFeeRecipient(address(feeRecipient));

        tellerV2.setCollateralManagerSuper(address(collateralManagerMock));

        marketRegistryMock.mock_setGlobalMarketsClosed(true);

        vm.expectRevert("Market is closed");

        tellerV2.lenderAcceptBid(bidId);
    }

    function test_lender_accept_bid_when_loan_expired() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        tellerV2.mock_initialize(); //set address this as owner

        lendingToken.approve(address(tellerV2), 1e20);

        //make address (this) be the one that makes the payment
        tellerV2.setMockMsgSenderForMarket(address(this));

        tellerV2.mock_setBidState(bidId, BidState.PENDING);

        tellerV2.setMarketRegistrySuper(address(marketRegistryMock));
        marketRegistryMock.setMarketFeeRecipient(address(feeRecipient));

        tellerV2.setCollateralManagerSuper(address(collateralManagerMock));

        tellerV2.mock_setBidExpirationTime(bidId, 1000);

        vm.warp(20000);

        vm.expectRevert() ;//BE ;

        tellerV2.lenderAcceptBid(bidId);
    }

    function test_repay_loan_internal() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        //set address(this) as the account that will be paying off the loan
        tellerV2.setMockMsgSenderForMarket(address(this));

        tellerV2.setReputationManagerSuper(address(reputationManagerMock));

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);
        vm.warp(2000);

        Payment memory payment = Payment({ principal: 90, interest: 10 });

        lendingToken.approve(address(tellerV2), 1e20);

        tellerV2._repayLoanSuper(bidId, payment, 100, false);

        BidState bidStateAfter = tellerV2.getBidState(bidId);

        require(bidStateAfter == BidState.PAID, "Should set state to PAID");
    }

    function test_repay_loan_internal_leave_state_as_liquidated() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        //set address(this) as the account that will be paying off the loan
        tellerV2.setMockMsgSenderForMarket(address(this));

        tellerV2.setReputationManagerSuper(address(reputationManagerMock));

        tellerV2.mock_setBidState(bidId, BidState.LIQUIDATED);
        vm.warp(2000);

        Payment memory payment = Payment({ principal: 90, interest: 10 });

        lendingToken.approve(address(tellerV2), 1e20);

        tellerV2._repayLoanSuper(bidId, payment, 100, false);

        BidState bidStateAfter = tellerV2.getBidState(bidId);

        require(
            bidStateAfter == BidState.LIQUIDATED,
            "Should retain state as LIQUIDATED"
        );
    }

    function test_repay_loan_minimum() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);
        vm.warp(2000);

        //set the account that will be paying the loan off
        tellerV2.setMockMsgSenderForMarket(address(this));

        //need to get some weth

        lendingToken.approve(address(tellerV2), 1e20);

        tellerV2.repayLoanMinimum(bidId);

        assertTrue(tellerV2.repayLoanWasCalled(), "repay loan was not called");
    }

    function test_repay_loan_minimum_invalid_balance() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);

        //need to get some weth
        vm.prank(address(borrower));
        //expect arithmetic overflow
        vm.expectRevert();

        tellerV2.repayLoanMinimum(bidId);
    }

    function test_repay_loan_minimum_invalid_state() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        tellerV2.mock_setBidState(bidId, BidState.PENDING);

        //how can i fill in a specific revert message ?
        vm.expectRevert();
        tellerV2.repayLoanMinimum(bidId);
    }

    function test_repay_loan() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);
        vm.warp(2000);

        //set the account that will be paying the loan off
        tellerV2.setMockMsgSenderForMarket(address(this));

        lendingToken.approve(address(tellerV2), 1e20);

        tellerV2.repayLoan(bidId, 100);

        assertTrue(tellerV2.repayLoanWasCalled(), "repay loan was not called");
    }

    function test_repay_loan_invalid_state() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        tellerV2.mock_setBidState(bidId, BidState.PENDING);

        //set the account that will be paying the loan off
        tellerV2.setMockMsgSenderForMarket(address(this));

        lendingToken.approve(address(tellerV2), 1e20);

        vm.expectRevert(
            abi.encodeWithSelector(
                ActionNotAllowed.selector,
                bidId,
                "rl",
                "Loan not accepted"
            )
        );

        tellerV2.repayLoan(bidId, 100);
    }

    function test_repay_loan_full() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);

        vm.warp(2000);

        //set the account that will be paying the loan off
        tellerV2.setMockMsgSenderForMarket(address(this));

        lendingToken.approve(address(tellerV2), 1e20);

        tellerV2.repayLoanFull(bidId);

        assertTrue(tellerV2.repayLoanWasCalled(), "repay loan was not called");
    }

    function test_repay_loan_full_invalid_state() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        tellerV2.mock_setBidState(bidId, BidState.PENDING);

        //set the account that will be paying the loan off
        tellerV2.setMockMsgSenderForMarket(address(this));

        lendingToken.approve(address(tellerV2), 1e20);

        vm.expectRevert(
            abi.encodeWithSelector(
                ActionNotAllowed.selector,
                bidId,
                "rl",
                "Loan not accepted"
            )
        );

        tellerV2.repayLoanFull(bidId);
    }

    function test_lender_close_loan_not_lender() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);
        tellerV2.mock_setBidDefaultDuration(bidId, 1000);
        vm.warp(2000 * 1e20);

        vm.expectRevert(); // NLL
        vm.prank(address(borrower));
        tellerV2.lenderCloseLoan(bidId);
    }

       
    function test_lender_close_loan() public {
        uint256 bidId = 1;
        setMockBid(bidId);

         //set the account that will be paying the loan off
       // tellerV2.setMockMsgSenderForMarket(address(lender));

        tellerV2.setCollateralManagerSuper(address(collateralManagerMock));
        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);
        tellerV2.mock_setBidDefaultDuration(bidId, 1000);
        vm.warp(2000 * 1e20);

        vm.prank(address(lender));
        tellerV2.lenderCloseLoan(bidId);

        // make sure the state is now CLOSED
        BidState state = tellerV2.getBidState(bidId);
        require(state == BidState.CLOSED, "bid was not closed");
    }

    function test_lender_close_loan_wrong_origin() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        tellerV2.setCollateralManagerSuper(address(collateralManagerMock));

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);
        vm.warp(2000 * 1e20);

        tellerV2.mock_setBidDefaultDuration(bidId, 1000);

        //set the account that will be paying the loan off
        tellerV2.setMockMsgSenderForMarket(address(borrower));

        lendingToken.approve(address(tellerV2), 1e20);

        vm.expectRevert(); // NLL
        tellerV2.lenderCloseLoan(bidId);
    }

    function test_liquidate_loan_full() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        tellerV2.setCollateralManagerSuper(address(collateralManagerMock));
        //tellerV2.setReputationManagerSuper(address(reputationManagerMock));

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);
        vm.warp(2000 * 1e20);

        tellerV2.mock_setBidDefaultDuration(bidId, 1000);

        //set the account that will be paying the loan off
        tellerV2.setMockMsgSenderForMarket(address(this));

        lendingToken.approve(address(tellerV2), 1e20);

        tellerV2.liquidateLoanFull(bidId);

        assertTrue(tellerV2.repayLoanWasCalled(), "repay loan was not called");

        BidState bidStateAfter = tellerV2.getBidState(bidId);

        require(
            bidStateAfter == BidState.LIQUIDATED,
            "invalid bid state after liquidate loan full"
        );
    }

    function test_liquidate_loan_full_invalid_state() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        tellerV2.setCollateralManagerSuper(address(collateralManagerMock));

        tellerV2.mock_setBidState(bidId, BidState.PAID);
        vm.warp(2000 * 1e20);

        tellerV2.mock_setBidDefaultDuration(bidId, 1000);

        //set the account that will be paying the loan off
        tellerV2.setMockMsgSenderForMarket(address(this));

        lendingToken.approve(address(tellerV2), 1e20);

        vm.expectRevert(
            abi.encodeWithSelector(
                ActionNotAllowed.selector,
                bidId,
                "ll",
                "Loan not accepted"
            )
        );
        tellerV2.liquidateLoanFull(bidId);
    }

    /*
    This specifically works in conjunction with the lender manager 
    */
    function test_claim_loan_nft() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        tellerV2.setProtocolPausingManagerSuper(address(protocolPausingManager));

        tellerV2.setLenderManagerSuper(address(lenderManagerMock));

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);

        tellerV2.setMockMsgSenderForMarket(address(lender));
        vm.prank(address(lender));

        tellerV2.claimLoanNFT(bidId);

        //assert
    }

    function test_claim_loan_nft_invalid_as_borrower() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        tellerV2.setProtocolPausingManagerSuper(address(protocolPausingManager));
       

        tellerV2.setLenderManagerSuper(address(lenderManagerMock));

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);

        tellerV2.setMockMsgSenderForMarket(address(borrower));
        vm.prank(address(borrower));

        vm.expectRevert("NV Lender");

        tellerV2.claimLoanNFT(bidId);
    }

    function test_claim_loan_nft_invalid_when_paused() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        tellerV2.mock_setLenderManager(address(lenderManagerMock));

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);

        tellerV2.setMockMsgSenderForMarket(address(lender));

        tellerV2.mock_initialize();
        
        tellerV2.setProtocolPausingManagerSuper(address(protocolPausingManager));
        protocolPausingManager.pauseProtocol();
       

        vm.expectRevert("Protocol paused");
        vm.prank(address(lender));
        tellerV2.claimLoanNFT(bidId);
    }
}

contract User {}
