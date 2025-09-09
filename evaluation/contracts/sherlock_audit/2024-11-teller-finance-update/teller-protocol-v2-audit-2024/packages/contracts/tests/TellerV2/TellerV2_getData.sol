// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { StdStorage, stdStorage } from "forge-std/StdStorage.sol";
import { Testable } from "../Testable.sol";
import { TellerV2_Override } from "./TellerV2_Override.sol";
import { Bid, BidState, Collateral, Payment, LoanDetails, Terms } from "../../contracts/TellerV2.sol";

import "@openzeppelin/contracts-upgradeable/utils/StringsUpgradeable.sol";

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

import { PaymentType, PaymentCycleType } from "../../contracts/libraries/V2Calculations.sol";

import { ReputationManagerMock } from "../../contracts/mock/ReputationManagerMock.sol";
import { CollateralManagerMock } from "../../contracts/mock/CollateralManagerMock.sol";
import { LenderManagerMock } from "../../contracts/mock/LenderManagerMock.sol";
import { MarketRegistryMock } from "../../contracts/mock/MarketRegistryMock.sol";

import "../../lib/forge-std/src/console.sol";

contract TellerV2_initialize is Testable {
    TellerV2_Override tellerV2;

    User borrower;
    User lender;
    User receiver;

    ERC20 lendingToken;

    LenderManagerMock lenderManagerMock;

    function setUp() public {
        tellerV2 = new TellerV2_Override();

        lendingToken = new ERC20("Wrapped Ether", "WETH");

        lenderManagerMock = new LenderManagerMock();

        borrower = new User();
        lender = new User();
    }

    /*

    TODO 


    //these can just be tested w the calculation contract-- not sure why coverage tool says they are not tested 
    FNDA:0,TellerV2.calculateAmountOwed
    FNDA:0,TellerV2.calculateAmountDue
    FNDA:0,TellerV2.calculateAmountDue
    FNDA:0,TellerV2.calculateNextDueDate

  
    FNDA:0,TellerV2.getLoanLender  

 
    FNDA:0,TellerV2.getBorrowerActiveLoanIds
    FNDA:0,TellerV2.getBorrowerLoanIds

    */

    function setMockBid(uint256 bidId) public {
        tellerV2.mock_setBid(
            bidId,
            Bid({
                borrower: address(borrower),
                lender: address(lender),
                receiver: address(receiver),
                marketplaceId: 100,
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

  /*  function test_getMetadataURI_without_mapping() public {
        uint256 bidId = 1;
        setMockBid(1);

        string memory uri = tellerV2.getMetadataURI(bidId);

        //why is this true ?
        uint256 expectedUri = 0x3078313233340000000000000000000000000000000000000000000000000000;

        // expect deprecated bytes32 uri as a string
        assertEq(uri, StringsUpgradeable.toHexString(expectedUri, 32));
    }

    function test_getMetadataURI_with_mapping() public {
        uint256 bidId = 1;
        setMockBid(1);

        tellerV2.mock_addUriToMapping(bidId, "0x1234");

        string memory uri = tellerV2.getMetadataURI(bidId);

        assertEq(uri, "0x1234");
    }*/

    function test_isLoanLiquidateable_with_valid_bid() public {
        uint256 bidId = 1;
        setMockBid(1);

        bool liquidateable = tellerV2.isLoanLiquidateable(bidId);

        assertEq(liquidateable, false);
    }

    function test_isLoanLiquidateable_with_very_old_bid() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        //set to accepted
        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);

        tellerV2.mock_setBidDefaultDuration(bidId, 1000);

        //fast forward timestamp
        vm.warp(1000000000);

        bool liquidateable = tellerV2.isLoanLiquidateable(bidId);

        assertEq(liquidateable, true);
    }

    function test_isLoanLiquidateable_when_repaid_sooner() public {
        uint256 bidId = 1;
        setMockBid(bidId);

        //set to accepted
        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);

        tellerV2.mock_setBidDefaultDuration(bidId, 1000);
        tellerV2.mock_setBidLastRepaidTimestamp(bidId, 1000);

        vm.warp(3110);
        bool defaulted = tellerV2.isLoanDefaulted(bidId);
        bool liquidateable = tellerV2.isLoanLiquidateable(bidId);

        assertEq(defaulted, false);
        assertEq(liquidateable, false);

        //fast forward timestamp past the  accepted time + payment cycle + default duration
        vm.warp(5110);
        defaulted = tellerV2.isLoanDefaulted(bidId);
        liquidateable = tellerV2.isLoanLiquidateable(bidId);

        assertEq(defaulted, true);
        assertEq(liquidateable, false);

        vm.warp(5110 + 1 days);
        defaulted = tellerV2.isLoanDefaulted(bidId);
        liquidateable = tellerV2.isLoanLiquidateable(bidId);

        assertEq(defaulted, true);
        assertEq(liquidateable, true);
    }

    function test_lastRepaidTimestamp() public {
        uint256 bidId = 1;
        setMockBid(1);

        uint256 lastRepaidTimestamp = tellerV2.lastRepaidTimestamp(bidId);

        assertEq(lastRepaidTimestamp, 100);
    }

    function test_getLoanLender_without_nft() public {
        uint256 bidId = 1;

        tellerV2.mock_setBid(
            bidId,
            Bid({
                borrower: address(borrower),
                lender: address(lender),
                receiver: address(receiver),
                marketplaceId: 100,
                _metadataURI: "0x1234",
                loanDetails: LoanDetails({
                    lendingToken: lendingToken,
                    principal: 100,
                    timestamp: 100,
                    acceptedTimestamp: 100,
                    lastRepaidTimestamp: 2000,
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

        tellerV2.mock_setLenderManager(address(lenderManagerMock));

        address result = tellerV2.getLoanLender(bidId);

        assertEq(
            result,
            address(lender),
            "getLoanLender did not return correct result"
        );
    }

    function test_getLoanLender_with_nft() public {
        uint256 bidId = 1;

        tellerV2.mock_setBid(
            bidId,
            Bid({
                borrower: address(borrower),
                lender: address(lenderManagerMock),
                receiver: address(receiver),
                marketplaceId: 100,
                _metadataURI: "0x1234",
                loanDetails: LoanDetails({
                    lendingToken: lendingToken,
                    principal: 100,
                    timestamp: 100,
                    acceptedTimestamp: 100,
                    lastRepaidTimestamp: 2000,
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

        tellerV2.mock_setLenderManager(address(lenderManagerMock));

        lenderManagerMock.registerLoan(bidId, address(this));
        address result = tellerV2.getLoanLender(bidId);
        assertEq(
            result,
            lenderManagerMock.ownerOf(bidId),
            "getLoanLender did not return correct result"
        );
    }

    function test_getLoanLendingToken() public {
        uint256 bidId = 1;
        setMockBid(1);

        address lendingToken = tellerV2.getLoanLendingToken(bidId);

        assertEq(lendingToken, lendingToken, "unexpected lending token");
    }

    function test_getLoanMarketId() public {
        uint256 bidId = 1;
        setMockBid(1);

        uint256 marketId = tellerV2.getLoanMarketId(bidId);

        assertEq(marketId, 100);
    }

    function test_getLoanSummary() public {
        uint256 bidId = 1;
        setMockBid(1);

        (
            address borrower,
            address lender,
            uint256 marketId,
            address principalTokenAddress,
            uint256 principalAmount,
            uint32 acceptedTimestamp,
            uint32 loanDuration,
            BidState bidState
        ) = tellerV2.getLoanSummary(bidId);

        assertEq(
            borrower,
            address(borrower),
            "unexpected borrower from summary"
        );
        assertEq(lender, address(lender), "unexpected lender from summary");
        assertEq(marketId, 100, "unexpected marketId from summary");
    }

    function test_getLoanSummary_lender_manager() public {
        uint256 bidId = 1;
        setMockBid(1);

        tellerV2.mock_setLenderManager(address(lenderManagerMock));

        lenderManagerMock.registerLoan(bidId, address(this));

        (
            address borrower,
            address lender,
            uint256 marketId,
            address principalTokenAddress,
            uint256 principalAmount,
            uint32 acceptedTimestamp,
            uint32 lastRepaidTimestamp,
            BidState bidState
        ) = tellerV2.getLoanSummary(bidId);

        assertEq(
            borrower,
            address(borrower),
            "unexpected borrower from summary"
        );
        assertEq(
            lender,
            tellerV2.getLoanLender(bidId),
            "unexpected lender from summary"
        );
        assertEq(marketId, 100, "unexpected marketId from summary");
    }

    /*
        there are many branches of this to test 
    */

    function test_calculateAmountDue_without_timestamp() public {
        uint256 bidId = 1;

        tellerV2.mock_setBid(
            bidId,
            Bid({
                borrower: address(borrower),
                lender: address(lender),
                receiver: address(receiver),
                marketplaceId: 100,
                _metadataURI: "0x1234",
                loanDetails: LoanDetails({
                    lendingToken: lendingToken,
                    principal: 100,
                    timestamp: 100,
                    acceptedTimestamp: 100,
                    lastRepaidTimestamp: 2000,
                    loanDuration: 5000,
                    totalRepaid: Payment({ principal: 0, interest: 5 })
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

        vm.warp(2500);

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);

        Payment memory amountDue = tellerV2.calculateAmountDue(
            bidId,
            block.timestamp
        );

        assertEq(amountDue.principal, 2);
    }

    function test_calculateAmountDue_without_timestamp_not_accepted() public {
        uint256 bidId = 1;

        tellerV2.mock_setBid(
            bidId,
            Bid({
                borrower: address(borrower),
                lender: address(lender),
                receiver: address(receiver),
                marketplaceId: 100,
                _metadataURI: "0x1234",
                loanDetails: LoanDetails({
                    lendingToken: lendingToken,
                    principal: 100,
                    timestamp: 100,
                    acceptedTimestamp: 100,
                    lastRepaidTimestamp: 2000,
                    loanDuration: 5000,
                    totalRepaid: Payment({ principal: 0, interest: 5 })
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

        tellerV2.mock_setBidState(bidId, BidState.PENDING);

        Payment memory amountDue = tellerV2.calculateAmountDue(
            bidId,
            block.timestamp
        );

        assertEq(amountDue.principal, 0);
    }

    function test_calculateAmountDue_with_timestamp() public {
        uint256 bidId = 1;

        tellerV2.mock_setBid(
            bidId,
            Bid({
                borrower: address(borrower),
                lender: address(lender),
                receiver: address(receiver),
                marketplaceId: 100,
                _metadataURI: "0x1234",
                loanDetails: LoanDetails({
                    lendingToken: lendingToken,
                    principal: 100,
                    timestamp: 100,
                    acceptedTimestamp: 100,
                    lastRepaidTimestamp: 2000,
                    loanDuration: 5000,
                    totalRepaid: Payment({ principal: 0, interest: 5 })
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

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);

        Payment memory amountDue = tellerV2.calculateAmountDue(bidId, 5000);

        assertEq(amountDue.principal, 100);
    }

    function test_calculateAmountDue_with_timestamp_not_accepted() public {
        uint256 bidId = 1;

        tellerV2.mock_setBid(
            bidId,
            Bid({
                borrower: address(borrower),
                lender: address(lender),
                receiver: address(receiver),
                marketplaceId: 100,
                _metadataURI: "0x1234",
                loanDetails: LoanDetails({
                    lendingToken: lendingToken,
                    principal: 100,
                    timestamp: 100,
                    acceptedTimestamp: 100,
                    lastRepaidTimestamp: 2000,
                    loanDuration: 5000,
                    totalRepaid: Payment({ principal: 0, interest: 5 })
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

        tellerV2.mock_setBidState(bidId, BidState.PENDING);

        Payment memory amountDue = tellerV2.calculateAmountDue(bidId, 1000);

        assertEq(amountDue.principal, 0);
    }

    function test_calculateAmountOwed_without_timestamp() public {
        uint256 bidId = 1;
        tellerV2.mock_setBid(
            bidId,
            Bid({
                borrower: address(borrower),
                lender: address(lender),
                receiver: address(receiver),
                marketplaceId: 100,
                _metadataURI: "0x1234",
                loanDetails: LoanDetails({
                    lendingToken: lendingToken,
                    principal: 100,
                    timestamp: 100,
                    acceptedTimestamp: 100,
                    lastRepaidTimestamp: 2000,
                    loanDuration: 5000,
                    totalRepaid: Payment({ principal: 0, interest: 5 })
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

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);
        vm.warp(2500);

        Payment memory amountOwed = tellerV2.calculateAmountOwed(
            bidId,
            block.timestamp
        );

        assertEq(amountOwed.principal, 100);
    }

    function test_calculateAmountOwed_with_timestamp() public {
        uint256 bidId = 1;
        tellerV2.mock_setBid(
            bidId,
            Bid({
                borrower: address(borrower),
                lender: address(lender),
                receiver: address(receiver),
                marketplaceId: 100,
                _metadataURI: "0x1234",
                loanDetails: LoanDetails({
                    lendingToken: lendingToken,
                    principal: 100,
                    timestamp: 100,
                    acceptedTimestamp: 100,
                    lastRepaidTimestamp: 2000,
                    loanDuration: 5000,
                    totalRepaid: Payment({ principal: 0, interest: 5 })
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

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);

        Payment memory amountOwed = tellerV2.calculateAmountOwed(bidId, 2500);

        assertEq(amountOwed.principal, 100);
    }

    function test_calculateNextDueDate_not_accepted() public {
        uint256 bidId = 1;
        setMockBid(1);

        tellerV2.mock_setBidState(bidId, BidState.PENDING);

        uint256 nextDueDate = tellerV2.calculateNextDueDate(bidId);

        assertEq(nextDueDate, 0);
    }

    function test_calculateNextDueDate_monthly() public {
        uint256 bidId = 1;

        tellerV2.mock_setBid(
            bidId,
            Bid({
                borrower: address(borrower),
                lender: address(lender),
                receiver: address(receiver),
                marketplaceId: 100,
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

        tellerV2.mock_setBidPaymentCycleType(bidId, PaymentCycleType.Monthly);

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);

        uint256 nextDueDate = tellerV2.calculateNextDueDate(bidId);

        assertEq(nextDueDate, 5100, "unexpected due date");
    }

    function test_calculateNextDueDate_monthly_after_repayment() public {
        uint256 bidId = 1;

        tellerV2.mock_setBid(
            bidId,
            Bid({
                borrower: address(borrower),
                lender: address(lender),
                receiver: address(receiver),
                marketplaceId: 100,
                _metadataURI: "0x1234",
                loanDetails: LoanDetails({
                    lendingToken: lendingToken,
                    principal: 100,
                    timestamp: 100,
                    acceptedTimestamp: 100,
                    lastRepaidTimestamp: 2000,
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

        tellerV2.mock_setBidPaymentCycleType(bidId, PaymentCycleType.Monthly);

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);

        uint256 nextDueDate = tellerV2.calculateNextDueDate(bidId);

        assertEq(nextDueDate, 5100, "unexpected due date");
    }

    function test_calculateNextDueDate_seconds() public {
        uint256 bidId = 1;

        tellerV2.mock_setBid(
            bidId,
            Bid({
                borrower: address(borrower),
                lender: address(lender),
                receiver: address(receiver),
                marketplaceId: 100,
                _metadataURI: "0x1234",
                loanDetails: LoanDetails({
                    lendingToken: lendingToken,
                    principal: 100,
                    timestamp: 100,
                    acceptedTimestamp: 100,
                    lastRepaidTimestamp: 2000,
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

        tellerV2.mock_setBidPaymentCycleType(bidId, PaymentCycleType.Seconds);

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);

        uint256 nextDueDate = tellerV2.calculateNextDueDate(bidId);

        assertEq(nextDueDate, 4100, "unexpected due date");
    }

    function test_calculateNextDueDate_lastPaymentCycle() public {
        uint256 bidId = 1;

        tellerV2.mock_setBid(
            bidId,
            Bid({
                borrower: address(borrower),
                lender: address(lender),
                receiver: address(receiver),
                marketplaceId: 100,
                _metadataURI: "0x1234",
                loanDetails: LoanDetails({
                    lendingToken: lendingToken,
                    principal: 100,
                    timestamp: 100,
                    acceptedTimestamp: 100,
                    lastRepaidTimestamp: 4000,
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

        //vm.warp(5200);

        tellerV2.mock_setBidPaymentCycleType(bidId, PaymentCycleType.Seconds);

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);

        uint256 nextDueDate = tellerV2.calculateNextDueDate(bidId);

        uint256 endOfLoanExpected = 100 + 5000;

        assertEq(nextDueDate, endOfLoanExpected, "unexpected due date");
    }

    function test_isPaymentLate_invalid_state() public {
        uint256 bidId = 1;
        setMockBid(1);

        tellerV2.mock_setBidState(bidId, BidState.PENDING);

        bool isLate = tellerV2.isPaymentLate(1);

        assertEq(isLate, false, "unexpected late status");
    }

    function test_isPaymentLate() public {
        uint256 bidId = 1;
        setMockBid(1);

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);

        bool isLate = tellerV2.isPaymentLate(1);

        assertEq(isLate, false, "unexpected late status");
    }

    //test all branches
    function test_canLiquidateLoan_internal() public {
        uint256 bidId = 1;
        setMockBid(1);

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);
        tellerV2.mock_setBidDefaultDuration(bidId, 1000);

        vm.warp(1e10);

        bool canLiq = tellerV2._canLiquidateLoanSuper(bidId, 500);

        assertEq(canLiq, true, "unexpected liquidation status");
    }

    function test_canLiquidateLoan_internal_invalid_state() public {
        uint256 bidId = 1;
        setMockBid(1);

        tellerV2.mock_setBidState(bidId, BidState.PENDING);
        tellerV2.mock_setBidDefaultDuration(bidId, 1000);

        bool canLiq = tellerV2._canLiquidateLoanSuper(bidId, 500);

        assertEq(canLiq, false, "unexpected liquidation status");
    }

    function test_canLiquidateLoan_internal_zero_default_duration() public {
        uint256 bidId = 1;
        setMockBid(1);

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);
        tellerV2.mock_setBidDefaultDuration(bidId, 0);

        vm.warp(1e10);

        bool canLiq = tellerV2._canLiquidateLoanSuper(bidId, 500);

        assertEq(canLiq, true, "unexpected liquidation status");
    }

    function test_canLiquidateLoan_internal_false() public {
        uint256 bidId = 1;
        setMockBid(1);

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);
        tellerV2.mock_setBidDefaultDuration(bidId, 10000000);

        vm.warp(50000);

        bool canLiq = tellerV2._canLiquidateLoanSuper(bidId, 5500);

        assertEq(canLiq, false, "unexpected liquidation status");
    }

    //test all branches
    function test_isLoanExpired() public {
        uint256 bidId = 1;
        setMockBid(1);

        tellerV2.mock_setBidState(bidId, BidState.PENDING);
        tellerV2.mock_setBidExpirationTime(bidId, 500);

        vm.warp(1e10);

        bool is_exp = tellerV2.isLoanExpired(bidId);
        assertEq(is_exp, true, "unexpected expiration status");
    }

    function test_isLoanExpired_false() public {
        uint256 bidId = 1;
        setMockBid(1);

        tellerV2.mock_setBidState(bidId, BidState.PENDING);
        tellerV2.mock_setBidDefaultDuration(bidId, 500);

        bool is_exp = tellerV2.isLoanExpired(bidId);
        assertEq(is_exp, false, "unexpected expiration status");
    }

    function test_isLoanExpired_invalid_state() public {
        uint256 bidId = 1;
        setMockBid(1);

        tellerV2.mock_setBidState(bidId, BidState.ACCEPTED);
        tellerV2.mock_setBidDefaultDuration(bidId, 500);

        vm.warp(1e10);

        bool is_exp = tellerV2.isLoanExpired(bidId);
        assertEq(is_exp, false, "unexpected expiration status");
    }

    function test_isLoanExpired_zero_default_duration() public {
        uint256 bidId = 1;
        setMockBid(1);

        tellerV2.mock_setBidState(bidId, BidState.PENDING);
        tellerV2.mock_setBidDefaultDuration(bidId, 0);

        vm.warp(1e10);

        bool is_exp = tellerV2.isLoanExpired(bidId);
        assertEq(is_exp, false, "unexpected expiration status");
    }
}

contract User {}
