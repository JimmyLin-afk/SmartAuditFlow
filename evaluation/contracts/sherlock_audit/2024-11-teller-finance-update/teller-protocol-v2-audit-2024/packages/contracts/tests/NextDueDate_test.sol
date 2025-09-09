pragma solidity ^0.8.0;
// SPDX-License-Identifier: MIT

import "./Testable.sol";
import "../contracts/TellerV2.sol";
import { BokkyPooBahsDateTimeLibrary as BPBDTL } from "../contracts/libraries/DateTimeLib.sol";

contract NextDueDate_Test is Testable, TellerV2 {
    Bid __bid;

    constructor() TellerV2(address(0)) {
        __bid.loanDetails.principal = 10000e6; // 10k USDC
        __bid.loanDetails.loanDuration = 365 days * 2; // 2 years
        __bid.terms.paymentCycle = 30 days; // 1 month
        __bid.terms.APR = 450; // 4.5%
        __bid.state = BidState.ACCEPTED;
    }

    function test_01_nextDueDate() public {
        __bid.loanDetails.acceptedTimestamp = uint32(
            BPBDTL.timestampFromDate(2020, 1, 31) // Leap year
        );
        bids[1] = __bid;
        bidPaymentCycleType[1] = PaymentCycleType.Monthly;
        // Expected date is Feb 29th
        uint32 expectedDate = uint32(BPBDTL.timestampFromDate(2020, 2, 29));
        nextDueDate_runner(expectedDate);
    }

    function test_02_nextDueDate() public {
        __bid.loanDetails.acceptedTimestamp = uint32(
            BPBDTL.timestampFromDate(2020, 2, 29)
        );
        bids[1] = __bid;
        bidPaymentCycleType[1] = PaymentCycleType.Monthly;
        // Expected date is March 29th
        uint32 expectedDate = uint32(BPBDTL.timestampFromDate(2020, 3, 29));
        nextDueDate_runner(expectedDate);
    }

    function test_03_nextDueDate() public {
        __bid.loanDetails.acceptedTimestamp = uint32(
            BPBDTL.timestampFromDate(2023, 2, 1)
        );
        bids[1] = __bid;
        bidPaymentCycleType[1] = PaymentCycleType.Monthly;
        // Expected date is March 1st
        uint32 expectedDate = uint32(BPBDTL.timestampFromDate(2023, 3, 1));
        nextDueDate_runner(expectedDate);
    }

    function test_04_nextDueDate() public {
        __bid.loanDetails.acceptedTimestamp = uint32(
            BPBDTL.timestampFromDate(2023, 1, 31)
        );
        __bid.loanDetails.lastRepaidTimestamp = uint32(
            BPBDTL.timestampFromDate(2023, 2, 1)
        );
        bids[1] = __bid;
        bidPaymentCycleType[1] = PaymentCycleType.Monthly;
        // Expected date is March 31st
        uint32 expectedDate = uint32(BPBDTL.timestampFromDate(2023, 3, 31));
        nextDueDate_runner(expectedDate);
    }

    function nextDueDate_runner(uint256 _expected) private {
        uint256 nextDueDate = calculateNextDueDate(1);
        assertEq(nextDueDate, _expected, "Next due date incorrect");
    }
}
