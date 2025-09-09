// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/utils/math/SafeCast.sol";

import "./Testable.sol";
import "../contracts/TellerV2.sol";

contract PMT_Test is Testable, TellerV2 {
    Bid __bid;

    constructor() TellerV2(address(0)) {}

    function test_01_pmt() public {
        __bid.loanDetails.principal = 10000e6; // 10k USDC
        __bid.loanDetails.loanDuration = 365 days * 3; // 3 years
        __bid.terms.paymentCycle = 365 days / 12; // 1 month
        __bid.terms.APR = 300; // 3.0%
        pmt_runner(290812096, 365 days);
    }

    function test_02_pmt() public {
        __bid.loanDetails.principal = 100000e6; // 100x USDC
        __bid.loanDetails.loanDuration = 365 days * 10; // 10 years
        __bid.terms.paymentCycle = 365 days / 12; // 1 month
        __bid.terms.APR = 800; // 8.0%
        pmt_runner(1213275944, 365 days);
    }

    function test_03_pmt() public {
        __bid.loanDetails.principal = 100000e6; // 100x USDC
        __bid.loanDetails.loanDuration = 365 days * 10; // 10 years
        __bid.terms.paymentCycle = 365 days / 12; // 1 month
        __bid.terms.APR = 0; // 0.0%
        pmt_runner(833333334, 365 days);
    }

    function test_04_pmt() public {
        __bid.loanDetails.principal = 100000e6; // 100x USDC
        __bid.loanDetails.loanDuration = 45 days; // 45 days
        __bid.terms.paymentCycle = 30 days; // 1 month
        __bid.terms.APR = 600; // 6.0%
        pmt_runner(50370166243, 365 days);
    }

    function pmt_runner(uint256 _expected, uint256 _daysInYear) private {
        uint256 pmt = NumbersLib.pmt(
            __bid.loanDetails.principal,
            __bid.loanDetails.loanDuration,
            __bid.terms.paymentCycle,
            __bid.terms.APR,
            _daysInYear
        );
        assertEq(pmt, _expected, "Loan payment for cycle incorrect");
    }
}
