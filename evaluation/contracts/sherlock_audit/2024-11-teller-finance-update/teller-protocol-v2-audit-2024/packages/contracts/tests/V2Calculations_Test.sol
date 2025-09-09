// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/utils/math/Math.sol";
import "@openzeppelin/contracts/utils/Arrays.sol";
import "@openzeppelin/contracts/utils/structs/EnumerableSet.sol";

import "./Testable.sol";
import "../contracts/TellerV2.sol";
import { Bid } from "../contracts/TellerV2Storage.sol";
import { PaymentType } from "../contracts/libraries/V2Calculations.sol";

contract V2Calculations_Test is Testable {
    using Arrays for uint256[];
    using EnumerableSet for EnumerableSet.UintSet;

    Bid __bid;
    EnumerableSet.UintSet cyclesToSkip;
    uint256[] cyclesWithExtraPayments;
    uint256[] cyclesWithExtraPaymentsAmounts;

    constructor() {
        __bid.loanDetails.principal = 100000e6; // 100k
        __bid.loanDetails.loanDuration = 365 days * 3; // 3 years
        __bid.terms.paymentCycle = 365 days / 12; // 1 month
        __bid.terms.APR = 1000; // 10.0%
    }

    function setup_beforeAll() public {
        delete cyclesToSkip;
        delete cyclesWithExtraPayments;
        delete cyclesWithExtraPaymentsAmounts;
    }

    // EMI loan

    function test_baseline_calculateAmountOwed() public {
        calculateAmountOwed_runner(
            36, //the number of payment cycles expected
            PaymentType.EMI,
            PaymentCycleType.Seconds
        );
    }

    function test_01_calculateAmountOwed() public {
        cyclesToSkip.add(2); //rename to 'missed payments' ?
        cyclesWithExtraPayments = [3, 4];
        cyclesWithExtraPaymentsAmounts = [25000e6, 25000e6];

        calculateAmountOwed_runner(
            18, //the number of payment cycles expected
            PaymentType.EMI,
            PaymentCycleType.Seconds
        );
    }

    // EMI loan
    function test_02_calculateAmountOwed() public {
        cyclesToSkip.add(3);
        cyclesToSkip.add(4);
        cyclesToSkip.add(5);

        calculateAmountOwed_runner(
            36,
            PaymentType.EMI,
            PaymentCycleType.Seconds
        );
    }

    // EMI loan
    function test_03_calculateAmountOwed() public {
        cyclesWithExtraPayments = [3, 7];
        cyclesWithExtraPaymentsAmounts = [35000e6, 20000e6];

        calculateAmountOwed_runner(
            16,
            PaymentType.EMI,
            PaymentCycleType.Seconds
        );
    }

    // EMI loan - Monthly payment cycle
    function test_04_calculateAmountOwed() public {
        cyclesToSkip.add(5);
        cyclesToSkip.add(7);

        calculateAmountOwed_runner(
            36,
            PaymentType.EMI,
            PaymentCycleType.Monthly
        );
    }

    // EMI loan - Monthly payment cycle
    function test_05_calculateAmountOwed() public {
        cyclesWithExtraPayments = [2, 6];
        cyclesWithExtraPaymentsAmounts = [35000e6, 20000e6];

        calculateAmountOwed_runner(
            16,
            PaymentType.EMI,
            PaymentCycleType.Monthly
        );
    }

    // Bullet loan
    function test_06_calculateAmountOwed() public {
        cyclesToSkip.add(6);
        calculateAmountOwed_runner(
            36,
            PaymentType.Bullet,
            PaymentCycleType.Seconds
        );
    }

    // Bullet loan
    function test_07_calculateAmountOwed() public {
        cyclesToSkip.add(12);
        cyclesWithExtraPayments = [1, 8];
        cyclesWithExtraPaymentsAmounts = [15000e6, 10000e6];
        calculateAmountOwed_runner(
            36,
            PaymentType.Bullet,
            PaymentCycleType.Seconds
        );
    }

    // Bullet loan - Monthly payment cycle
    function test_08_calculateAmountOwed() public {
        cyclesToSkip.add(5);
        calculateAmountOwed_runner(
            36,
            PaymentType.Bullet,
            PaymentCycleType.Monthly
        );
    }

    // Bullet loan - Monthly paymenty cycle
    function test_09_calculateAmountOwed() public {
        cyclesToSkip.add(8);
        cyclesWithExtraPayments = [3];
        cyclesWithExtraPaymentsAmounts = [13000e6];
        calculateAmountOwed_runner(
            36,
            PaymentType.Bullet,
            PaymentCycleType.Monthly
        );
    }

    function calculateAmountOwed_runner(
        uint256 expectedTotalCycles,
        PaymentType _paymentType,
        PaymentCycleType _paymentCycleType
    ) private {
        // Calculate payment cycle amount
        uint256 paymentCycleAmount = V2Calculations.calculatePaymentCycleAmount(
            _paymentType,
            _paymentCycleType,
            __bid.loanDetails.principal,
            __bid.loanDetails.loanDuration,
            __bid.terms.paymentCycle,
            __bid.terms.APR
        );

        //need this here or else it defaults to EMI !
        __bid.paymentType = _paymentType;

        uint32 _paymentCycleDuration = __bid.terms.paymentCycle;

        // Set the bid's payment cycle amount
        __bid.terms.paymentCycleAmount = paymentCycleAmount;
        // Set accepted bid timestamp to now
        __bid.loanDetails.acceptedTimestamp = uint32(block.timestamp);

        uint256 nowTimestamp = block.timestamp;
        uint256 skippedPaymentCounter;
        uint256 owedPrincipal = __bid.loanDetails.principal;
        uint256 cycleCount = Math.ceilDiv(
            __bid.loanDetails.loanDuration,
            __bid.terms.paymentCycle
        );
        uint256 cycleIndex;
        while (owedPrincipal > 0) {
            // Increment cycle index
            cycleIndex++;

            // Increase timestamp
            nowTimestamp += __bid.terms.paymentCycle;

            uint256 duePrincipal;
            uint256 interest;
            (owedPrincipal, duePrincipal, interest) = V2Calculations
                .calculateAmountOwed(
                    __bid,
                    nowTimestamp,
                    _paymentCycleType,
                    _paymentCycleDuration
                );

            // Check if we should skip this cycle for payments
            if (cyclesToSkip.length() > 0) {
                if (cyclesToSkip.contains(cycleIndex)) {
                    // Add this cycle's payment amount to the next cycle's expected payment
                    skippedPaymentCounter++;
                    continue;
                }
            }

            skippedPaymentCounter = 0;

            uint256 extraPaymentAmount;
            // Add additional payment amounts for cycles
            if (cyclesWithExtraPayments.length > 0) {
                uint256 index = cyclesWithExtraPayments.findUpperBound(
                    cycleIndex
                );
                if (
                    index < cyclesWithExtraPayments.length &&
                    cyclesWithExtraPayments[index] == cycleIndex
                ) {
                    extraPaymentAmount = cyclesWithExtraPaymentsAmounts[index];
                }
            }

            // Mark repayment amounts
            uint256 principalPayment;
            principalPayment = duePrincipal + extraPaymentAmount;
            if (principalPayment > 0) {
                __bid.loanDetails.totalRepaid.principal += principalPayment;
                // Subtract principal owed for while loop execution check
                owedPrincipal -= principalPayment;
            }

            __bid.loanDetails.totalRepaid.interest += interest;

            // Set last repaid time
            __bid.loanDetails.lastRepaidTimestamp = uint32(nowTimestamp);
        }
        assertEq(
            cycleIndex,
            expectedTotalCycles,
            "Expected number of cycles incorrect"
        );
        assertEq(
            cycleIndex <= cycleCount + 1,
            true,
            "Payment cycle exceeded agreed terms"
        );
    }

    function test_calculateAmountOwed() public {
        uint256 principal = 24486571879936808846;
        uint256 repaidPrincipal = 23410087846643631232;
        uint16 interestRate = 3000;
        __bid.loanDetails.principal = principal;
        __bid.terms.APR = interestRate;
        __bid.loanDetails.totalRepaid.principal = repaidPrincipal;
        __bid.terms.paymentCycleAmount = 8567977538702439153;
        __bid.terms.paymentCycle = 2592000;
        __bid.loanDetails.acceptedTimestamp = 1646159355;
        __bid.paymentType = PaymentType.EMI;

        (uint256 _owedPrincipal, uint256 _duePrincipal, uint256 _interest) = V2Calculations
            .calculateAmountOwed(
                __bid,
                1658159355, // last repaid timestamp
                1663189241, //timestamp
                PaymentCycleType.Seconds,
                2592000
            );

        assertEq(
            _owedPrincipal,
            1076484033293177614,
            "Expected number of cycles incorrect"
        );
        assertEq(
            _duePrincipal,
            1076484033293177614,
            "Expected number of cycles incorrect"
        );
    }

    function test_calculateAmountOwed_irregular_time_end_of_second_to_last_cycle()
        public
    {
        uint256 principal = 10000;
        uint256 repaidPrincipal = 0;
        uint16 interestRate = 0;
        __bid.loanDetails.principal = principal;
        __bid.loanDetails.loanDuration = 8000;
        __bid.terms.APR = interestRate;
        __bid.loanDetails.totalRepaid.principal = repaidPrincipal;
        __bid.terms.paymentCycleAmount = 3000;
        __bid.terms.paymentCycle = 3000;
        __bid.loanDetails.acceptedTimestamp = 2000000;
        __bid.paymentType = PaymentType.EMI;

        (uint256 _owedPrincipal, uint256 _duePrincipal, uint256 _interest) = V2Calculations
            .calculateAmountOwed(
                __bid,
                2000000 + 3000, //last repaid timestamp
                2000000 + 5500, //timestamp
                PaymentCycleType.Seconds,
                3000
            );

        assertEq(_owedPrincipal, 10000, "Expected owed principal incorrect");
        assertEq(_duePrincipal, 2500, "Expected due principal incorrect");
    }

    function test_calculateAmountOwed_irregular_time_last_cycle() public {
        uint256 principal = 10000;
        uint256 repaidPrincipal = 0;
        uint16 interestRate = 0;
        __bid.loanDetails.principal = principal;
        __bid.loanDetails.loanDuration = 8000;
        __bid.terms.APR = interestRate;
        __bid.loanDetails.totalRepaid.principal = repaidPrincipal;
        __bid.terms.paymentCycleAmount = 3000;
        __bid.terms.paymentCycle = 3000;
        __bid.loanDetails.acceptedTimestamp = 2000000;
        __bid.paymentType = PaymentType.EMI;

        (uint256 _owedPrincipal, uint256 _duePrincipal, uint256 _interest) = V2Calculations
            .calculateAmountOwed(
                __bid,
                2000000 + 3000, //last repaid timestamp
                2000000 + 7500, //timestamp
                PaymentCycleType.Seconds,
                3000
            );

        assertEq(_owedPrincipal, 10000, "Expected owed principal incorrect");
        assertEq(_duePrincipal, 10000, "Expected due principal incorrect");
    }

    function test_calculateAmountOwed_irregular_time_late() public {
        uint256 principal = 10000;
        uint256 repaidPrincipal = 0;
        uint16 interestRate = 0;
        __bid.loanDetails.principal = principal;
        __bid.loanDetails.loanDuration = 8000;
        __bid.terms.APR = interestRate;
        __bid.loanDetails.totalRepaid.principal = repaidPrincipal;
        __bid.terms.paymentCycleAmount = 3000;
        __bid.terms.paymentCycle = 3000;
        __bid.loanDetails.acceptedTimestamp = 2000000;
        __bid.paymentType = PaymentType.EMI;

        (uint256 _owedPrincipal, uint256 _duePrincipal, uint256 _interest) = V2Calculations
            .calculateAmountOwed(
                __bid,
                2000000 + 3000, //last repaid timestamp
                2000000 + 19500, //timestamp
                PaymentCycleType.Seconds,
                3000
            );

        assertEq(_owedPrincipal, 10000, "Expected owed principal incorrect");
        assertEq(_duePrincipal, 10000, "Expected due principal incorrect");
    }

    function test_calculateBulletAmountOwed() public {
        uint256 _principal = 100000e6;
        uint256 _repaidPrincipal = 0;
        uint16 _apr = 3000;
        uint256 _acceptedTimestamp = 1646159355;
        uint256 _lastRepaidTimestamp = _acceptedTimestamp;
        __bid.loanDetails.principal = _principal;
        __bid.terms.APR = _apr;
        __bid.loanDetails.totalRepaid.principal = _repaidPrincipal;
        __bid.terms.paymentCycleAmount = 8567977538702439153;
        __bid.terms.paymentCycle = 2592000;
        __bid.loanDetails.acceptedTimestamp = uint32(_acceptedTimestamp);
        __bid.paymentType = PaymentType.Bullet;
        uint256 _paymentCycleAmount = V2Calculations
            .calculatePaymentCycleAmount(
                PaymentType.Bullet,
                PaymentCycleType.Seconds,
                _principal,
                365 days,
                365 days / 12,
                _apr
            );
        __bid.terms.paymentCycleAmount = _paymentCycleAmount;

        // Within the first payment cycle
        uint256 _timestamp = _acceptedTimestamp + ((365 days / 12) / 2);

        (
            uint256 _owedPrincipal,
            uint256 _duePrincipal,
            uint256 _interest
        ) = V2Calculations.calculateAmountOwed(
                __bid,
                _lastRepaidTimestamp,
                _timestamp,
                PaymentCycleType.Seconds,
                2592000
            );

        assertEq(
            _owedPrincipal,
            _principal,
            "First cycle bullet owed principal incorrect"
        );
        assertEq(
            _duePrincipal,
            0,
            "First cycle bullet due principal incorrect"
        );
        assertEq(
            _interest,
            1250000000,
            "First cycle bullet interest incorrect"
        );

        // Within random payment cycle
        _timestamp = _acceptedTimestamp + ((365 days / 12) * 3);

        __bid.terms.paymentCycle = 365 days / 12;
        __bid.loanDetails.loanDuration = 365 days;

        (_owedPrincipal, _duePrincipal, _interest) = V2Calculations
            .calculateAmountOwed(
                __bid,
                _lastRepaidTimestamp,
                _timestamp,
                PaymentCycleType.Seconds,
                __bid.terms.paymentCycle
            );

        assertEq(
            _owedPrincipal,
            _principal,
            "Second cycle bullet Owed principal incorrect"
        );
        assertEq(_duePrincipal, 0, "Second cycle bullet principal incorrect");
        assertEq(
            _interest,
            7500000000,
            "Second cycle bullet interest incorrect"
        );

        // Last payment cycle
        _timestamp = _acceptedTimestamp + 360 days;

        (_owedPrincipal, _duePrincipal, _interest) = V2Calculations
            .calculateAmountOwed(
                __bid,
                _lastRepaidTimestamp,
                _timestamp,
                PaymentCycleType.Seconds,
                __bid.terms.paymentCycle
            );

        assertEq(
            _owedPrincipal,
            _principal,
            "Final cycle bullet Owed principal incorrect"
        );
        assertEq(
            _duePrincipal,
            _principal,
            "Final cycle bullet principal incorrect"
        );
        assertEq(
            _interest,
            29589041095,
            "Final cycle bullet interest incorrect"
        );

        // Beyond last payment cycle (checks for overflow protection)
        _timestamp = _acceptedTimestamp + 365 days * 2;

        (_owedPrincipal, _duePrincipal, _interest) = V2Calculations
            .calculateAmountOwed(
                __bid,
                _lastRepaidTimestamp,
                _timestamp,
                PaymentCycleType.Seconds,
                __bid.terms.paymentCycle
            );

        assertEq(
            _owedPrincipal,
            _principal,
            "Final cycle bullet Owed principal incorrect"
        );
        assertEq(
            _duePrincipal,
            _principal,
            "Final cycle bullet principal incorrect"
        );
        assertEq(
            _interest,
            ((_principal * _apr) / 10000) * 2,
            "Final cycle bullet interest incorrect"
        );
    }

    function test_calculateEMIAmountOwed_last_cycle() public {
        uint256 _principal = 100000e6;
        uint256 _repaidPrincipal = 91666666667;
        uint16 _apr = 3000;
        uint256 _acceptedTimestamp = 1646159355;
        uint256 _lastRepaidTimestamp = _acceptedTimestamp +
            (365 days - 30 days);
        __bid.loanDetails.loanDuration = 365 days * 1; //1 year
        __bid.loanDetails.principal = _principal;
        __bid.terms.APR = _apr;
        __bid.loanDetails.totalRepaid.principal = _repaidPrincipal;
        __bid.terms.paymentCycleAmount = 8333333333;
        __bid.terms.paymentCycle = 365 days / 12; // 1 month
        __bid.loanDetails.acceptedTimestamp = uint32(_acceptedTimestamp);
        __bid.paymentType = PaymentType.EMI;
        uint256 _paymentCycleAmount = V2Calculations
            .calculatePaymentCycleAmount(
                PaymentType.EMI,
                PaymentCycleType.Seconds,
                _principal,
                365 days,
                365 days / 12,
                _apr
            );
        __bid.terms.paymentCycleAmount = _paymentCycleAmount;

        // Within the first payment cycle
        uint256 _timestamp = _acceptedTimestamp + ((365 days - 2 days)); //we are in the last cycle

        (
            uint256 _owedPrincipal,
            uint256 _duePrincipal,
            uint256 _interest
        ) = V2Calculations.calculateAmountOwed(
                __bid,
                _lastRepaidTimestamp,
                _timestamp,
                PaymentCycleType.Seconds,
                __bid.terms.paymentCycle
            );

        assertEq(
            _owedPrincipal,
            _principal - _repaidPrincipal,
            "Last cycle EMI owed principal incorrect"
        );
        assertEq(
            _duePrincipal,
            _principal - _repaidPrincipal,
            "Last cycle EMI due principal incorrect"
        );

        /*  assertEq(
            _interest,
            191780821,
            "Last cycle EMI interest incorrect"
        );*/
    }
}
