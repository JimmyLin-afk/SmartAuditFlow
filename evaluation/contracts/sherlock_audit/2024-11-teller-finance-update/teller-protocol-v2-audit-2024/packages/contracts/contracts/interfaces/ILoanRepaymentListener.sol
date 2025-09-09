// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0 <0.9.0;

interface ILoanRepaymentListener {
    function repayLoanCallback(
        uint256 bidId,
        address repayer,
        uint256 principalAmount,
        uint256 interestAmount
    ) external;
}
