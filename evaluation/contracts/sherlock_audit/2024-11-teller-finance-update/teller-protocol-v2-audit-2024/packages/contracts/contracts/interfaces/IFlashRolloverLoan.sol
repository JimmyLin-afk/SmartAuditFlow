// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IFlashRolloverLoan {
    struct RolloverCallbackArgs { 
        uint256 loanId;
        address borrower;
        uint256 borrowerAmount;
        bytes acceptCommitmentArgs;
    }
}
