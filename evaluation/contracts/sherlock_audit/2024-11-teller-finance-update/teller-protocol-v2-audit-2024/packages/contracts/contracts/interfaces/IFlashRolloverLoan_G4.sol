// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IFlashRolloverLoan_G4 {
    struct RolloverCallbackArgs {
        address lenderCommitmentForwarder;
        uint256 loanId;
        address borrower;
        uint256 borrowerAmount;
        bytes acceptCommitmentArgs;
    }
}
