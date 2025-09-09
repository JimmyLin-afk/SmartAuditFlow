// SPDX-Licence-Identifier: MIT
pragma solidity >=0.8.0 <0.9.0;

interface ICommitmentRolloverLoan {
    struct AcceptCommitmentArgs {
        uint256 commitmentId;
        uint256 principalAmount;
        uint256 collateralAmount;
        uint256 collateralTokenId;
        address collateralTokenAddress;
        uint16 interestRate;
        uint32 loanDuration;
    }
}
