// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

enum CommitmentCollateralType {
    NONE, // no collateral required
    ERC20,
    ERC721,
    ERC1155,
    ERC721_ANY_ID,
    ERC1155_ANY_ID,
    ERC721_MERKLE_PROOF,
    ERC1155_MERKLE_PROOF
}

interface ISmartCommitment {
    function getPrincipalTokenAddress() external view returns (address);

    function getMarketId() external view returns (uint256);

    function getCollateralTokenAddress() external view returns (address);

    function getCollateralTokenType()
        external
        view
        returns (CommitmentCollateralType);

    function getCollateralTokenId() external view returns (uint256);

    function getMinInterestRate(uint256 _delta) external view returns (uint16);

    function getMaxLoanDuration() external view returns (uint32);

    function getPrincipalAmountAvailableToBorrow()
        external
        view
        returns (uint256);

    function getRequiredCollateral(
        uint256 _principalAmount,
        uint256 _maxPrincipalPerCollateralAmount
        
        )
        external
        view
        returns (uint256);

    
    function acceptFundsForAcceptBid(
        address _borrower,
        uint256 _bidId,
        uint256 _principalAmount,
        uint256 _collateralAmount,
        address _collateralTokenAddress,
        uint256 _collateralTokenId,
        uint32 _loanDuration,
        uint16 _interestRate
    ) external;
}
