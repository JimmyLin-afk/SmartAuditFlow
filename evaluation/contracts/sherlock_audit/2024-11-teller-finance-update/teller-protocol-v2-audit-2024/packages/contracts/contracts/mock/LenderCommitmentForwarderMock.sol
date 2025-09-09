// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/utils/Address.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
//import "../TellerV2MarketForwarder.sol";

import "../TellerV2Context.sol";

//import { LenderCommitmentForwarder } from "../contracts/LenderCommitmentForwarder.sol";

import "../interfaces/ITellerV2.sol";
import "../interfaces/ILenderCommitmentForwarder.sol";

import "../interfaces/ITellerV2MarketForwarder.sol";

import { Collateral, CollateralType } from "../interfaces/escrow/ICollateralEscrowV1.sol";

import "../mock/MarketRegistryMock.sol";

contract LenderCommitmentForwarderMock is
    ILenderCommitmentForwarder,
    ITellerV2MarketForwarder
{
    mapping(uint256 => Commitment) public commitments;

    uint256 commitmentCount;

    bool public submitBidWithCollateralWasCalled;
    bool public acceptBidWasCalled;
    bool public submitBidWasCalled;
    bool public acceptCommitmentWithRecipientWasCalled;
    bool public acceptCommitmentWithRecipientAndProofWasCalled;

    mapping(uint256 => uint256) public commitmentPrincipalAccepted;

    constructor() {}

    function createCommitment(
        Commitment calldata _commitment,
        address[] calldata _borrowerAddressList
    ) external returns (uint256) {}

    function setCommitment(uint256 _commitmentId, Commitment memory _commitment)
        public
    {
        commitments[_commitmentId] = _commitment;
    }

    function getCommitmentLender(uint256 _commitmentId)
        public
        view
        returns (address)
    {
        return commitments[_commitmentId].lender;
    }

    function getCommitmentMarketId(uint256 _commitmentId)
        public
        view
        returns (uint256)
    {
        return commitments[_commitmentId].marketId;
    }

    function getCommitmentAcceptedPrincipal(uint256 _commitmentId)
        public
        view
        returns (uint256)
    {
        return commitmentPrincipalAccepted[_commitmentId];
    }

    function getCommitmentMaxPrincipal(uint256 _commitmentId)
        public
        view
        returns (uint256)
    {
        return commitments[_commitmentId].maxPrincipal;
    }

    /* function _getEscrowCollateralTypeSuper(CommitmentCollateralType _type)
        public
        returns (CollateralType)
    {
        return super._getEscrowCollateralType(_type);
    }

    function validateCommitmentSuper(uint256 _commitmentId) public {
        super.validateCommitment(commitments[_commitmentId]);
    }*/

    function acceptCommitmentWithRecipient(
        uint256 _commitmentId,
        uint256 _principalAmount,
        uint256 _collateralAmount,
        uint256 _collateralTokenId,
        address _collateralTokenAddress,
        address _recipient,
        uint16 _interestRate,
        uint32 _loanDuration
    ) public returns (uint256 bidId) {
        acceptCommitmentWithRecipientWasCalled = true;

        Commitment storage commitment = commitments[_commitmentId];

        address lendingToken = commitment.principalTokenAddress;

        _mockAcceptCommitmentTokenTransfer(
            lendingToken,
            _principalAmount,
            _recipient
        );
    }

    function acceptCommitmentWithRecipientAndProof(
        uint256 _commitmentId,
        uint256 _principalAmount,
        uint256 _collateralAmount,
        uint256 _collateralTokenId,
        address _collateralTokenAddress,
        address _recipient,
        uint16 _interestRate,
        uint32 _loanDuration,
        bytes32[] calldata _merkleProof
    ) public returns (uint256 bidId) {
        acceptCommitmentWithRecipientAndProofWasCalled = true;

        Commitment storage commitment = commitments[_commitmentId];

        address lendingToken = commitment.principalTokenAddress;

        _mockAcceptCommitmentTokenTransfer(
            lendingToken,
            _principalAmount,
            _recipient
        );
    }

    function _mockAcceptCommitmentTokenTransfer(
        address lendingToken,
        uint256 principalAmount,
        address _recipient
    ) internal {
        IERC20(lendingToken).transfer(_recipient, principalAmount);
    }

    /*
        Override methods 
    */

    /* function _submitBid(CreateLoanArgs memory, address)
        internal
        override
        returns (uint256 bidId)
    {
        submitBidWasCalled = true;
        return 1;
    }

    function _submitBidWithCollateral(CreateLoanArgs memory, address)
        internal
        override
        returns (uint256 bidId)
    {
        submitBidWithCollateralWasCalled = true;
        return 1;
    }

    function _acceptBid(uint256, address) internal override returns (bool) {
        acceptBidWasCalled = true;

        return true;
    }
    */
}
