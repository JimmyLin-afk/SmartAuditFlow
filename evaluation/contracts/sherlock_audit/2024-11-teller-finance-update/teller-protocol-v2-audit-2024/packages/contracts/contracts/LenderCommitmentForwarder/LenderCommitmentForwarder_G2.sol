pragma solidity >=0.8.0 <0.9.0;
// SPDX-License-Identifier: MIT

// Contracts
import "../TellerV2MarketForwarder_G2.sol";

// Interfaces
import "../interfaces/ICollateralManager.sol";
import "../interfaces/ILenderCommitmentForwarder.sol";
import { Collateral, CollateralType } from "../interfaces/escrow/ICollateralEscrowV1.sol";

import "@openzeppelin/contracts-upgradeable/utils/structs/EnumerableSetUpgradeable.sol";

// Libraries
import { MathUpgradeable } from "@openzeppelin/contracts-upgradeable/utils/math/MathUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/token/ERC20/extensions/IERC20MetadataUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/cryptography/MerkleProofUpgradeable.sol";

contract LenderCommitmentForwarder_G2 is
    TellerV2MarketForwarder_G2,
    ILenderCommitmentForwarder
{
    using EnumerableSetUpgradeable for EnumerableSetUpgradeable.AddressSet;

    // CommitmentId => commitment
    mapping(uint256 => Commitment) public commitments;

    uint256 commitmentCount;

    //https://github.com/OpenZeppelin/openzeppelin-contracts-upgradeable/blob/master/contracts/utils/structs/EnumerableSetUpgradeable.sol
    mapping(uint256 => EnumerableSetUpgradeable.AddressSet)
        internal commitmentBorrowersList;

    mapping(uint256 => uint256) public commitmentPrincipalAccepted;

    /**
     * @notice This event is emitted when a lender's commitment is created.
     * @param lender The address of the lender.
     * @param marketId The Id of the market the commitment applies to.
     * @param lendingToken The address of the asset being committed.
     * @param tokenAmount The amount of the asset being committed.
     */
    event CreatedCommitment(
        uint256 indexed commitmentId,
        address lender,
        uint256 marketId,
        address lendingToken,
        uint256 tokenAmount
    );

    /**
     * @notice This event is emitted when a lender's commitment is updated.
     * @param commitmentId The id of the commitment that was updated.
     * @param lender The address of the lender.
     * @param marketId The Id of the market the commitment applies to.
     * @param lendingToken The address of the asset being committed.
     * @param tokenAmount The amount of the asset being committed.
     */
    event UpdatedCommitment(
        uint256 indexed commitmentId,
        address lender,
        uint256 marketId,
        address lendingToken,
        uint256 tokenAmount
    );

    /**
     * @notice This event is emitted when the allowed borrowers for a commitment is updated.
     * @param commitmentId The id of the commitment that was updated.
     */
    event UpdatedCommitmentBorrowers(uint256 indexed commitmentId);

    /**
     * @notice This event is emitted when a lender's commitment has been deleted.
     * @param commitmentId The id of the commitment that was deleted.
     */
    event DeletedCommitment(uint256 indexed commitmentId);

    /**
     * @notice This event is emitted when a lender's commitment is exercised for a loan.
     * @param commitmentId The id of the commitment that was exercised.
     * @param borrower The address of the borrower.
     * @param tokenAmount The amount of the asset being committed.
     * @param bidId The bid id for the loan from TellerV2.
     */
    event ExercisedCommitment(
        uint256 indexed commitmentId,
        address borrower,
        uint256 tokenAmount,
        uint256 bidId
    );

    error InsufficientCommitmentAllocation(
        uint256 allocated,
        uint256 requested
    );
    error InsufficientBorrowerCollateral(uint256 required, uint256 actual);

    /** Modifiers **/

    modifier commitmentLender(uint256 _commitmentId) {
        require(
            commitments[_commitmentId].lender == _msgSender(),
            "unauthorized commitment lender"
        );
        _;
    }

    function validateCommitment(Commitment storage _commitment) internal {
        require(
            _commitment.expiration > uint32(block.timestamp),
            "expired commitment"
        );
        require(
            _commitment.maxPrincipal > 0,
            "commitment principal allocation 0"
        );

        if (_commitment.collateralTokenType != CommitmentCollateralType.NONE) {
            require(
                _commitment.maxPrincipalPerCollateralAmount > 0,
                "commitment collateral ratio 0"
            );

            if (
                _commitment.collateralTokenType ==
                CommitmentCollateralType.ERC20
            ) {
                require(
                    _commitment.collateralTokenId == 0,
                    "commitment collateral token id must be 0 for ERC20"
                );
            }
        }
    }

    /** External Functions **/

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor(address _protocolAddress, address _marketRegistry)
        TellerV2MarketForwarder_G2(_protocolAddress, _marketRegistry)
    {}

    /**
     * @notice Creates a loan commitment from a lender for a market.
     * @param _commitment The new commitment data expressed as a struct
     * @param _borrowerAddressList The array of borrowers that are allowed to accept loans using this commitment
     * @return commitmentId_ returns the commitmentId for the created commitment
     */
    function createCommitment(
        Commitment calldata _commitment,
        address[] calldata _borrowerAddressList
    ) public returns (uint256 commitmentId_) {
        commitmentId_ = commitmentCount++;

        require(
            _commitment.lender == _msgSender(),
            "unauthorized commitment creator"
        );

        commitments[commitmentId_] = _commitment;

        //make sure the commitment data adheres to required specifications and limits
        validateCommitment(commitments[commitmentId_]);

        //the borrower allowlists is in a different storage space so we append them to the array with this method s
        _addBorrowersToCommitmentAllowlist(commitmentId_, _borrowerAddressList);

        emit CreatedCommitment(
            commitmentId_,
            _commitment.lender,
            _commitment.marketId,
            _commitment.principalTokenAddress,
            _commitment.maxPrincipal
        );
    }

    /**
     * @notice Updates the commitment of a lender to a market.
     * @param _commitmentId The Id of the commitment to update.
     * @param _commitment The new commitment data expressed as a struct
     */
    function updateCommitment(
        uint256 _commitmentId,
        Commitment calldata _commitment
    ) public commitmentLender(_commitmentId) {
        require(
            _commitment.lender == _msgSender(),
            "Commitment lender cannot be updated."
        );

        require(
            _commitment.principalTokenAddress ==
                commitments[_commitmentId].principalTokenAddress,
            "Principal token address cannot be updated."
        );
        require(
            _commitment.marketId == commitments[_commitmentId].marketId,
            "Market Id cannot be updated."
        );

        commitments[_commitmentId] = _commitment;

        //make sure the commitment data still adheres to required specifications and limits
        validateCommitment(commitments[_commitmentId]);

        emit UpdatedCommitment(
            _commitmentId,
            _commitment.lender,
            _commitment.marketId,
            _commitment.principalTokenAddress,
            _commitment.maxPrincipal
        );
    }

    /**
     * @notice Updates the borrowers allowed to accept a commitment
     * @param _commitmentId The Id of the commitment to update.
     * @param _borrowerAddressList The array of borrowers that are allowed to accept loans using this commitment
     */
    function addCommitmentBorrowers(
        uint256 _commitmentId,
        address[] calldata _borrowerAddressList
    ) public commitmentLender(_commitmentId) {
        _addBorrowersToCommitmentAllowlist(_commitmentId, _borrowerAddressList);
    }

    /**
     * @notice Updates the borrowers allowed to accept a commitment
     * @param _commitmentId The Id of the commitment to update.
     * @param _borrowerAddressList The array of borrowers that are allowed to accept loans using this commitment
     */
    function removeCommitmentBorrowers(
        uint256 _commitmentId,
        address[] calldata _borrowerAddressList
    ) public commitmentLender(_commitmentId) {
        _removeBorrowersFromCommitmentAllowlist(
            _commitmentId,
            _borrowerAddressList
        );
    }

    /**
     * @notice Adds a borrower to the allowlist for a commmitment.
     * @param _commitmentId The id of the commitment that will allow the new borrower
     * @param _borrowerArray the address array of the borrowers that will be allowed to accept loans using the commitment
     */
    function _addBorrowersToCommitmentAllowlist(
        uint256 _commitmentId,
        address[] calldata _borrowerArray
    ) internal {
        for (uint256 i = 0; i < _borrowerArray.length; i++) {
            commitmentBorrowersList[_commitmentId].add(_borrowerArray[i]);
        }
        emit UpdatedCommitmentBorrowers(_commitmentId);
    }

    /**
     * @notice Removes a borrower to the allowlist for a commmitment.
     * @param _commitmentId The id of the commitment that will allow the new borrower
     * @param _borrowerArray the address array of the borrowers that will be allowed to accept loans using the commitment
     */
    function _removeBorrowersFromCommitmentAllowlist(
        uint256 _commitmentId,
        address[] calldata _borrowerArray
    ) internal {
        for (uint256 i = 0; i < _borrowerArray.length; i++) {
            commitmentBorrowersList[_commitmentId].remove(_borrowerArray[i]);
        }
        emit UpdatedCommitmentBorrowers(_commitmentId);
    }

    /**
     * @notice Removes the commitment of a lender to a market.
     * @param _commitmentId The id of the commitment to delete.
     */
    function deleteCommitment(uint256 _commitmentId)
        public
        commitmentLender(_commitmentId)
    {
        delete commitments[_commitmentId];
        delete commitmentBorrowersList[_commitmentId];
        emit DeletedCommitment(_commitmentId);
    }

    /**
     * @notice Accept the commitment to submitBid and acceptBid using the funds
     * @dev LoanDuration must be longer than the market payment cycle
     * @param _commitmentId The id of the commitment being accepted.
     * @param _principalAmount The amount of currency to borrow for the loan.
     * @param _collateralAmount The amount of collateral to use for the loan.
     * @param _collateralTokenId The tokenId of collateral to use for the loan if ERC721 or ERC1155.
     * @param _collateralTokenAddress The contract address to use for the loan collateral tokens.
     * @param _recipient The address to receive the loan funds.
     * @param _interestRate The interest rate APY to use for the loan in basis points.
     * @param _loanDuration The overall duration for the loan.  Must be longer than market payment cycle duration.
     * @return bidId The ID of the loan that was created on TellerV2
     */
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
        require(
            commitments[_commitmentId].collateralTokenType <=
                CommitmentCollateralType.ERC1155_ANY_ID,
            "Invalid commitment collateral type"
        );

        return
            _acceptCommitment(
                _commitmentId,
                _principalAmount,
                _collateralAmount,
                _collateralTokenId,
                _collateralTokenAddress,
                _recipient,
                _interestRate,
                _loanDuration
            );
    }

    function acceptCommitment(
        uint256 _commitmentId,
        uint256 _principalAmount,
        uint256 _collateralAmount,
        uint256 _collateralTokenId,
        address _collateralTokenAddress,
        uint16 _interestRate,
        uint32 _loanDuration
    ) public returns (uint256 bidId) {
        return
            acceptCommitmentWithRecipient(
                _commitmentId,
                _principalAmount,
                _collateralAmount,
                _collateralTokenId,
                _collateralTokenAddress,
                address(0),
                _interestRate,
                _loanDuration
            );
    }

    /**
     * @notice Accept the commitment to submitBid and acceptBid using the funds
     * @dev LoanDuration must be longer than the market payment cycle
     * @param _commitmentId The id of the commitment being accepted.
     * @param _principalAmount The amount of currency to borrow for the loan.
     * @param _collateralAmount The amount of collateral to use for the loan.
     * @param _collateralTokenId The tokenId of collateral to use for the loan if ERC721 or ERC1155.
     * @param _collateralTokenAddress The contract address to use for the loan collateral tokens.
     * @param _recipient The address to receive the loan funds.
     * @param _interestRate The interest rate APY to use for the loan in basis points.
     * @param _loanDuration The overall duration for the loan.  Must be longer than market payment cycle duration.
     * @param _merkleProof An array of bytes32 which are the roots down the merkle tree, the merkle proof.
     * @return bidId The ID of the loan that was created on TellerV2
     */
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
        require(
            commitments[_commitmentId].collateralTokenType ==
                CommitmentCollateralType.ERC721_MERKLE_PROOF ||
                commitments[_commitmentId].collateralTokenType ==
                CommitmentCollateralType.ERC1155_MERKLE_PROOF,
            "Invalid commitment collateral type"
        );

        bytes32 _merkleRoot = bytes32(
            commitments[_commitmentId].collateralTokenId
        );
        bytes32 _leaf = keccak256(abi.encodePacked(_collateralTokenId));

        //make sure collateral token id is a leaf within the proof
        require(
            MerkleProofUpgradeable.verifyCalldata(
                _merkleProof,
                _merkleRoot,
                _leaf
            ),
            "Invalid proof"
        );

        return
            _acceptCommitment(
                _commitmentId,
                _principalAmount,
                _collateralAmount,
                _collateralTokenId,
                _collateralTokenAddress,
                _recipient,
                _interestRate,
                _loanDuration
            );
    }

    function acceptCommitmentWithProof(
        uint256 _commitmentId,
        uint256 _principalAmount,
        uint256 _collateralAmount,
        uint256 _collateralTokenId,
        address _collateralTokenAddress,
        uint16 _interestRate,
        uint32 _loanDuration,
        bytes32[] calldata _merkleProof
    ) public returns (uint256 bidId) {
        return
            acceptCommitmentWithRecipientAndProof(
                _commitmentId,
                _principalAmount,
                _collateralAmount,
                _collateralTokenId,
                _collateralTokenAddress,
                address(0),
                _interestRate,
                _loanDuration,
                _merkleProof
            );
    }

    /**
     * @notice Accept the commitment to submitBid and acceptBid using the funds
     * @dev LoanDuration must be longer than the market payment cycle
     * @param _commitmentId The id of the commitment being accepted.
     * @param _principalAmount The amount of currency to borrow for the loan.
     * @param _collateralAmount The amount of collateral to use for the loan.
     * @param _collateralTokenId The tokenId of collateral to use for the loan if ERC721 or ERC1155.
     * @param _collateralTokenAddress The contract address to use for the loan collateral tokens.
     * @param _recipient The address to receive the loan funds.
     * @param _interestRate The interest rate APY to use for the loan in basis points.
     * @param _loanDuration The overall duration for the loan.  Must be longer than market payment cycle duration.
     * @return bidId The ID of the loan that was created on TellerV2
     */
    function _acceptCommitment(
        uint256 _commitmentId,
        uint256 _principalAmount,
        uint256 _collateralAmount,
        uint256 _collateralTokenId,
        address _collateralTokenAddress,
        address _recipient,
        uint16 _interestRate,
        uint32 _loanDuration
    ) internal returns (uint256 bidId) {
        Commitment storage commitment = commitments[_commitmentId];

        //make sure the commitment data adheres to required specifications and limits
        validateCommitment(commitment);

        //the collateral token of the commitment should be the same as the acceptor expects
        require(
            _collateralTokenAddress == commitment.collateralTokenAddress,
            "Mismatching collateral token"
        );
        //the interest rate must be at least as high has the commitment demands. The borrower can use a higher interest rate although that would not be beneficial to the borrower.
        require(
            _interestRate >= commitment.minInterestRate,
            "Invalid interest rate"
        );
        //the loan duration must be less than the commitment max loan duration. The lender who made the commitment expects the money to be returned before this window.
        require(
            _loanDuration <= commitment.maxDuration,
            "Invalid loan max duration"
        );

        require(
            commitmentPrincipalAccepted[_commitmentId] <= commitment.maxPrincipal,
            "Invalid loan max principal"
        );

        require(
            commitmentBorrowersList[_commitmentId].length() == 0 ||
                commitmentBorrowersList[_commitmentId].contains(_msgSender()),
            "unauthorized commitment borrower"
        );
        //require that the borrower accepting the commitment cannot borrow more than the commitments max principal
        if (_principalAmount > commitment.maxPrincipal) {
            revert InsufficientCommitmentAllocation({
                allocated: commitment.maxPrincipal,
                requested: _principalAmount
            });
        }

        uint256 requiredCollateral = getRequiredCollateral(
            _principalAmount,
            commitment.maxPrincipalPerCollateralAmount,
            commitment.collateralTokenType,
            commitment.collateralTokenAddress,
            commitment.principalTokenAddress
        );

        if (_collateralAmount < requiredCollateral) {
            revert InsufficientBorrowerCollateral({
                required: requiredCollateral,
                actual: _collateralAmount
            });
        }

        //ERC721 assets must have a quantity of 1
        if (
            commitment.collateralTokenType == CommitmentCollateralType.ERC721 ||
            commitment.collateralTokenType ==
            CommitmentCollateralType.ERC721_ANY_ID ||
            commitment.collateralTokenType ==
            CommitmentCollateralType.ERC721_MERKLE_PROOF
        ) {
            require(
                _collateralAmount == 1,
                "invalid commitment collateral amount for ERC721"
            );
        }

        //ERC721 and ERC1155 types strictly enforce a specific token Id.  ERC721_ANY and ERC1155_ANY do not.
        if (
            commitment.collateralTokenType == CommitmentCollateralType.ERC721 ||
            commitment.collateralTokenType == CommitmentCollateralType.ERC1155
        ) {
            require(
                commitment.collateralTokenId == _collateralTokenId,
                "invalid commitment collateral tokenId"
            );
        }

        commitmentPrincipalAccepted[_commitmentId] += _principalAmount;

        require(
            commitmentPrincipalAccepted[_commitmentId] <=
                commitment.maxPrincipal,
            "Exceeds max principal of commitment"
        );

        CreateLoanArgs memory createLoanArgs;
        createLoanArgs.marketId = commitment.marketId;
        createLoanArgs.lendingToken = commitment.principalTokenAddress;
        createLoanArgs.principal = _principalAmount;
        createLoanArgs.duration = _loanDuration;
        createLoanArgs.interestRate = _interestRate;
        createLoanArgs.recipient = _recipient;
        if (commitment.collateralTokenType != CommitmentCollateralType.NONE) {
            createLoanArgs.collateral = new Collateral[](1);
            createLoanArgs.collateral[0] = Collateral({
                _collateralType: _getEscrowCollateralType(
                    commitment.collateralTokenType
                ),
                _tokenId: _collateralTokenId,
                _amount: _collateralAmount,
                _collateralAddress: commitment.collateralTokenAddress
            });
        }

        bidId = _submitBidWithCollateral(createLoanArgs, _msgSender());

        _acceptBid(bidId, commitment.lender);

        emit ExercisedCommitment(
            _commitmentId,
            _msgSender(),
            _principalAmount,
            bidId
        );
    }

    /**
     * @notice Calculate the amount of collateral required to borrow a loan with _principalAmount of principal
     * @param _principalAmount The amount of currency to borrow for the loan.
     * @param _maxPrincipalPerCollateralAmount The ratio for the amount of principal that can be borrowed for each amount of collateral. This is expanded additionally by the principal decimals.
     * @param _collateralTokenType The type of collateral for the loan either ERC20, ERC721, ERC1155, or None.
     * @param _collateralTokenAddress The contract address for the collateral for the loan.
     * @param _principalTokenAddress The contract address for the principal for the loan.
     */
    function getRequiredCollateral(
        uint256 _principalAmount,
        uint256 _maxPrincipalPerCollateralAmount,
        CommitmentCollateralType _collateralTokenType,
        address _collateralTokenAddress,
        address _principalTokenAddress
    ) public view virtual returns (uint256) {
        if (_collateralTokenType == CommitmentCollateralType.NONE) {
            return 0;
        }

        uint8 collateralDecimals;
        uint8 principalDecimals = IERC20MetadataUpgradeable(
            _principalTokenAddress
        ).decimals();

        if (_collateralTokenType == CommitmentCollateralType.ERC20) {
            collateralDecimals = IERC20MetadataUpgradeable(
                _collateralTokenAddress
            ).decimals();
        }

        /*
         * The principalAmount is expanded by (collateralDecimals+principalDecimals) to increase precision
         * and then it is divided by _maxPrincipalPerCollateralAmount which should already been expanded by principalDecimals
         */
        return
            MathUpgradeable.mulDiv(
                _principalAmount,
                (10**(collateralDecimals + principalDecimals)),
                _maxPrincipalPerCollateralAmount,
                MathUpgradeable.Rounding.Up
            );
    }

    /**
     * @notice Return the array of borrowers that are allowlisted for a commitment
     * @param _commitmentId The commitment id for the commitment to query.
     * @return borrowers_ An array of addresses restricted to accept the commitment. Empty array means unrestricted.
     */
    function getCommitmentBorrowers(uint256 _commitmentId)
        external
        view
        returns (address[] memory borrowers_)
    {
        borrowers_ = commitmentBorrowersList[_commitmentId].values();
    }

    /**
     * @notice Return the collateral type based on the commitmentcollateral type.  Collateral type is used in the base lending protocol.
     * @param _type The type of collateral to be used for the loan.
     */
    function _getEscrowCollateralType(CommitmentCollateralType _type)
        internal
        pure
        returns (CollateralType)
    {
        if (_type == CommitmentCollateralType.ERC20) {
            return CollateralType.ERC20;
        }
        if (
            _type == CommitmentCollateralType.ERC721 ||
            _type == CommitmentCollateralType.ERC721_ANY_ID ||
            _type == CommitmentCollateralType.ERC721_MERKLE_PROOF
        ) {
            return CollateralType.ERC721;
        }
        if (
            _type == CommitmentCollateralType.ERC1155 ||
            _type == CommitmentCollateralType.ERC1155_ANY_ID ||
            _type == CommitmentCollateralType.ERC1155_MERKLE_PROOF
        ) {
            return CollateralType.ERC1155;
        }

        revert("Unknown Collateral Type");
    }

    function getCommitmentMarketId(uint256 _commitmentId)
        external
        view
        returns (uint256)
    {
        return commitments[_commitmentId].marketId;
    }

    function getCommitmentLender(uint256 _commitmentId)
        external
        view
        returns (address)
    {
        return commitments[_commitmentId].lender;
    }

    function getCommitmentAcceptedPrincipal(uint256 _commitmentId)
        external
        view
        returns (uint256)
    {
        return commitmentPrincipalAccepted[_commitmentId];
    }

    function getCommitmentMaxPrincipal(uint256 _commitmentId)
        external
        view
        returns (uint256)
    {
        return commitments[_commitmentId].maxPrincipal;
    }
}
