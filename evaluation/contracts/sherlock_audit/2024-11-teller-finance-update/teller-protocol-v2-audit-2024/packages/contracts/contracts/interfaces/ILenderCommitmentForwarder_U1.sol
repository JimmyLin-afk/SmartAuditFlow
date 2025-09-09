// SPDX-Licence-Identifier: MIT
pragma solidity >=0.8.0 <0.9.0;

interface ILenderCommitmentForwarder_U1 {
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

    /**
     * @notice Details about a lender's capital commitment.
     * @param maxPrincipal Amount of tokens being committed by the lender. Max amount that can be loaned.
     * @param expiration Expiration time in seconds, when the commitment expires.
     * @param maxDuration Length of time, in seconds that the lender's capital can be lent out for.
     * @param minInterestRate Minimum Annual percentage to be applied for loans using the lender's capital.
     * @param collateralTokenAddress The address for the token contract that must be used to provide collateral for loans for this commitment.
     * @param maxPrincipalPerCollateralAmount The amount of principal that can be used for a loan per each unit of collateral, expanded additionally by principal decimals.
     * @param collateralTokenType The type of asset of the collateralTokenAddress (ERC20, ERC721, or ERC1155).
     * @param lender The address of the lender for this commitment.
     * @param marketId The market id for this commitment.
     * @param principalTokenAddress The address for the token contract that will be used to provide principal for loans of this commitment.
     */
    struct Commitment {
        uint256 maxPrincipal;
        uint32 expiration;
        uint32 maxDuration;
        uint16 minInterestRate;
        address collateralTokenAddress;
        uint256 collateralTokenId; //we use this for the MerkleRootHash  for type ERC721_MERKLE_PROOF
        uint256 maxPrincipalPerCollateralAmount;
        CommitmentCollateralType collateralTokenType;
        address lender;
        uint256 marketId;
        address principalTokenAddress;
    }

    struct PoolRouteConfig {
        address pool;
        bool zeroForOne;
        uint32 twapInterval;
        uint256 token0Decimals;
        uint256 token1Decimals;
    }

    // mapping(uint256 => Commitment) public commitments;

    function getCommitmentMarketId(uint256 _commitmentId)
        external
        view
        returns (uint256);

    function getCommitmentLender(uint256 _commitmentId)
        external
        view
        returns (address);

    function getCommitmentAcceptedPrincipal(uint256 _commitmentId)
        external
        view
        returns (uint256);

    function getCommitmentMaxPrincipal(uint256 _commitmentId)
        external
        view
        returns (uint256);

    function createCommitmentWithUniswap(
        Commitment calldata _commitment,
        address[] calldata _borrowerAddressList,
        PoolRouteConfig[] calldata _poolRoutes,
        uint16 _poolOracleLtvRatio
    ) external returns (uint256);

    function acceptCommitmentWithRecipient(
        uint256 _commitmentId,
        uint256 _principalAmount,
        uint256 _collateralAmount,
        uint256 _collateralTokenId,
        address _collateralTokenAddress,
        address _recipient,
        uint16 _interestRate,
        uint32 _loanDuration
    ) external returns (uint256 bidId_);

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
    ) external returns (uint256 bidId_);


    function getCommitmentPrincipalTokenAddress(uint256 _commitmentId)
        external
        view
        returns (address);

    function getCommitmentCollateralTokenAddress(uint256 _commitmentId)
        external
        view
        returns (address);


}
