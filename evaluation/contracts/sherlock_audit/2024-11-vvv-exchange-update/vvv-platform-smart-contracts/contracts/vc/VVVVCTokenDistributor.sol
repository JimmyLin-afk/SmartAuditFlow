//SPDX-License-Identifier: MIT
pragma solidity 0.8.23;

import { ECDSA } from "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import { IERC20 } from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import { SafeERC20 } from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import { VVVAuthorizationRegistryChecker } from "contracts/auth/VVVAuthorizationRegistryChecker.sol";

/**
 * @title VVV VC Token Distributor
 * @notice This contract facilitates token distribution for VVV VC projects
 */
contract VVVVCTokenDistributor is VVVAuthorizationRegistryChecker {
    using SafeERC20 for IERC20;

    /// @notice EIP-712 standard definitions
    bytes32 public constant DOMAIN_TYPEHASH =
        keccak256(bytes("EIP712Domain(string name,uint256 chainId,address verifyingContract)"));
    bytes32 public constant CLAIM_TYPEHASH =
        keccak256(
            bytes(
                "ClaimParams(address kycAddress,address projectTokenAddress,address[] projectTokenProxyWallets,uint256[] tokenAmountsToClaim,uint256 nonce,uint256 deadline)"
            )
        );
    bytes32 public immutable DOMAIN_SEPARATOR;

    /// @notice The address authorized to sign investment transactions
    address public signer;

    /// @notice flag to pause claims
    bool public claimIsPaused;

    /// @notice Mapping to store a nonce for each KYC address
    mapping(address => uint256) public nonces;

    /**
        @notice Parameters for claim function
        @param kycAddress Address of the user's KYC wallet
        @param projectTokenAddress Address of the project token to be claimed
        @param projectTokenProxyWallets Array of addresses of the wallets from which the project token is to be claimed
        @param tokenAmountsToClaim Array of amounts of project tokens to claim
        @param nonce KYC-wallet-based nonce for replay protection
        @param deadline Deadline for signature validity
        @param signature Signature of the user's KYC wallet address
     */
    struct ClaimParams {
        address kycAddress;
        address projectTokenAddress;
        address[] projectTokenProxyWallets;
        uint256[] tokenAmountsToClaim;
        uint256 nonce;
        uint256 deadline;
        bytes signature;
    }

    /**
        @notice Emitted when a user claims tokens
        @param kycAddress Address of the user's KYC wallet
        @param projectTokenAddress Address of the project token to be claimed
        @param projectTokenProxyWallets Addresses of the wallets from which the project token is to be claimed
        @param tokenAmountsToClaim Amounts of project tokens claimed from each wallet
        @param nonce KYC-wallet-based nonce
     */
    event VCClaim(
        address indexed kycAddress,
        address indexed projectTokenAddress,
        address[] projectTokenProxyWallets,
        uint256[] tokenAmountsToClaim,
        uint256 nonce
    );

    /// @notice Error thrown when a claim is attempted while claims are paused
    error ClaimIsPaused();

    /// @notice Error thrown when the signer address is not recovered from the provided signature
    error InvalidSignature();

    /// @notice Error thrown when the nonce is not greater than the stored nonce for the KYC address
    error InvalidNonce();

    /// @notice Error thrown when the lengths of the projectTokenProxyWallets and tokenAmountsToClaim arrays do not match
    error ArrayLengthMismatch();

    constructor(
        address _signer,
        string memory _environmentTag,
        address _authorizationRegistryAddress
    ) VVVAuthorizationRegistryChecker(_authorizationRegistryAddress) {
        signer = _signer;

        // EIP-712 domain separator
        DOMAIN_SEPARATOR = keccak256(
            abi.encode(
                DOMAIN_TYPEHASH,
                keccak256(abi.encodePacked("VVV", _environmentTag)),
                block.chainid,
                address(this)
            )
        );
    }

    /**
        @notice Allows any address which is an alias of a KYC address to claim tokens across multiple rounds which provide that token
        @param _params A ClaimParams struct describing the desired claim(s)
     */
    function claim(ClaimParams memory _params) public {
        if (claimIsPaused) {
            revert ClaimIsPaused();
        }

        if (_params.projectTokenProxyWallets.length != _params.tokenAmountsToClaim.length) {
            revert ArrayLengthMismatch();
        }

        if (_params.nonce <= nonces[_params.kycAddress]) {
            revert InvalidNonce();
        }

        if (!_isSignatureValid(_params)) {
            revert InvalidSignature();
        }

        // update nonce
        nonces[_params.kycAddress] = _params.nonce;

        // define token to transfer
        IERC20 projectToken = IERC20(_params.projectTokenAddress);

        // transfer tokens from each wallet to the caller
        for (uint256 i = 0; i < _params.projectTokenProxyWallets.length; i++) {
            projectToken.safeTransferFrom(
                _params.projectTokenProxyWallets[i],
                msg.sender,
                _params.tokenAmountsToClaim[i]
            );
        }

        emit VCClaim(
            _params.kycAddress,
            _params.projectTokenAddress,
            _params.projectTokenProxyWallets,
            _params.tokenAmountsToClaim,
            _params.nonce
        );
    }

    /// @notice external wrapper for _isSignatureValid
    function isSignatureValid(ClaimParams memory _params) external view returns (bool) {
        return _isSignatureValid(_params);
    }

    /**
     * @notice Checks if the provided signature is valid
     * @param _params A ClaimParams struct containing the investment parameters
     * @return true if the signer address is recovered from the signature, false otherwise
     */
    function _isSignatureValid(ClaimParams memory _params) private view returns (bool) {
        bytes32 digest = keccak256(
            abi.encodePacked(
                "\x19\x01",
                DOMAIN_SEPARATOR,
                keccak256(
                    abi.encode(
                        CLAIM_TYPEHASH,
                        _params.kycAddress,
                        _params.projectTokenAddress,
                        _params.projectTokenProxyWallets,
                        _params.tokenAmountsToClaim,
                        _params.nonce,
                        _params.deadline
                    )
                )
            )
        );

        address recoveredAddress = ECDSA.recover(digest, _params.signature);

        bool isSigner = recoveredAddress == signer;
        bool isExpired = block.timestamp > _params.deadline;
        return isSigner && !isExpired;
    }

    /// @notice admin function to pause claims
    function setClaimIsPaused(bool _isPaused) external onlyAuthorized {
        claimIsPaused = _isPaused;
    }
}
