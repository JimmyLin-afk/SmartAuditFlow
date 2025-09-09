// SPDX-License-Identifier: GPL-3.0-or-later
// solhint-disable immutable-vars-naming, gas-custom-errors
pragma solidity 0.8.18;

import { MerkleProof } from "openzeppelin-contracts-next/contracts/utils/cryptography/MerkleProof.sol";
import { IERC20 } from "openzeppelin-contracts-next/contracts/token/ERC20/IERC20.sol";
import { SafeERC20 } from "openzeppelin-contracts-next/contracts/token/ERC20/utils/SafeERC20.sol";
import { ECDSA } from "openzeppelin-contracts-next/contracts/utils/cryptography/ECDSA.sol";
import { SignatureChecker } from "openzeppelin-contracts-next/contracts/utils/cryptography/SignatureChecker.sol";
import { Strings } from "openzeppelin-contracts-next/contracts/utils/Strings.sol";
import { ReentrancyGuard } from "openzeppelin-contracts-next/contracts/security/ReentrancyGuard.sol";

import { ILocking } from "./locking/interfaces/ILocking.sol";

/**
 * @title Airgrab
 * @author Mento Labs
 * @notice This contract implements a token airgrab gated by a MerkleTree and KYC using fractal.
 * The airgrab also forces claimers to immediately lock their tokens as veTokens for a
 * predetermined period.
 */
contract Airgrab is ReentrancyGuard {
  using SafeERC20 for IERC20;

  uint32 public constant MAX_CLIFF_PERIOD = 103;
  uint32 public constant MAX_SLOPE_PERIOD = 104;

  /**
   * @notice FractalProof struct for KYC/KYB verification.
   * @param sig The signature of the Fractal Credential.
   * @param validUntil The timestamp when the Fractal Credential expires.
   * @param approvedAt The timestamp when the Fractal Credential was approved.
   * @param fractalId The Fractal Credential ID.
   */
  struct FractalProof {
    bytes sig;
    uint256 validUntil;
    uint256 approvedAt;
    string fractalId;
  }

  /**
   * @notice Emitted when tokens are claimed
   * @param claimer The account claiming the tokens
   * @param amount The amount of tokens being claimed
   * @param lockId The ID of the resulting veMento lock
   */
  event TokensClaimed(address indexed claimer, uint256 amount, uint256 lockId);

  /**
   * @notice Emitted when tokens are drained
   * @param token The token addresses that was drained
   * @param amount The amount drained
   */
  event TokensDrained(address indexed token, uint256 amount);

  /// @notice The root of the merkle tree.
  bytes32 public immutable root;
  /// @notice The Fractal Credential message signer for KYC/KYB.
  address public immutable fractalSigner;
  /// @notice The Fractal Credential maximum age in seconds
  uint256 public immutable fractalMaxAge;
  /// @notice The timestamp when the airgrab ends.
  uint256 public immutable endTimestamp;
  /// @notice The slope period that the airgrab will be locked for.
  uint32 public immutable slopePeriod;
  /// @notice The cliff period that the airgrab will be locked for.
  uint32 public immutable cliffPeriod;
  /// @notice The token in the airgrab.
  IERC20 public immutable token;
  /// @notice The locking contract for veToken.
  ILocking public immutable locking;
  /// @notice The Mento Treasury address where unclaimed tokens will be refunded to.
  address payable public immutable mentoTreasury;

  /// @notice The map of addresses that have claimed
  mapping(address => bool) public claimed;

  /**
   * @dev Check if the account has a valid kyc signature.
   * See: https://docs.developer.fractal.id/fractal-credentials-api
   *      https://github.com/trustfractal/credentials-api-verifiers
   * @notice This function checks the kyc signature with the data provided.
   * @param account The address of the account to check.
   * @param proof FractaLProof kyc proof data for the account.
   */
  modifier hasValidKyc(address account, FractalProof memory proof) {
    require(block.timestamp < proof.validUntil, "Airgrab: KYC no longer valid");
    require(fractalMaxAge == 0 || block.timestamp < proof.approvedAt + fractalMaxAge, "Airgrab: KYC not recent enough");
    string memory accountString = Strings.toHexString(uint256(uint160(account)), 20);

    bytes32 signedMessageHash = ECDSA.toEthSignedMessageHash(
      abi.encodePacked(
        accountString,
        ";",
        proof.fractalId,
        ";",
        Strings.toString(proof.approvedAt),
        ";",
        Strings.toString(proof.validUntil),
        ";",
        // ISO 3166-1 alpha-2 country codes
        // DRC, CUBA, GB, IRAN, DPKR, MALI, MYANMAR, SOUTH SUDAN, SYRIA, US, YEMEN
        "level:plus+liveness;citizenship_not:;residency_not:cd,cu,gb,ir,kp,ml,mm,ss,sy,us,ye"
      )
    );

    require(SignatureChecker.isValidSignatureNow(fractalSigner, signedMessageHash, proof.sig), "Airgrab: Invalid KYC");

    _;
  }

  /**
   * @dev Check if the account can claim
   * @notice This modifier checks if the airgrab is still active,
   * if the account hasn't already claimed and if it's included
   * in the MerkleTree.
   * @param account The address of the account to check.
   * @param amount The amount of tokens to be claimed.
   * @param merkleProof The merkle proof for the account.
   */
  modifier canClaim(address account, uint256 amount, bytes32[] calldata merkleProof) {
    require(block.timestamp <= endTimestamp, "Airgrab: finished");
    require(!claimed[account], "Airgrab: already claimed");
    bytes32 leaf = keccak256(bytes.concat(keccak256(abi.encode(account, amount))));
    require(MerkleProof.verify(merkleProof, root, leaf), "Airgrab: not in tree");
    _;
  }

  /**
   * @dev Constructor for the Airgrab contract.
   * @notice It checks and configures all immutable params
   * @param root_ The root of the merkle tree.
   * @param fractalSigner_ The Fractal message signer for KYC/KYB.
   * @param fractalMaxAge_ The Fractal Credential maximum age in seconds.
   * @param endTimestamp_ The timestamp when the airgrab ends.
   * @param cliffPeriod_ The cliff period that the airgrab will be locked for.
   * @param slopePeriod_ The slope period that the airgrab will be locked for.
   * @param token_ The token address in the airgrab.
   * @param locking_ The locking contract for veToken.
   * @param mentoTreasury_ The Mento Treasury address where unclaimed tokens will be refunded to.
   */
  constructor(
    bytes32 root_,
    address fractalSigner_,
    uint256 fractalMaxAge_,
    uint256 endTimestamp_,
    uint32 cliffPeriod_,
    uint32 slopePeriod_,
    address token_,
    address locking_,
    address payable mentoTreasury_
  ) {
    require(root_ != bytes32(0), "Airgrab: invalid root");
    require(fractalSigner_ != address(0), "Airgrab: invalid fractal issuer");
    require(endTimestamp_ > block.timestamp, "Airgrab: invalid end timestamp");
    require(cliffPeriod_ <= MAX_CLIFF_PERIOD, "Airgrab: cliff period too large");
    require(slopePeriod_ <= MAX_SLOPE_PERIOD, "Airgrab: slope period too large");
    require(token_ != address(0), "Airgrab: invalid token");
    require(locking_ != address(0), "Airgrab: invalid locking");
    require(mentoTreasury_ != address(0), "Airgrab: invalid Mento Treasury");

    root = root_;
    fractalSigner = fractalSigner_;
    fractalMaxAge = fractalMaxAge_;
    endTimestamp = endTimestamp_;
    cliffPeriod = cliffPeriod_;
    slopePeriod = slopePeriod_;
    token = IERC20(token_);
    locking = ILocking(locking_);
    mentoTreasury = mentoTreasury_;

    require(token.approve(locking_, type(uint256).max), "Airgrab: approval failed");
  }

  /**
   * @dev Allows `msg.sender` to claim `amount` tokens if the merkle proof and kyc is valid.
   * @notice This function can be called by anybody, but the (msg.sender, amount) pair
   * must be in the merkle tree, has to not have claimed yet, and must have
   * an associated KYC signature from Fractal. And the airgrab must not have ended.
   * The tokens will be locked for the cliff and slope configured at the contract level.
   * @param amount The amount of tokens to be claimed.
   * @param delegate The address of the account that gets voting power delegated
   * @param merkleProof The merkle proof for the account.
   * @param fractalProof The Fractal KYC proof for the account.
   */
  function claim(
    uint96 amount,
    address delegate,
    bytes32[] calldata merkleProof,
    FractalProof calldata fractalProof
  ) external hasValidKyc(msg.sender, fractalProof) canClaim(msg.sender, amount, merkleProof) nonReentrant {
    require(token.balanceOf(address(this)) >= amount, "Airgrab: insufficient balance");
    _claim(amount, delegate);
  }

  /**
   * @dev Internal function to claim tokens and lock them.
   * @param amount The amount of tokens to be claimed.
   * @param delegate The address of the account that gets voting power delegated
   */
  function _claim(uint96 amount, address delegate) internal {
    claimed[msg.sender] = true;
    uint256 lockId = locking.lock(msg.sender, delegate, amount, slopePeriod, cliffPeriod);
    emit TokensClaimed(msg.sender, amount, lockId);
  }

  /**
   * @dev Allows the Mento Treasury to reclaim any tokens after the airgrab has ended.
   * @notice This function can only be called after the airgrab has ended.
   * @param tokenToDrain Token is parameterized in case the contract has been sent
   *  tokens other than the airgrab token.
   */
  function drain(address tokenToDrain) external nonReentrant {
    require(block.timestamp > endTimestamp, "Airgrab: not finished");
    uint256 balance = IERC20(tokenToDrain).balanceOf(address(this));
    require(balance > 0, "Airgrab: nothing to drain");
    IERC20(tokenToDrain).safeTransfer(mentoTreasury, balance);
    emit TokensDrained(tokenToDrain, balance);
  }
}
