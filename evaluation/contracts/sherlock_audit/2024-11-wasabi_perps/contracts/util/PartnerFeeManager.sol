// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import {UUPSUpgradeable} from "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import {ReentrancyGuardUpgradeable} from "@openzeppelin/contracts-upgradeable/utils/ReentrancyGuardUpgradeable.sol";
import {OwnableUpgradeable} from "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import {Math} from "@openzeppelin/contracts/utils/math/Math.sol";

import {IPartnerFeeManager} from "./IPartnerFeeManager.sol";
import {PerpManager} from "../admin/PerpManager.sol";

contract PartnerFeeManager is UUPSUpgradeable, ReentrancyGuardUpgradeable, OwnableUpgradeable, IPartnerFeeManager {
    using SafeERC20 for IERC20;
    using Math for uint256;

    uint256 private constant FEE_DENOMINATOR = 10000;
    uint256 private constant MAX_FEE_SHARE_BIPS = 5000;
    uint256 private constant DEFAULT_FEE_SHARE_BIPS = 2000; // 20%

    address private _longPool;
    address private _shortPool;
    mapping(address => uint256) private _partnerFeeShareBips;
    mapping(address => mapping(address => uint256)) private _accruedFees;

    /// @dev Checks if the caller is an admin
    modifier onlyAdmin() {
        _getManager().isAdmin(msg.sender);
        _;
    }

    /// @dev Checks if the caller is one of the pool contracts
    modifier onlyPool() {
        if (msg.sender != address(_shortPool)) {
            // Nested checks save a little gas compared to using &&
            if (msg.sender != address(_longPool)) revert CallerNotPool();
        }
        _;
    }

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    /// @dev Initializes the contract
    /// @param manager the PerpManager contract address
    /// @param longPool the long pool contract address
    /// @param shortPool the short pool contract address
    function initialize(address manager, address longPool, address shortPool) external initializer {
        __UUPSUpgradeable_init();
        __ReentrancyGuard_init();
        __Ownable_init(manager);
        _longPool = longPool;
        _shortPool = shortPool;
    }

    /// @inheritdoc IPartnerFeeManager
    function isPartner(address partner) external view returns (bool) {
        return _partnerFeeShareBips[partner] != 0;
    }

    /// @inheritdoc IPartnerFeeManager
    function getAccruedFees(address partner, address feeToken) external view returns (uint256) {
        return _accruedFees[partner][feeToken];
    }

    /// @inheritdoc IPartnerFeeManager
    function computePartnerFees(address partner, uint256 totalFees) external view returns (uint256) {
        if (partner == address(0)) return 0;
        uint256 feeShareBips = _partnerFeeShareBips[partner];
        if (feeShareBips == 0) {
            feeShareBips = DEFAULT_FEE_SHARE_BIPS;
        }
        return totalFees.mulDiv(feeShareBips, FEE_DENOMINATOR);
    }

    /// @inheritdoc IPartnerFeeManager
    function accrueFees(
        address partner, 
        address feeToken, 
        uint256 partnerFees
    ) external onlyPool {
        if (partnerFees == 0) return;

        IERC20(feeToken).safeTransferFrom(msg.sender, address(this), partnerFees);
        _accruedFees[partner][feeToken] += partnerFees;

        emit FeesAccrued(partner, feeToken, partnerFees);
    }

    /// @inheritdoc IPartnerFeeManager
    function claimFees(address[] calldata feeTokens) external nonReentrant {
        uint256 length = feeTokens.length;
        for (uint256 i = 0; i < length; i++) {
            address feeToken = feeTokens[i];
            uint256 amount = _accruedFees[msg.sender][feeToken];
            if (amount == 0) continue;
            _accruedFees[msg.sender][feeToken] = 0;
            IERC20(feeToken).safeTransfer(msg.sender, amount);
            emit FeesClaimed(msg.sender, feeToken, amount);
        }
    }

    /// @inheritdoc IPartnerFeeManager
    function setFeeShareBips(address partner, uint256 feeShareBips) external onlyAdmin {
        if (feeShareBips > MAX_FEE_SHARE_BIPS) revert InvalidFeeShareBips();
        _partnerFeeShareBips[partner] = feeShareBips;
    }

    /// @inheritdoc IPartnerFeeManager
    function adminAddFees(address partner, address feeToken, uint256 amount) external onlyAdmin {
        IERC20(feeToken).safeTransferFrom(msg.sender, address(this), amount);
        _accruedFees[partner][feeToken] += amount;
        emit FeesAccrued(partner, feeToken, amount);
    }

    /// @inheritdoc UUPSUpgradeable
    function _authorizeUpgrade(address) internal view override onlyAdmin {}

    /// @dev returns the manager of the contract
    function _getManager() internal view returns (PerpManager) {
        return PerpManager(owner());
    }
}
