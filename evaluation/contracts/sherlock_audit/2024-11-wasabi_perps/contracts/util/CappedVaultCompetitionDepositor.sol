// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/cryptography/EIP712Upgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

import "../admin/PerpManager.sol";
import "../admin/Roles.sol";
import "../vaults/IWasabiVault.sol";

contract CappedVaultCompetitionDepositor is UUPSUpgradeable, ReentrancyGuardUpgradeable, OwnableUpgradeable {
    using SafeERC20 for IERC20;

    error AllocationsLengthMismatch();
    error SenderNotAllocated();
    error InsufficientBalance();
    error DepositWindowClosed();

    IWasabiVault public vault;
    mapping(address => uint256) public depositAllocations;

    /**
     * @dev Checks if the caller is an admin
     */
    modifier onlyAdmin() {
        _getManager().isAdmin(msg.sender);
        _;
    }

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    /// @dev Initializes the contract
    /// @param _vault The vault to deposit into
    /// @param _manager The PerpManager contract that handles role management
    function initialize(IWasabiVault _vault, PerpManager _manager) public initializer {
        __UUPSUpgradeable_init();
        __ReentrancyGuard_init();
        __Ownable_init(address(_manager));

        vault = _vault;
    }
    
    /// @dev Raises the deposit cap and deposits the user's allocation into the vault
    /// @notice Caller must have been allocated a deposit amount and must have the full amount in their wallet
    /// @notice This contract must have ADMIN role to raise the deposit cap
    function deposit() external nonReentrant {
        // Check if the user is allocated
        uint256 allocation = depositAllocations[msg.sender];
        if (allocation == 0) revert SenderNotAllocated();
        delete depositAllocations[msg.sender];

        // Check if the user has the full allocation in their wallet
        IERC20 asset = IERC20(vault.asset());
        if (asset.balanceOf(msg.sender) < allocation) revert InsufficientBalance();

        // Check if the contract has VAULT_ADMIN role
        (bool isVaultAdmin, ) = _getManager().hasRole(Roles.VAULT_ADMIN_ROLE, address(this));
        if (!isVaultAdmin) revert DepositWindowClosed();

        // Increase the deposit cap (this contract must have ADMIN role)
        uint256 currentDepositCap = vault.getDepositCap();
        uint256 totalAssets = vault.totalAssets();
        if (totalAssets > currentDepositCap) {
            currentDepositCap = totalAssets;
        }
        vault.setDepositCap(currentDepositCap + allocation);

        // Deposit on behalf of the user
        asset.safeTransferFrom(msg.sender, address(this), allocation);
        asset.forceApprove(address(vault), allocation);
        vault.deposit(allocation, msg.sender);
    }

    /// @dev Sets the allocations for the competition winners
    /// @param _users The addresses of the competition winners
    /// @param _allocations The allocations for the competition winners
    function setAllocations(address[] calldata _users, uint256[] calldata _allocations) external onlyAdmin {
        uint256 usersLength = _users.length;
        if (usersLength != _allocations.length) revert AllocationsLengthMismatch();

        for (uint256 i; i < usersLength; ) {
            depositAllocations[_users[i]] = _allocations[i];
            unchecked {
                i++;
            }
        }
    }

    /// @dev Renounces the VAULT_ADMIN role from the contract
    /// @notice Once called, the contract will no longer be able to raise the deposit cap
    function renounceVaultAdmin() external onlyAdmin {
        _getManager().renounceRole(Roles.VAULT_ADMIN_ROLE, address(this));
    }

    /// @inheritdoc UUPSUpgradeable
    function _authorizeUpgrade(address) internal view override onlyAdmin {}

    /// @dev returns the manager of the contract
    function _getManager() internal view returns (PerpManager) {
        return PerpManager(owner());
    }
}