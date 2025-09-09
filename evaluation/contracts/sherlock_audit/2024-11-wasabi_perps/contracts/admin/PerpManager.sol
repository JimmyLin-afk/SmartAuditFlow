// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/manager/AccessManagerUpgradeable.sol";

contract PerpManager is UUPSUpgradeable, AccessManagerUpgradeable {
    event AuthorizedSignerChanged(
        address indexed trader,
        address indexed signer,
        bool isAuthorized
    );

    mapping(address trader => mapping(address signer => bool isAuthorized)) private _isAuthorizedSigner;

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    /// @dev initializer for proxy
    function initialize() public virtual initializer {
        __AccessManager_init(msg.sender);
    }

    /// @inheritdoc UUPSUpgradeable
    function _authorizeUpgrade(address) internal view override {
        isAdmin(msg.sender);
    }

    /// @dev check if account is admin
    function isAdmin(address account) public view {
        checkRole(ADMIN_ROLE, account);
    }

    /// @dev check if account has the given role
    /// @param roleId role id
    /// @param account account address
    function checkRole(uint64 roleId, address account) public view {
        (bool hasRole, ) = hasRole(roleId, account);
        if (!hasRole) revert AccessManagerUnauthorizedAccount(account, roleId);
    }

    /// @notice check if a signer is authorized to sign for a trader
    /// @param trader address of the account to sign on behalf of
    /// @param signer address of the signer
    /// @return isAuthorized true if the signer is authorized to sign for the trader, false otherwise
    function isAuthorizedSigner(address trader, address signer) public view returns (bool) {
        return _isAuthorizedSigner[trader][signer];
    }

    /// @notice Authorize or deauthorize a signer for a trader
    /// @param signer address of the signer to authorize or deauthorize
    /// @param isAuthorized true to authorize the signer, false to deauthorize
    function setAuthorizedSigner(address signer, bool isAuthorized) public {
        _isAuthorizedSigner[msg.sender][signer] = isAuthorized;
        emit AuthorizedSignerChanged(msg.sender, signer, isAuthorized);
    }
}