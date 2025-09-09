///SPDX-License-Identifier: MIT
pragma solidity 0.8.23;

import { AccessControlDefaultAdminRules } from "@openzeppelin/contracts/access/extensions/AccessControlDefaultAdminRules.sol";

/**
    @title VVVAuthorizationRegistry
    @notice This contract manages function-level permissions for relevant VVV contracts
 */

contract VVVAuthorizationRegistry is AccessControlDefaultAdminRules {
    ///@notice maps contract address (20 bytes) + function selector (4 bytes) to role
    mapping(bytes24 => bytes32) public permissions;

    ///@notice AccessControlDefaultAdminRules constructor sets up DEFAULT_ADMIN_ROLE
    constructor(
        uint48 _defaultAdminTransferDelay,
        address _defaultAdmin
    ) AccessControlDefaultAdminRules(_defaultAdminTransferDelay, _defaultAdmin) {}

    /**
        @notice checks if caller is authorized to call the function on the specified contract
        @param _contractToCall The address of the contract being called by the user 
        @param _functionSelector The function selector
        @return bool true if the caller is authorized
     */

    function isAuthorized(
        address _contractToCall,
        bytes4 _functionSelector,
        address _caller
    ) external view returns (bool) {
        bytes24 key = _keyFromAddressAndSelector(_contractToCall, _functionSelector);
        bytes32 requiredRole = permissions[key];
        return hasRole(requiredRole, _caller);
    }

    /**
    @notice sets the role required to call a function on a 
    @param _contract The address of the contract
    @param _functionSelector The function selector
    @param _role The role required to call the function
    @dev setting _role to bytes32(0) will revoke the permission from all except DEFAULT_ADMIN_ROLE (=0x00)
     */
    function setPermission(
        address _contract,
        bytes4 _functionSelector,
        bytes32 _role
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        bytes24 key = _keyFromAddressAndSelector(_contract, _functionSelector);
        permissions[key] = _role;
    }

    ///@notice returns the key for the permissions mapping given a contract address and function selector
    function _keyFromAddressAndSelector(
        address _contract,
        bytes4 _functionSelector
    ) private pure returns (bytes24) {
        return bytes24(keccak256(abi.encodePacked(_contract, _functionSelector)));
    }
}
