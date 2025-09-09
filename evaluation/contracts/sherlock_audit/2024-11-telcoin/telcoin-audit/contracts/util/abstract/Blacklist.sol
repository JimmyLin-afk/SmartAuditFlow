// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {AccessControlUpgradeable} from "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";

/**
 * @title Blacklist
 * @author Amir Shirif
 * @notice A Telcoin Contract
 * @notice This contract is meant to allow for the prevention of the interaction of certain addreses
 */
abstract contract Blacklist is AccessControlUpgradeable {
    /// @custom:storage-location erc7201:telcoin.storage.Blacklist
    struct BlacklistStorage {
        mapping(address => bool) _blacklist;
    }

    // keccak256(abi.encode(uint256(keccak256("erc7201.telcoin.storage.Blacklist")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant BlacklistStorageLocation =
        0x74958077aaf7e7942dc6838d7ab914c8ab92cc0aa85fe20a608ddf5aa4c04000;

    function _getBlacklistStorage()
        private
        pure
        returns (BlacklistStorage storage $)
    {
        assembly {
            $.slot := BlacklistStorageLocation
        }
    }

    bytes32 public constant BLACKLISTER_ROLE = keccak256("BLACKLISTER_ROLE");

    /**
     * @dev reverts when blacklist attempting to interact
     */
    error Blacklisted(address account);

    /**
     * @dev reverts if the blacklisting of an already blacklisted address is attempted
     */
    error AlreadyBlacklisted(address user);

    /**
     * @dev reverts if the removal of a blacklisting of an address not blacklisted is attempted
     */
    error NotBlacklisted(address user);

    /**
     * @dev emits when address is blacklisted
     */
    event AddedBlacklist(address user);

    /**
     * @dev emits when address is removed from blacklist
     */
    event RemovedBlacklist(address user);

    /************************************************
     *   blacklist fuctions
     ************************************************/

    /**
     * @notice returns blacklsit status of address
     * @return bool representing blacklist status
     */
    function blacklisted(address user) public view returns (bool) {
        BlacklistStorage storage $ = _getBlacklistStorage();
        return $._blacklist[user];
    }

    /**
     * @notice updates blacklisted list to include user
     * @dev restricted to BLACKLISTER_ROLE
     * @param user blacklisted address
     */
    function addBlackList(
        address user
    ) public virtual onlyRole(BLACKLISTER_ROLE) {
        if (blacklisted(user)) revert AlreadyBlacklisted(user);
        _onceBlacklisted(user);
        // do this after so it doesnt trip blacklisted restirctions
        _setBlacklist(user, true);
        emit AddedBlacklist(user);
    }

    /**
     * @notice updates blacklisted list to remove user
     * @dev restricted to BLACKLISTER_ROLE
     * @param user blacklisted address
     */
    function removeBlackList(
        address user
    ) public virtual onlyRole(BLACKLISTER_ROLE) {
        if (!blacklisted(user)) revert NotBlacklisted(user);
        _setBlacklist(user, false);
        emit RemovedBlacklist(user);
    }

    // Internal function to set the blacklist state of an address
    function _setBlacklist(address user, bool state) internal virtual {
        BlacklistStorage storage $ = _getBlacklistStorage();
        $._blacklist[user] = state;
    }

    // Internal hook that can be overridden for custom logic when an address is blacklisted
    function _onceBlacklisted(address user) internal virtual {}
}
