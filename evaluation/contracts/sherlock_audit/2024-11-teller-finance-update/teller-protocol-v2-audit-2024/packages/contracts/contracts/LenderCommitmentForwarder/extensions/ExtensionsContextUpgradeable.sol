// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "../../interfaces/IExtensionsContext.sol";
import "@openzeppelin/contracts-upgradeable/metatx/ERC2771ContextUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/structs/EnumerableSetUpgradeable.sol";

abstract contract ExtensionsContextUpgradeable is IExtensionsContext {
    using EnumerableSetUpgradeable for EnumerableSetUpgradeable.AddressSet;

    // Mapping from owner to operator approvals
    mapping(address => mapping(address => bool)) private userExtensions;

    event ExtensionAdded(address extension, address sender);
    event ExtensionRevoked(address extension, address sender);

    function hasExtension(address account, address extension)
        public
        view
        returns (bool)
    {
        return userExtensions[account][extension];
    }

    function addExtension(address extension) external {
        require(
            _msgSender() != extension,
            "ExtensionsContextUpgradeable: cannot approve own extension"
        );

        userExtensions[_msgSender()][extension] = true;
        emit ExtensionAdded(extension, _msgSender());
    }

    function revokeExtension(address extension) external {
        userExtensions[_msgSender()][extension] = false;
        emit ExtensionRevoked(extension, _msgSender());
    }

    function _msgSender() internal view virtual returns (address sender) {
        address sender;

        if (msg.data.length >= 20) {
            assembly {
                sender := shr(96, calldataload(sub(calldatasize(), 20)))
            }

            if (hasExtension(sender, msg.sender)) {
                return sender;
            }
        }

        return msg.sender;
    }

    /**
     * @dev This empty reserved space is put in place to allow future versions to add new
     * variables without shifting down storage in the inheritance chain.
     * See https://docs.openzeppelin.com/contracts/4.x/upgradeable#storage_gaps
     */
    uint256[49] private __gap;
}
