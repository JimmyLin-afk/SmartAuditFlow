// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IExtensionsContext {
    function hasExtension(address extension, address account)
        external
        view
        returns (bool);

    function addExtension(address extension) external;

    function revokeExtension(address extension) external;
}
