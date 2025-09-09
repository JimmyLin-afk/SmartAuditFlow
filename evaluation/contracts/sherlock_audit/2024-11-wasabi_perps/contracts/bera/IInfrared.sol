// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {IInfraredVault} from "./IInfraredVault.sol";
import {IInfraredBGT} from "./IInfraredBGT.sol";

/**
 * @dev Simplified interface for the Infrared contract, which is responsible for deploying new InfraredVaults.
 */
interface IInfrared {
    /**
     * @notice Registers a new vault for a given asset
     * @dev Infrared.sol must be admin over MINTER_ROLE on InfraredBGT to grant minter role to deployed vault
     * @param _asset The address of the asset, such as a specific LP token
     * @return vault The address of the newly created InfraredVault contract
     * @custom:emits NewVault with the caller, asset address, and new vault address.
     */
    function registerVault(address _asset) external returns (IInfraredVault vault);
}