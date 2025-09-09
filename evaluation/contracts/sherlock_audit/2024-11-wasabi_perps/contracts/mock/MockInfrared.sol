// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {IInfrared, IInfraredVault} from "../bera/IInfrared.sol";
import {MockInfraredVault} from "./MockInfraredVault.sol";
import {IRewardVaultFactory} from "@berachain/pol-contracts/src/pol/interfaces/IRewardVaultFactory.sol";

contract MockInfrared is IInfrared {
    IRewardVaultFactory public rewardVaultFactory;
    mapping(address asset => IInfraredVault vault) public assetToInfraredVault;

    constructor(IRewardVaultFactory _rewardVaultFactory) {
        rewardVaultFactory = _rewardVaultFactory;
    }

    function registerVault(address _asset) external override returns (IInfraredVault vault) {
        address rewardVault = rewardVaultFactory.getVault(_asset);
        if (rewardVault == address(0)) {
            rewardVault = rewardVaultFactory.createRewardVault(_asset);
        }
        vault = new MockInfraredVault(_asset, rewardVault);
        assetToInfraredVault[_asset] = vault;
        emit NewVault(msg.sender, _asset, address(vault));
    }

    event NewVault(address indexed caller, address indexed asset, address indexed vault);
}