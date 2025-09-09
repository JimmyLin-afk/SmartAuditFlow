// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "../blast/AbstractBlastContract.sol";
import "./WasabiVault.sol";

contract BlastVault is WasabiVault, AbstractBlastContract {

    error UnexpectedTotalSupply();

    /// @dev Initializer for proxy
    /// @notice This function should only be called to initialize a new vault - for upgrading an existing V1 vault use `reinitialize`
    /// @param _longPool The WasabiLongPool contract
    /// @param _shortPool The WasabiShortPool contract
    /// @param _addressProvider The address provider
    /// @param _manager The PerpManager contract that will own this vault
    /// @param _asset The asset
    /// @param name The name of the vault
    /// @param symbol The symbol of the vault
    function initialize(
        IWasabiPerps _longPool,
        IWasabiPerps _shortPool,
        IAddressProvider _addressProvider,
        PerpManager _manager,
        IERC20 _asset,
        string memory name,
        string memory symbol
    ) public override initializer {
        __AbstractBlastContract_init();
        __WasabiVault_init(_longPool, _shortPool, _addressProvider, _manager, _asset, name, symbol);
    }

    /// @dev claim all gas
    function claimGas(address contractAddress, address recipientOfGas) external onlyAdmin returns (uint256) {
        return _getBlast().claimMaxGas(contractAddress, recipientOfGas);
    }

    /// @dev claims yield
    function claimYield(uint256 _expectedVaultSupply) external onlyAdmin {
        if (totalSupply() != _expectedVaultSupply) {
            revert UnexpectedTotalSupply();
        }
        address assetAddress = asset();
        if (assetAddress == BlastConstants.WETH || assetAddress == BlastConstants.USDB) {
            IERC20Rebasing token = IERC20Rebasing(assetAddress);
            uint256 claimable = token.getClaimableAmount(address(this));
            if (claimable > 0) {
                uint256 claimed = token.claim(address(this), claimable);
                totalAssetValue += claimed;
                emit NativeYieldClaimed(assetAddress, claimed);
            }
        } else {
            revert CannotClaimNonYieldBearingAsset(assetAddress);
        }
    }
}