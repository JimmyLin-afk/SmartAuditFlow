// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import "../vaults/BeraVault.sol";

contract MockBeraVault is BeraVault {

    /// @dev Initializer for proxy
    /// @notice This function should only be called to initialize a new vault
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
        __ERC20_init(name, symbol);
        __Ownable_init(address(_manager));
        __ERC4626_init(_asset);
        __ReentrancyGuard_init();
        __UUPSUpgradeable_init();
        addressProvider = _addressProvider;
        longPool = _longPool;
        shortPool = _shortPool;
        interestFeeBips = 1000; // 10%
    }

    /// @notice Initialize the reward vaults
    function initializeRewardVaultsWithInfrared(IInfrared _infrared) external onlyAdmin {
        RewardStorage storage rs = _getRewardStorage();
        rs.infraredVault = _infrared.registerVault(address(this));
        rs.rewardVault = rs.infraredVault.rewardsVault();
        rs.rewardFeeBips = 1000; // 10%

        _approve(address(this), address(rs.infraredVault), type(uint256).max);
    }

    function getRewardFeeUserBalance(address account) external view returns (uint256) {
        return _getRewardStorage().rewardFeeUserBalance[account];
    }
}