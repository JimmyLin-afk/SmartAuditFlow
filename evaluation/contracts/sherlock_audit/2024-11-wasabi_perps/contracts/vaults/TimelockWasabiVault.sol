// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "./WasabiVault.sol";
import "./ITimelock.sol";

contract TimelockWasabiVault is WasabiVault, ITimelock {
    using SafeERC20 for IERC20;

    // @notice The slot where the TimelockStorage struct is stored
    // @dev This equals bytes32(uint256(keccak256("wasabi.vault.timelock_storage")) - 1)
    bytes32 private constant TIMELOCK_STORAGE_SLOT = 0x62e75e49e19da7df0349e60d38b9c7cdb591478ceb82eff336d44e21a7b66787;

    /// @dev Initializer for proxy
    /// @notice This function should only be called to initialize a new vault
    /// @param _longPool The WasabiLongPool contract
    /// @param _shortPool The WasabiShortPool contract
    /// @param _addressProvider The address provider
    /// @param _manager The PerpManager contract that will own this vault
    /// @param _asset The asset
    /// @param name The name of the vault
    /// @param symbol The symbol of the vault
    /// @param _cooldownDuration The duration of the cooldown period in seconds
    function initialize(
        IWasabiPerps _longPool,
        IWasabiPerps _shortPool,
        IAddressProvider _addressProvider,
        PerpManager _manager,
        IERC20 _asset,
        string memory name,
        string memory symbol,
        uint256 _cooldownDuration
    ) public virtual initializer {
        __WasabiVault_init(_longPool, _shortPool, _addressProvider, _manager, _asset, name, symbol);
        _getTimelockStorage().cooldownDuration = _cooldownDuration;
        emit CooldownDurationUpdated(0, _cooldownDuration);
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                          GETTERS                           */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    function getCooldownDuration() external view returns (uint256) {
        return _getTimelockStorage().cooldownDuration;
    }

    function getCooldowns(address account) external view returns (Cooldown[] memory) {
        return _getTimelockStorage().cooldowns[account];
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                           WRITES                           */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    function startCooldown(uint256 amount) external {
        if (amount == 0) revert InvalidCooldownAmount();

        Cooldown[] storage cooldowns = _getTimelockStorage().cooldowns[msg.sender];
        uint256 stakedBalance = balanceOf(msg.sender);
        for (uint256 i; i < cooldowns.length; ) {
            stakedBalance -= cooldowns[i].amount;
            unchecked {
                i++;
            }
        }
        if (stakedBalance < amount) revert InvalidCooldownAmount();

        // Add cooldown to the end of the array
        // @dev Cooldowns should always remain sorted by cooldownStart
        cooldowns.push(Cooldown({
            cooldownStart: block.timestamp,
            amount: amount
        }));

        emit CooldownStarted(msg.sender, amount);
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                       ADMIN FUNCTIONS                      */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    function setCooldownDuration(uint256 duration) external onlyAdmin {
        TimelockStorage storage ts = _getTimelockStorage();
        emit CooldownDurationUpdated(ts.cooldownDuration, duration);
        ts.cooldownDuration = duration;
    }

    /*´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:*/
    /*                     INTERNAL FUNCTIONS                     */
    /*.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•*/

    /// @inheritdoc ERC4626Upgradeable
    function _withdraw(
        address caller,
        address receiver,
        address owner,
        uint256 assets,
        uint256 shares
    ) internal override {
        _checkAndUpdateCooldowns(owner, shares);
        super._withdraw(caller, receiver, owner, assets, shares);
    }

    function _checkAndUpdateCooldowns(address owner, uint256 withdrawAmount) internal {
        TimelockStorage storage ts = _getTimelockStorage();
        uint256 cooldownDuration = ts.cooldownDuration;

        Cooldown[] storage cooldowns = ts.cooldowns[owner];
        uint256 cooldownsLength = cooldowns.length;
        uint256 nextCooldownIndex = ts.nextCooldownIndex[owner];
        // If there are no cooldowns or the next cooldown index is out of bounds, the cooldowns are insufficient
        // If cooldown duration is 0, withdrawals are not timelocked
        if (cooldownDuration != 0) {
            if (cooldownsLength == 0 || nextCooldownIndex >= cooldownsLength) revert InsufficientCooldown();
        }

        for (uint256 i = nextCooldownIndex; i < cooldownsLength; ) {
            Cooldown memory cooldown = cooldowns[i];
            // Break if the cooldown has not ended, since all subsequent cooldowns will also not have ended
            if (block.timestamp < cooldown.cooldownStart + cooldownDuration) break;
            if (withdrawAmount >= cooldown.amount) {
                // Completely withdraw this cooldown
                withdrawAmount -= cooldown.amount;
                // Set values to 0 without shifting or changing array length
                delete cooldowns[i];
                // Skip this cooldown in future iterations
                ts.nextCooldownIndex[owner] = i + 1;
                if (withdrawAmount == 0) break;
                unchecked {
                    i++;
                }
            } else {
                // Withdraw partial amount from this cooldown
                cooldowns[i].amount -= withdrawAmount;
                withdrawAmount = 0;
                break;
            }
        }

        // If there is still a remaining amount to withdraw, the cooldowns are insufficient
        if (withdrawAmount > 0 && cooldownDuration != 0) {
            revert InsufficientCooldown();
        }
    }

    function _getTimelockStorage() internal pure returns (TimelockStorage storage $) {
        assembly {
            $.slot := TIMELOCK_STORAGE_SLOT
        }
    }
}