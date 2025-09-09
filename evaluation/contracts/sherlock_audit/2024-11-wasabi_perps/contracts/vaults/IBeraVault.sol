// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {IWasabiVault} from "./IWasabiVault.sol";
import {IRewardVault, IInfraredVault} from "../bera/IInfraredVault.sol";

interface IBeraVault is IWasabiVault {
    error InvalidFeeBips(); // 0x82c96382
    error NoSharesToUnstake(); // 0xb24d96eb

    event RewardFeeBipsUpdated(uint256 oldFeeBips, uint256 newFeeBips);

    /// @notice Get the POL RewardVault contract where spicy tokens are staked
    function getRewardVault() external view returns (IRewardVault);

    /// @notice Get the InfraredVault contract that handles staking and gives iBGT rewards
    function getInfraredVault() external view returns (IInfraredVault);

    /// @notice Get the fee charged on BGT rewards
    function getRewardFeeBips() external view returns (uint256);

    /// @notice Get the total balance of a user, including shares staked in the InfraredVault and RewardVault
    function cumulativeBalanceOf(address account) external view returns (uint256);

    /// @notice Unstake shares from the RewardVault that were staked on the user's behalf
    function unstakeShares() external;

    /// @notice Set the fee charged on BGT rewards
    /// @param _rewardFeeBips The fee charged on BGT rewards in basis points
    function setRewardFeeBips(uint256 _rewardFeeBips) external;
    
    /// @notice Claim the rewards accrued to the vault from the InfraredVault
    /// @dev Infrared vaults can give multiple rewards, so return array has same length of InfraredVault.getAllRewardTokens()
    /// @param _receiver The address to receive the rewards
    /// @return The amount of each reward token claimed
    function claimRewardFees(address _receiver) external returns (uint256[] memory);
}