// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import {IRewardVault} from "@berachain/pol-contracts/src/pol/interfaces/IRewardVault.sol";
import {IMultiRewards} from "./IMultiRewards.sol";

interface IInfraredVault is IMultiRewards {
    /**
     * @notice A struct to hold a user's reward information
     * @param token The address of the reward token
     * @param amount The amount of reward tokens
     */
    struct UserReward {
        address token;
        uint256 amount;
    }

    /**
     * @notice Returns all reward tokens
     * @return An array of reward token addresses
     */
    function getAllRewardTokens() external view returns (address[] memory);

    /**
     * @notice Returns all rewards for a user
     * @notice Only up to date since the `lastUpdateTime`
     * @param _user The address of the user
     * @return An array of UserReward structs
     */
    function getAllRewardsForUser(address _user)
        external
        view
        returns (UserReward[] memory);

    /**
     * @notice Returns the Infrared protocol coordinator
     * @return The address of the Infrared contract
     */
    function infrared() external view returns (address);

    /**
     * @notice Returns the associated Berachain rewards vault
     * @return The rewards vault contract instance
     */
    function rewardsVault() external view returns (IRewardVault);

    /**
     * @notice Updates reward duration for a specific reward token
     * @dev Only callable by Infrared contract
     * @param _rewardsToken The address of the reward token
     * @param _rewardsDuration The new duration in seconds
     * @custom:access-control Requires INFRARED_ROLE
     */
    function updateRewardsDuration(
        address _rewardsToken,
        uint256 _rewardsDuration
    ) external;

    /**
     * @notice Pauses staking functionality on a specific vault
     * @custom:access-control Requires INFRARED_ROLE
     */
    function pauseStaking() external;

    /**
     * @notice Un-pauses staking functionality on a specific vault
     * @custom:access-control Requires INFRARED_ROLE
     */
    function unpauseStaking() external;

    /**
     * @notice Adds a new reward token to the vault
     * @dev Cannot exceed maximum number of reward tokens
     * @param _rewardsToken The reward token to add
     * @param _rewardsDuration The reward period duration
     * @custom:access-control Requires INFRARED_ROLE
     */
    function addReward(address _rewardsToken, uint256 _rewardsDuration)
        external;

    /**
     * @notice Used to remove malicious or unused reward tokens
     * @param _rewardsToken The reward token to remove
     * @custom:access-control Requires INFRARED_ROLE
     */
    function removeReward(address _rewardsToken) external;

    /**
     * @notice Notifies the vault of newly added rewards
     * @dev Updates internal reward rate calculations
     * @param _rewardToken The reward token address
     * @param _reward The amount of new rewards
     */
    function notifyRewardAmount(address _rewardToken, uint256 _reward)
        external;

    /**
     * @notice Recovers accidentally sent tokens
     * @dev Cannot recover staking token or active reward tokens
     * @param _to The address to receive the recovered tokens
     * @param _token The token to recover
     * @param _amount The amount to recover
     */
    function recoverERC20(address _to, address _token, uint256 _amount)
        external;
}