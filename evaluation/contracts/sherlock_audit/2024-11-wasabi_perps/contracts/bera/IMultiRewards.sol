// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IMultiRewards {
    /**
     * @notice Emitted when tokens are staked
     * @param user The address of the user who staked
     * @param amount The amount of tokens staked
     */
    event Staked(address indexed user, uint256 amount);

    /**
     * @notice Emitted when tokens are withdrawn
     * @param user The address of the user who withdrew
     * @param amount The amount of tokens withdrawn
     */
    event Withdrawn(address indexed user, uint256 amount);

    /**
     * @notice Emitted when rewards are claimed
     * @param user The address of the user claiming the reward
     * @param rewardsToken The address of the reward token
     * @param reward The amount of rewards claimed
     */
    event RewardPaid(
        address indexed user, address indexed rewardsToken, uint256 reward
    );

    /**
     * @notice Emitted when rewards are added to the contract
     * @param rewardsToken The address of the reward token
     * @param reward The amount of rewards added
     */
    event RewardAdded(address indexed rewardsToken, uint256 reward);

    /**
     * @notice Emitted when a reward is removed from the contract
     */
    event RewardRemoved(address indexed rewardsToken);

    /**
     * @notice Emitted when a rewards distributor is updaRewardAddedd
     * @param rewardsToken The address of the reward token
     * @param newDistributor The address of the new distributor
     */
    event RewardsDistributorUpdated(
        address indexed rewardsToken, address indexed newDistributor
    );

    /**
     * @notice Emitted when the rewards duration for a token is updated
     * @param token The reward token address whose duration was updated
     * @param newDuration The new duration set for the rewards period
     */
    event RewardsDurationUpdated(address token, uint256 newDuration);

    /**
     * @notice Emitted when tokens are recovered from the contract
     * @param token The address of the token that was recovered
     * @param amount The amount of tokens that were recovered
     */
    event Recovered(address token, uint256 amount);

    /**
     * @notice Emitted when new reward data is stored
     * @param rewardsToken The address of the reward token
     * @param rewardsDuration The duration set for the reward period
     */
    event RewardStored(address rewardsToken, uint256 rewardsDuration);

    /**
     * @notice Reward data for a particular reward token
     * @dev Struct containing all relevant information for reward distribution
     */
    struct Reward {
        address rewardsDistributor;
        uint256 rewardsDuration;
        uint256 periodFinish;
        uint256 rewardRate;
        uint256 lastUpdateTime;
        uint256 rewardPerTokenStored;
        uint256 rewardResidual;
    }

    /**
     * @notice Returns the total amount of staked tokens in the contract
     * @return uint256 The total supply of staked tokens
     */
    function totalSupply() external view returns (uint256);

    /**
     * @notice Stakes tokens into the contract
     * @param amount The amount of tokens to stake
     * @dev Transfers `amount` of staking tokens from the user to this contract
     */
    function stake(uint256 amount) external;

    /**
     * @notice Withdraws staked tokens from the contract
     * @param amount The amount of tokens to withdraw
     * @dev Transfers `amount` of staking tokens back to the user
     */
    function withdraw(uint256 amount) external;

    /**
     * @notice Claims all pending rewards for the caller
     * @dev Transfers all accrued rewards to the caller
     */
    function getReward() external;

    /**
     * @notice Withdraws all staked tokens and claims pending rewards
     * @dev Combines withdraw and getReward operations
     */
    function exit() external;

    /**
     * @notice Returns the balance of staked tokens for the given account
     * @param account The account to get the balance for
     * @return The balance of staked tokens
     */
    function balanceOf(address account) external view returns (uint256);

    /**
     * @notice Calculates the last time reward is applicable for a given rewards token
     * @param _rewardsToken The address of the rewards token
     * @return The timestamp when the reward was last applicable
     */
    function lastTimeRewardApplicable(address _rewardsToken)
        external
        view
        returns (uint256);

    /**
     * @notice Calculates the reward per token for a given rewards token
     * @param _rewardsToken The address of the rewards token
     * @return The reward amount per token
     */
    function rewardPerToken(address _rewardsToken)
        external
        view
        returns (uint256);

    /**
     * @notice Calculates the earned rewards for a given account and rewards token
     * @param account The address of the account
     * @param _rewardsToken The address of the rewards token
     * @return The amount of rewards earned
     */
    function earned(address account, address _rewardsToken)
        external
        view
        returns (uint256);

    /**
     * @notice Calculates the total reward for the duration of a given rewards token
     * @param _rewardsToken The address of the rewards token
     * @return The total reward amount for the duration of a given rewards token
     */
    function getRewardForDuration(address _rewardsToken)
        external
        view
        returns (uint256);

    /**
     * @notice Gets the reward data for a given rewards token
     * @param _rewardsToken The address of the rewards token
     * @return rewardsDistributor The address authorized to distribute rewards
     * @return rewardsDuration The duration of the reward period
     * @return periodFinish The timestamp when rewards finish
     * @return rewardRate The rate of rewards distributed per second
     * @return lastUpdateTime The last time rewards were updated
     * @return rewardPerTokenStored The last calculated reward per token
     */
    function rewardData(address _rewardsToken)
        external
        view
        returns (
            address rewardsDistributor,
            uint256 rewardsDuration,
            uint256 periodFinish,
            uint256 rewardRate,
            uint256 lastUpdateTime,
            uint256 rewardPerTokenStored,
            uint256 rewardResidual
        );

    /**
     * @notice Returns the reward token address at a specific index
     * @param index The index in the reward tokens array
     * @return The address of the reward token at the given index
     */
    function rewardTokens(uint256 index) external view returns (address);

    /**
     * @notice Claims all pending rewards for a specified user
     * @dev Iterates through all reward tokens and transfers any accrued rewards to the user
     * @param _user The address of the user to claim rewards for
     */
    function getRewardForUser(address _user) external;
}