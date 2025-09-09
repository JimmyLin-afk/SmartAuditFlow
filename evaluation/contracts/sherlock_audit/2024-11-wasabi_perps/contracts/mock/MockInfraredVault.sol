// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {IInfraredVault, IRewardVault} from "../bera/IInfraredVault.sol";
import {MultiRewards, SafeERC20, ERC20} from "./MultiRewards.sol";

contract MockInfraredVault is MultiRewards, IInfraredVault {
    using SafeERC20 for ERC20;

    error ZeroAddress();
    error ZeroAmount();
    error MaxNumberOfRewards();

    IRewardVault public override rewardsVault;
    address public override infrared;

    uint256 public constant MAX_NUM_REWARD_TOKENS = 10;

    constructor(address _stakingToken, address _rewardsVault) MultiRewards(_stakingToken) {
        rewardsVault = IRewardVault(_rewardsVault);
        infrared = msg.sender;
    }

    /**
     * @notice Transfers to berachain low level module on staking of LP tokens with the vault after transferring tokens in
     * @param amount The amount of staking token transferred in to the contract
     */
    function onStake(uint256 amount) internal override {
        stakingToken.forceApprove(address(rewardsVault), amount);
        rewardsVault.stake(amount);
    }

    /**
     * @notice Redeems from berachain low level module on withdraw of LP tokens from the vault before transferring tokens out
     * @param amount The amount of staking token transferred out of the contract
     */
    function onWithdraw(uint256 amount) internal override {
        rewardsVault.withdraw(amount);
    }

    /**
     * @notice hook called after the reward is claimed to harvest the rewards from the berachain rewards vault
     * @dev does nothing in this mock implementation
     */
    function onReward() internal override {}

    /// @inheritdoc IInfraredVault
    function updateRewardsDuration(
        address _rewardsToken,
        uint256 _rewardsDuration
    ) external {
        if (_rewardsToken == address(0)) revert ZeroAddress();
        if (_rewardsDuration == 0) revert ZeroAmount();
        _setRewardsDuration(_rewardsToken, _rewardsDuration);
    }

    /// @inheritdoc IInfraredVault
    function unpauseStaking() external {
        if (!paused()) return;
        _unpause();
    }

    /// @inheritdoc IInfraredVault
    function pauseStaking() external {
        if (paused()) return;
        _pause();
    }

    /// @inheritdoc IInfraredVault
    function addReward(address _rewardsToken, uint256 _rewardsDuration) external {
        if (_rewardsToken == address(0)) revert ZeroAddress();
        if (_rewardsDuration == 0) revert ZeroAmount();
        if (rewardTokens.length == MAX_NUM_REWARD_TOKENS) revert MaxNumberOfRewards();
        _addReward(_rewardsToken, infrared, _rewardsDuration);
    }

    /// @inheritdoc IInfraredVault
    function removeReward(address _rewardsToken) external {
        if (_rewardsToken == address(0)) revert ZeroAddress();
        _removeReward(_rewardsToken);
    }

    /// @inheritdoc IInfraredVault
    function notifyRewardAmount(address _rewardToken, uint256 _reward) external {
        if (_rewardToken == address(0)) revert ZeroAddress();
        if (_reward == 0) revert ZeroAmount();
        _notifyRewardAmount(_rewardToken, _reward);
    }

    /// @inheritdoc IInfraredVault
    function recoverERC20(address _to, address _token, uint256 _amount) external {
        if (_to == address(0) || _token == address(0)) {
            revert ZeroAddress();
        }
        if (_amount == 0) revert ZeroAmount();
        _recoverERC20(_to, _token, _amount);
    }

    /// @inheritdoc IInfraredVault
    function getAllRewardTokens() external view returns (address[] memory) {
        return rewardTokens;
    }

    /// @inheritdoc IInfraredVault
    function getAllRewardsForUser(address _user)
        external
        view
        returns (UserReward[] memory)
    {
        uint256 len = rewardTokens.length;
        UserReward[] memory tempRewards = new UserReward[](len);
        uint256 count = 0;

        for (uint256 i = 0; i < len; i++) {
            uint256 amount = earned(_user, rewardTokens[i]);
            if (amount > 0) {
                tempRewards[count] =
                    UserReward({token: rewardTokens[i], amount: amount});
                count++;
            }
        }

        // Create a new array with the exact size of non-zero rewards
        UserReward[] memory userRewards = new UserReward[](count);
        for (uint256 j = 0; j < count; j++) {
            userRewards[j] = tempRewards[j];
        }

        return userRewards;
    }
}