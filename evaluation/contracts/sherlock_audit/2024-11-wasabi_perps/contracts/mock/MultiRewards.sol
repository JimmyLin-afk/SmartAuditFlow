// SPDX-License-Identifier: BUSL-1.1
pragma solidity 0.8.26;

import {IMultiRewards} from "../bera/IMultiRewards.sol";

import {Pausable} from "@openzeppelin/contracts/utils/Pausable.sol";
import {ReentrancyGuard} from
    "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/**
 * @title MultiRewards
 * @dev Fork of https://github.com/curvefi/multi-rewards with hooks on stake/withdraw of LP tokens
 */
abstract contract MultiRewards is ReentrancyGuard, Pausable, IMultiRewards {
    using SafeERC20 for ERC20;

    /*//////////////////////////////////////////////////////////////
                                STATE
    //////////////////////////////////////////////////////////////*/

    /**
     * @notice The token that users stake to earn rewards
     * @dev This is the base token that users deposit into the contract
     */
    ERC20 public immutable stakingToken;

    /**
     * @notice Stores reward-related data for each reward token
     * @dev Maps reward token addresses to their Reward struct containing distribution parameters
     */
    mapping(address => Reward) public override rewardData;

    /**
     * @notice Array of all reward token addresses
     * @dev Used to iterate through all reward tokens when updating or claiming rewards
     */
    address[] public rewardTokens;

    /**
     * @notice Tracks the reward per token paid to each user for each reward token
     * @dev Maps user address to reward token address to amount already paid
     * Used to calculate new rewards since last claim
     */
    mapping(address => mapping(address => uint256)) public
        userRewardPerTokenPaid;

    /**
     * @notice Tracks the unclaimed rewards for each user for each reward token
     * @dev Maps user address to reward token address to unclaimed amount
     */
    mapping(address => mapping(address => uint256)) public rewards;

    /**
     * @notice The total amount of staking tokens in the contract
     * @dev Used to calculate rewards per token
     */
    uint256 internal _totalSupply;

    /**
     * @notice Maps user addresses to their staked token balance
     * @dev Internal mapping used to track individual stake amounts
     */
    mapping(address => uint256) internal _balances;

    /*//////////////////////////////////////////////////////////////
                        MODIFIERS
    //////////////////////////////////////////////////////////////*/

    /**
     * @notice Updates the reward for the given account before executing the
     * function body.
     * @param account address The account to update the reward for.
     */
    modifier updateReward(address account) {
        for (uint256 i; i < rewardTokens.length; i++) {
            address token = rewardTokens[i];

            uint256 latestRewardPerToken = rewardPerToken(token);
            rewardData[token].rewardPerTokenStored = latestRewardPerToken;
            rewardData[token].lastUpdateTime = lastTimeRewardApplicable(token);

            if (account != address(0)) {
                rewards[account][token] = earned(account, token);
                userRewardPerTokenPaid[account][token] = latestRewardPerToken;
            }
        }
        _;
    }

    /*//////////////////////////////////////////////////////////////
                        CONSTRUCTOR
    //////////////////////////////////////////////////////////////*/

    /**
     * @notice Constructs the MultiRewards contract.
     * @param _stakingToken address The token that users stake to earn rewards.
     */
    constructor(address _stakingToken) {
        stakingToken = ERC20(_stakingToken);
    }

    /*//////////////////////////////////////////////////////////////
                               READS
    //////////////////////////////////////////////////////////////*/

    /// @inheritdoc IMultiRewards
    function totalSupply() external view returns (uint256) {
        return _totalSupply;
    }

    /// @inheritdoc IMultiRewards
    function balanceOf(address account)
        external
        view
        returns (uint256 _balance)
    {
        return _balances[account];
    }

    /// @inheritdoc IMultiRewards
    function lastTimeRewardApplicable(address _rewardsToken)
        public
        view
        returns (uint256)
    {
        // min value between timestamp and period finish
        uint256 periodFinish = rewardData[_rewardsToken].periodFinish;
        uint256 ts = block.timestamp;
        return ts < periodFinish ? ts : periodFinish;
    }

    /// @inheritdoc IMultiRewards
    function rewardPerToken(address _rewardsToken)
        public
        view
        returns (uint256)
    {
        if (_totalSupply == 0) {
            return rewardData[_rewardsToken].rewardPerTokenStored;
        }
        return rewardData[_rewardsToken].rewardPerTokenStored
            + (
                lastTimeRewardApplicable(_rewardsToken)
                    - rewardData[_rewardsToken].lastUpdateTime
            ) * rewardData[_rewardsToken].rewardRate * 1e18 / _totalSupply;
    }

    /// @inheritdoc IMultiRewards
    function earned(address account, address _rewardsToken)
        public
        view
        returns (uint256)
    {
        return (
            _balances[account]
                * (
                    rewardPerToken(_rewardsToken)
                        - userRewardPerTokenPaid[account][_rewardsToken]
                )
        ) / 1e18 + rewards[account][_rewardsToken];
    }

    /// @inheritdoc IMultiRewards
    function getRewardForDuration(address _rewardsToken)
        external
        view
        returns (uint256)
    {
        return rewardData[_rewardsToken].rewardRate
            * rewardData[_rewardsToken].rewardsDuration;
    }

    /*//////////////////////////////////////////////////////////////
                            WRITES
    //////////////////////////////////////////////////////////////*/

    /// @inheritdoc IMultiRewards
    function stake(uint256 amount)
        external
        nonReentrant
        whenNotPaused
        updateReward(msg.sender)
    {
        require(amount > 0, "Cannot stake 0");
        _totalSupply = _totalSupply + amount;
        _balances[msg.sender] = _balances[msg.sender] + amount;

        // transfer staking token in then hook stake, for hook to have access to collateral
        stakingToken.safeTransferFrom(msg.sender, address(this), amount);
        onStake(amount);
        emit Staked(msg.sender, amount);
    }

    /**
     * @notice Hook called in the stake function after transfering staking token in
     * @param amount The amount of staking token transferred in to the contract
     */
    function onStake(uint256 amount) internal virtual;

    /// @inheritdoc IMultiRewards
    function withdraw(uint256 amount)
        public
        nonReentrant
        updateReward(msg.sender)
    {
        require(amount > 0, "Cannot withdraw 0");
        _totalSupply = _totalSupply - amount;
        _balances[msg.sender] = _balances[msg.sender] - amount;

        // hook withdraw then transfer staking token out
        onWithdraw(amount);
        stakingToken.safeTransfer(msg.sender, amount);
        emit Withdrawn(msg.sender, amount);
    }

    /**
     * @notice Hook called in withdraw function before transferring staking token out
     * @param amount The amount of staking token to be transferred out of the contract
     */
    function onWithdraw(uint256 amount) internal virtual;

    /// @inheritdoc IMultiRewards
    function getRewardForUser(address _user)
        public
        nonReentrant
        updateReward(_user)
    {
        onReward();
        uint256 len = rewardTokens.length;
        for (uint256 i; i < len; i++) {
            address _rewardsToken = rewardTokens[i];
            uint256 reward = rewards[_user][_rewardsToken];
            if (reward > 0) {
                (bool success, bytes memory data) = _rewardsToken.call{
                    gas: 200000
                }(
                    abi.encodeWithSelector(
                        ERC20.transfer.selector, _user, reward
                    )
                );
                if (success && (data.length == 0 || abi.decode(data, (bool)))) {
                    rewards[_user][_rewardsToken] = 0;
                    emit RewardPaid(_user, _rewardsToken, reward);
                } else {
                    continue;
                }
            }
        }
    }

    /**
     * @notice Hook called in getRewardForUser function
     */
    function onReward() internal virtual;

    /// @inheritdoc IMultiRewards
    function getReward() public {
        getRewardForUser(msg.sender);
    }

    /// @inheritdoc IMultiRewards
    function exit() external {
        withdraw(_balances[msg.sender]);
        getReward();
    }

    /*//////////////////////////////////////////////////////////////
                            RESTRICTED
    //////////////////////////////////////////////////////////////*/

    /**
     * @notice Adds a reward token to the contract.
     * @param _rewardsToken       address The address of the reward token.
     * @param _rewardsDistributor address The address of the rewards distributor.
     * @param _rewardsDuration    uint256 The duration of the rewards period.
     */
    function _addReward(
        address _rewardsToken,
        address _rewardsDistributor,
        uint256 _rewardsDuration
    ) internal {
        require(rewardData[_rewardsToken].rewardsDuration == 0);
        rewardTokens.push(_rewardsToken);
        rewardData[_rewardsToken].rewardsDistributor = _rewardsDistributor;
        rewardData[_rewardsToken].rewardsDuration = _rewardsDuration;
        emit RewardStored(_rewardsToken, _rewardsDuration);
    }

    /**
     * @notice Removes a reward token from the contract.
     * @param _rewardsToken address The address of the reward token.
     */
    function _removeReward(address _rewardsToken) internal {
        require(block.timestamp >= rewardData[_rewardsToken].periodFinish);
        // Remove from the array
        for (uint256 i; i < rewardTokens.length; i++) {
            if (rewardTokens[i] == _rewardsToken) {
                rewardTokens[i] = rewardTokens[rewardTokens.length - 1];
                rewardTokens.pop();
                break;
            }
        }

        delete rewardData[_rewardsToken];
        emit RewardRemoved(_rewardsToken);
    }

    /**
     * @notice Notifies the contract that reward tokens is being sent to the contract.
     * @param _rewardsToken address The address of the reward token.
     * @param reward        uint256 The amount of reward tokens is being sent to the contract.
     */
    function _notifyRewardAmount(address _rewardsToken, uint256 reward)
        internal
        updateReward(address(0))
    {
        // handle the transfer of reward tokens via `transferFrom` to reduce the number
        // of transactions required and ensure correctness of the reward amount
        ERC20(_rewardsToken).safeTransferFrom(msg.sender, address(this), reward);
        // add in the prior residual amount and account for new residual
        // @dev residual used to account for precision loss when dividing reward by rewardsDuration
        reward = reward + rewardData[_rewardsToken].rewardResidual;
        rewardData[_rewardsToken].rewardResidual =
            reward % rewardData[_rewardsToken].rewardsDuration;
        reward = reward - rewardData[_rewardsToken].rewardResidual;

        if (block.timestamp >= rewardData[_rewardsToken].periodFinish) {
            rewardData[_rewardsToken].rewardRate =
                reward / rewardData[_rewardsToken].rewardsDuration;
        } else {
            uint256 remaining =
                rewardData[_rewardsToken].periodFinish - block.timestamp;
            uint256 leftover = remaining * rewardData[_rewardsToken].rewardRate;

            // Calculate total and its residual
            uint256 totalAmount =
                reward + leftover + rewardData[_rewardsToken].rewardResidual;
            rewardData[_rewardsToken].rewardResidual =
                totalAmount % rewardData[_rewardsToken].rewardsDuration;

            // Remove residual before setting rate
            totalAmount = totalAmount - rewardData[_rewardsToken].rewardResidual;
            rewardData[_rewardsToken].rewardRate =
                totalAmount / rewardData[_rewardsToken].rewardsDuration;
        }

        rewardData[_rewardsToken].lastUpdateTime = block.timestamp;
        rewardData[_rewardsToken].periodFinish =
            block.timestamp + rewardData[_rewardsToken].rewardsDuration;
        emit RewardAdded(_rewardsToken, reward);
    }

    /**
     * @notice Recovers ERC20 tokens sent to the contract.
     * @dev Added to support recovering LP Rewards from other systems such as BAL to be distributed to holders
     * @param to           address The address to send the tokens to.
     * @param tokenAddress address The address of the token to withdraw.
     * @param tokenAmount  uint256 The amount of tokens to withdraw.
     */
    function _recoverERC20(
        address to,
        address tokenAddress,
        uint256 tokenAmount
    ) internal {
        require(
            rewardData[tokenAddress].lastUpdateTime == 0,
            "Cannot withdraw reward token"
        );
        ERC20(tokenAddress).safeTransfer(to, tokenAmount);
        emit Recovered(tokenAddress, tokenAmount);
    }

    /**
     * @notice Updates the reward duration for a reward token.
     * @param _rewardsToken    address The address of the reward token.
     * @param _rewardsDuration uint256 The new duration of the rewards period.
     */
    function _setRewardsDuration(
        address _rewardsToken,
        uint256 _rewardsDuration
    ) internal {
        require(_rewardsDuration > 0, "Reward duration must be non-zero");

        if (block.timestamp < rewardData[_rewardsToken].periodFinish) {
            uint256 remaining =
                rewardData[_rewardsToken].periodFinish - block.timestamp;
            uint256 leftover = remaining * rewardData[_rewardsToken].rewardRate;

            // Calculate total and its residual
            uint256 totalAmount =
                leftover + rewardData[_rewardsToken].rewardResidual;
            rewardData[_rewardsToken].rewardResidual =
                totalAmount % _rewardsDuration;

            // Remove residual before setting rate
            totalAmount = totalAmount - rewardData[_rewardsToken].rewardResidual;
            rewardData[_rewardsToken].rewardRate =
                totalAmount / _rewardsDuration;
        }

        rewardData[_rewardsToken].lastUpdateTime = block.timestamp;
        rewardData[_rewardsToken].periodFinish =
            block.timestamp + _rewardsDuration;

        rewardData[_rewardsToken].rewardsDuration = _rewardsDuration;
        emit RewardsDurationUpdated(_rewardsToken, _rewardsDuration);
    }
}