// SPDX-License-Identifier: MIT

pragma solidity >=0.8.18;

import { AccessControlEnumerableUpgradeable } from "@openzeppelin/contracts-upgradeable/access/extensions/AccessControlEnumerableUpgradeable.sol";
import { Initializable } from "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import { ReentrancyGuardUpgradeable } from "@openzeppelin/contracts-upgradeable/utils/ReentrancyGuardUpgradeable.sol";
import { PausableUpgradeable } from "@openzeppelin/contracts-upgradeable/utils/PausableUpgradeable.sol";
import { IERC20 } from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import { SafeERC20 } from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import { IERC20Metadata } from "@openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol";

/**
 * @title SymmStaking
 * @notice An upgradeable staking contract that supports multiple reward tokens.
 * @dev This contract is designed to be used with the Transparent Upgradeable Proxy pattern.
 */
contract SymmStaking is Initializable, AccessControlEnumerableUpgradeable, ReentrancyGuardUpgradeable, PausableUpgradeable {
	using SafeERC20 for IERC20;

	//--------------------------------------------------------------------------
	// Constants
	//--------------------------------------------------------------------------

	uint256 public constant DEFAULT_REWARDS_DURATION = 1 weeks;
	uint256 public constant STANDARD_DECIMALS = 18;

	bytes32 public constant REWARD_MANAGER_ROLE = keccak256("REWARD_MANAGER_ROLE");
	bytes32 public constant REWARD_NOTIFIER_ROLE = keccak256("REWARD_NOTIFIER_ROLE");
	bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
	bytes32 public constant UNPAUSER_ROLE = keccak256("UNPAUSER_ROLE");

	//--------------------------------------------------------------------------
	// Errors
	//--------------------------------------------------------------------------

	/// @notice Thrown when the staked or withdrawn amount is zero.
	error ZeroAmount();

	/// @notice Thrown when the staked for zero address.
	error ZeroAddress();

	/// @notice Thrown when the user does not have enough staked balance.
	/// @param available The available staked balance.
	/// @param required The required amount.
	error InsufficientBalance(uint256 available, uint256 required);

	/// @notice Thrown when a token is not whitelisted for rewards.
	/// @param token The token address.
	error TokenNotWhitelisted(address token);

	/// @notice Thrown when the two arrays passed as parameters have different lengths.
	error ArraysMismatched();

	/// @notice Thrown when the whitelist status is already set.
	/// @param token The token address.
	error TokenAlreadyAdded(address token);

	/// @notice Thrown when the token decimals are invalid.
	error InvalidTokenDecimals(address token, uint8 decimals);

	/// @notice Thrown when failed to read token decimals.
	error FailedToReadDecimals(address token);

	//--------------------------------------------------------------------------
	// Events
	//--------------------------------------------------------------------------

	/**
	 * @notice Emitted when rewards are added.
	 * @param rewardsTokens Array of reward token addresses.
	 * @param rewards Array of reward amounts.
	 * @param newRates Array of new rates.
	 */
	event RewardNotified(address[] rewardsTokens, uint256[] rewards, uint256[] newRates);

	/**
	 * @notice Emitted when a deposit is made.
	 * @param sender The address initiating the deposit.
	 * @param amount The staked amount.
	 * @param receiver The address that receives the staking balance.
	 */
	event Deposit(address indexed sender, uint256 amount, address indexed receiver);

	/**
	 * @notice Emitted when a withdrawal is made.
	 * @param sender The address initiating the withdrawal.
	 * @param amount The withdrawn amount.
	 * @param to The address receiving the tokens.
	 */
	event Withdraw(address indexed sender, uint256 amount, address indexed to);

	/**
	 * @notice Emitted when a reward is paid.
	 * @param user The user receiving the reward.
	 * @param rewardsToken The token in which the reward is paid.
	 * @param reward The amount of reward paid.
	 */
	event RewardClaimed(address indexed user, address indexed rewardsToken, uint256 reward);

	/**
	 * @notice Emitted when a token is added as reward token.
	 * @param token The token address.
	 */
	event AddRewardToken(address indexed token);

	/**
	 * @notice Emitted when admin rescue tokens.
	 * @param token the token address.
	 * @param amount the amount to be rescued.
	 */
	event RescueToken(address token, uint256 amount, address receiver);

	//--------------------------------------------------------------------------
	// Structs
	//--------------------------------------------------------------------------

	struct TokenRewardState {
		uint256 duration;
		uint256 periodFinish;
		uint256 rate;
		uint256 lastUpdated;
		uint256 perTokenStored;
	}

	//--------------------------------------------------------------------------
	// State Variables
	//--------------------------------------------------------------------------
	address public stakingToken;

	uint256 public totalSupply;
	mapping(address => uint256) public balanceOf;

	// Mapping from reward token to reward state.
	mapping(address => TokenRewardState) public rewardState;
	// Array of reward tokens.
	address[] public rewardTokens;
	// Mapping to track if a token is whitelisted for rewards.
	mapping(address => bool) public isRewardToken;
	// Mapping to track token decimals
	mapping(address => uint8) public tokenDecimals;

	// Mapping from user => reward token => user paid reward per token.
	mapping(address => mapping(address => uint256)) public userRewardPerTokenPaid;
	// Mapping from user => reward token => reward amount.
	mapping(address => mapping(address => uint256)) public rewards;

	//--------------------------------------------------------------------------
	// Initialization
	//--------------------------------------------------------------------------

	/**
	 * @notice Initializes the staking contract.
	 * @param admin The admin of the contract.
	 */
	function initialize(address admin, address _stakingToken) external initializer {
		__AccessControlEnumerable_init();
		__ReentrancyGuard_init();
		__Pausable_init();

		if (admin == address(0) || _stakingToken == address(0)) revert ZeroAddress();

		stakingToken = _stakingToken;

		_grantRole(DEFAULT_ADMIN_ROLE, admin);
		_grantRole(REWARD_MANAGER_ROLE, admin);
		_grantRole(PAUSER_ROLE, admin);
		_grantRole(UNPAUSER_ROLE, admin);
		_grantRole(REWARD_NOTIFIER_ROLE, admin);
	}

	//--------------------------------------------------------------------------
	// Views
	//--------------------------------------------------------------------------

	/**
	 * @notice Returns the number of reward tokens.
	 * @return The length of the rewardTokens array.
	 */
	function rewardTokensCount() external view returns (uint256) {
		return rewardTokens.length;
	}

	/**
	 * @notice Returns the last applicable time for rewards.
	 * @param _rewardsToken The reward token address.
	 * @return The last time at which rewards are applicable.
	 */
	function lastTimeRewardApplicable(address _rewardsToken) public view returns (uint256) {
		return block.timestamp < rewardState[_rewardsToken].periodFinish ? block.timestamp : rewardState[_rewardsToken].periodFinish;
	}

	/**
	 * @notice Calculate the scaling factor for a token based on its decimals
	 * @param _token The token address
	 * @return The scaling factor to use for this token
	 */
	function getScalingFactor(address _token) public view returns (uint256) {
		uint8 decimals = tokenDecimals[_token];
		// Default to 18 decimals (no scaling) if decimals not set
		if (decimals == 0) return 1;
		// Only apply scaling if decimals are less than 18
		if (decimals >= STANDARD_DECIMALS) return 1;
		// Calculate dynamic scaling factor: 10^(18 - token_decimals)
		return 10 ** (STANDARD_DECIMALS - decimals);
	}

	/**
	 * @notice Calculates the reward per token for a given reward token.
	 * @param _rewardsToken The reward token address.
	 * @return The reward per token.
	 */
	function rewardPerToken(address _rewardsToken) public view returns (uint256) {
		if (totalSupply == 0) {
			return rewardState[_rewardsToken].perTokenStored;
		}

		uint256 scalingFactor = getScalingFactor(_rewardsToken);

		return
			rewardState[_rewardsToken].perTokenStored +
			(((lastTimeRewardApplicable(_rewardsToken) - rewardState[_rewardsToken].lastUpdated) *
				rewardState[_rewardsToken].rate *
				1e18 *
				scalingFactor) / totalSupply);
	}

	/**
	 * @notice Calculates the earned rewards for an account and a specific reward token.
	 * @param account The user address.
	 * @param _rewardsToken The reward token address.
	 * @return The amount of earned rewards.
	 */
	function _earned(address account, address _rewardsToken) internal view returns (uint256) {
		return
			((balanceOf[account] * (rewardPerToken(_rewardsToken) - userRewardPerTokenPaid[account][_rewardsToken])) / 1e18) +
			rewards[account][_rewardsToken];
	}

	/**
	 * @notice Calculates the earned rewards for an account and a specific reward token.
	 * @param account The user address.
	 * @param _rewardsToken The reward token address.
	 * @return The amount of earned rewards.
	 */
	function earned(address account, address _rewardsToken) external view returns (uint256) {
		return _earned(account, _rewardsToken) / getScalingFactor(_rewardsToken);
	}

	/**
	 * @notice Returns the reward amount for the entire reward duration.
	 * @param _rewardsToken The reward token address.
	 * @return The reward amount for the reward duration.
	 */
	function getFullPeriodReward(address _rewardsToken) external view returns (uint256) {
		return rewardState[_rewardsToken].rate * rewardState[_rewardsToken].duration;
	}

	//--------------------------------------------------------------------------
	// Mutative Functions
	//--------------------------------------------------------------------------

	/**
	 * @notice Deposits SYMM tokens for staking on behalf of a receiver.
	 * @param amount The amount of SYMM tokens to deposit.
	 * @param receiver The address receiving the staking balance.
	 */
	function deposit(uint256 amount, address receiver) external nonReentrant whenNotPaused {
		_updateRewardsStates(receiver);

		if (amount == 0) revert ZeroAmount();
		if (receiver == address(0)) revert ZeroAddress();
		IERC20(stakingToken).safeTransferFrom(msg.sender, address(this), amount);
		totalSupply += amount;
		balanceOf[receiver] += amount;
		emit Deposit(msg.sender, amount, receiver);
	}

	/**
	 * @notice Withdraws staked SYMM tokens.
	 * @param amount The amount of tokens to withdraw.
	 * @param to The address receiving the tokens.
	 */
	function withdraw(uint256 amount, address to) external nonReentrant whenNotPaused {
		_updateRewardsStates(msg.sender);

		if (amount == 0) revert ZeroAmount();
		if (to == address(0)) revert ZeroAddress();
		if (amount > balanceOf[msg.sender]) revert InsufficientBalance(balanceOf[msg.sender], amount);
		IERC20(stakingToken).safeTransfer(to, amount);
		totalSupply -= amount;
		balanceOf[msg.sender] -= amount;
		emit Withdraw(msg.sender, amount, to);
	}

	/**
	 * @notice Claims all earned rewards for the caller.
	 */
	function claimRewards() external nonReentrant whenNotPaused {
		_updateRewardsStates(msg.sender);
		_claimRewardsFor(msg.sender);
	}

	/**
	 * @notice Notifies the contract about new reward amounts.
	 * @param tokens Array of reward token addresses.
	 * @param amounts Array of reward amounts corresponding to each token.
	 */
	function notifyRewardAmount(
		address[] calldata tokens,
		uint256[] calldata amounts
	) external nonReentrant whenNotPaused onlyRole(REWARD_NOTIFIER_ROLE) {
		_updateRewardsStates(address(0));
		if (tokens.length != amounts.length) revert ArraysMismatched();

		uint256 len = tokens.length;

		uint256[] memory newRates = new uint256[](len);

		for (uint256 i = 0; i < len; i++) {
			address token = tokens[i];
			uint256 amount = amounts[i];

			if (amount == 0) continue;
			if (!isRewardToken[token]) revert TokenNotWhitelisted(token);

			IERC20(token).safeTransferFrom(msg.sender, address(this), amount);
			newRates[i] = _addRewardsForToken(token, amount);
		}
		emit RewardNotified(tokens, amounts, newRates);
	}

	//--------------------------------------------------------------------------
	// Restricted Functions
	//--------------------------------------------------------------------------

	/**
	 * @notice Allows admin to claim rewards on behalf of a user.
	 * @param user The user address for which to claim rewards.
	 */
	function claimFor(address user) external nonReentrant onlyRole(REWARD_MANAGER_ROLE) whenNotPaused {
		_updateRewardsStates(user);
		_claimRewardsFor(user);
	}

	/**
	 * @notice Add a reward token.
	 * @param token The token address.
	 */
	function addRewardToken(address token) external onlyRole(REWARD_MANAGER_ROLE) {
		_updateRewardsStates(address(0));

		if (token == address(0)) revert ZeroAddress();
		if (isRewardToken[token]) revert TokenAlreadyAdded(token);

		isRewardToken[token] = true;

		rewardTokens.push(token);
		rewardState[token].duration = DEFAULT_REWARDS_DURATION;

		// Read and set token decimals
		try IERC20Metadata(token).decimals() returns (uint8 decimals) {
			if (decimals == 0) revert InvalidTokenDecimals(token, decimals);
			tokenDecimals[token] = decimals;
		} catch {
			// Revert if the decimals function call fails
			revert FailedToReadDecimals(token);
		}

		emit AddRewardToken(token);
	}

	/**
	 * @notice Withdraw specific amount of token.
	 * @param token The token address.
	 * @param amount The amount.
	 * @param receiver The address of receiver
	 */
	function rescueTokens(address token, uint256 amount, address receiver) external nonReentrant onlyRole(DEFAULT_ADMIN_ROLE) {
		IERC20(token).safeTransfer(receiver, amount);
		emit RescueToken(token, amount, receiver);
	}

	/**
	 * @notice Pauses contract operations.
	 */
	function pause() external onlyRole(PAUSER_ROLE) {
		_pause();
	}

	/**
	 * @notice Unpauses contract operations.
	 */
	function unpause() external onlyRole(UNPAUSER_ROLE) {
		_unpause();
	}

	//--------------------------------------------------------------------------
	// Internal Functions
	//--------------------------------------------------------------------------

	function _addRewardsForToken(address token, uint256 amount) internal returns (uint256) {
		TokenRewardState storage state = rewardState[token];

		if (block.timestamp >= state.periodFinish) {
			state.rate = amount / state.duration;
		} else {
			uint256 remaining = state.periodFinish - block.timestamp;
			uint256 leftover = remaining * state.rate;
			state.rate = (amount + leftover) / state.duration;
		}

		state.lastUpdated = block.timestamp;
		state.periodFinish = block.timestamp + state.duration;
		return state.rate;
	}

	/**
	 * @notice Internal function to claim rewards for a given user.
	 * Assumes updateRewards(user) has already been called.
	 */
	function _claimRewardsFor(address user) internal {
		uint256 length = rewardTokens.length;
		for (uint256 i = 0; i < length; ) {
			address token = rewardTokens[i];
			uint256 reward = rewards[user][token];
			if (reward > 0) {
				// Apply reverse scaling for tokens with non-standard decimals
				uint256 scalingFactor = getScalingFactor(token);
				if (scalingFactor > 1) {
					// Divide by scaling factor to get the actual amount to transfer
					reward = reward / scalingFactor;
				}
				rewards[user][token] = 0;
				IERC20(token).safeTransfer(user, reward);
				emit RewardClaimed(user, token, reward);
			}
			unchecked {
				++i;
			}
		}
	}

	/**
	 * @dev Updates the rewards for an account for all reward tokens.
	 * @param account The account to update.
	 */
	function _updateRewardsStates(address account) internal {
		uint256 length = rewardTokens.length;
		for (uint256 i = 0; i < length; ) {
			address token = rewardTokens[i];
			TokenRewardState storage state = rewardState[token];

			state.perTokenStored = rewardPerToken(token);
			state.lastUpdated = lastTimeRewardApplicable(token);

			if (account != address(0)) {
				rewards[account][token] = _earned(account, token);
				userRewardPerTokenPaid[account][token] = state.perTokenStored;
			}
			unchecked {
				++i;
			}
		}
	}
}
