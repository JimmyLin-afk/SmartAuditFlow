// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "../IWasabiPerps.sol";
import "./IStakingAccount.sol";

interface IStakingAccountFactory {
    error CallerNotPool();
    error StakingContractNotSetForToken(address _token);
    error StakingAccountNotDeployed(address _user);

    event StakingAccountCreated(address indexed user, address stakingAccount);
    event StakedPosition(
        address indexed user,
        address stakingAccount,
        address stakingContract,
        IStakingAccount.StakingType stakingType,
        uint256 positionId,
        uint256 collateralAmount
    );
    event UnstakedPosition(
        address indexed user,
        address stakingAccount,
        address stakingContract,
        IStakingAccount.StakingType stakingType,
        uint256 positionId,
        uint256 collateralAmount
    );
    event StakingRewardsClaimed(
        address indexed user,
        address stakingAccount,
        address stakingContract,
        IStakingAccount.StakingType stakingType,
        address rewardToken,
        uint256 amount
    );

    /// @notice Stakes the collateral of a position in the Infrared vault via the trader's StakingAccount
    /// @param _position The position to stake
    /// @param _existingPosition The existing position, if editing an existing position
    function stakePosition(IWasabiPerps.Position memory _position, IWasabiPerps.Position memory _existingPosition) external;

    /// @notice Unstakes the collateral of a position, sends it to the pool via the StakingAccount, and claims rewards
    /// @param _position The position to unstake
    /// @param _amount The amount to unstake
    function unstakePosition(IWasabiPerps.Position memory _position, uint256 _amount) external;

    /// @notice Claims the rewards from the Infrared vault
    /// @param _stakingToken The staking token to claim rewards for
    function claimRewards(address _stakingToken) external;

    /// @notice Sets the vault for a staking token
    /// @param _stakingToken The staking token to set the vault for
    /// @param _stakingContract The staking contract to set
    /// @param _stakingType The type of the staking contract
    function setStakingContractForToken(address _stakingToken, address _stakingContract, IStakingAccount.StakingType _stakingType) external;

    /// @notice Upgrades the StakingAccount proxies to a new implementation
    /// @param _newImplementation The new implementation to upgrade to
    function upgradeBeacon(address _newImplementation) external;
}
