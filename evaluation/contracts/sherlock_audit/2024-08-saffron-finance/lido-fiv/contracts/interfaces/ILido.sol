// SPDX-License-Identifier: MIT

pragma solidity 0.8.18;

/// @title Lido / StETH contract interface
interface ILido {
  /// @notice Send funds to the pool with optional _referral parameter
  /// @dev This function is alternative way to submit funds. Supports optional referral address.
  /// @return Amount of StETH shares generated

  function submit(address _referral) external payable returns (uint256);

  /// @notice Moves `_amount` tokens from the caller's account to the `_recipient` account.
  /// @return a boolean value indicating whether the operation succeeded.
  function transfer(address _recipient, uint256 _amount) external returns (bool);

  /// @notice Sets amount as the allowance of spender over the caller’s StETH tokens.
  /// @return bool indicating whether the operation succeeded
  function approve(address spender, uint256 amount) external returns (bool);

  /// @return uint256 Returns the amount of ether that corresponds to _sharesAmount token shares.
  function getPooledEthByShares(uint256 _sharesAmount) external view returns (uint256);

  /// @return the amount of shares that corresponds to `_ethAmount` protocol-controlled Ether.
  function getSharesByPooledEth(uint256 _ethAmount) external view returns (uint256);

  /// @return uint256 Returns stETH balance of the account
  function balanceOf(address account) external view returns (uint256);

  /// @return uint256 Returns amount of shares owned by account.
  function sharesOf(address _account) external view returns (uint256);
}
