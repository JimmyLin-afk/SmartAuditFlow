// SPDX-License-Identifier: MIT

pragma solidity 0.8.18;

/// @title Lido Withdrawal Queue ERC721 contract interface
interface ILidoWithdrawalQueueERC721 {
  struct WithdrawalRequestStatus {
    uint256 amountOfStETH;
    uint256 amountOfShares;
    address owner;
    uint256 timestamp;
    bool isFinalized;
    bool isClaimed;
  }

  /// @notice Batch request the _amounts of stETH for withdrawal to the _owner address. For each request, the respective amount of stETH is transferred to this contract address, and an unstETH NFT is minted to the _owner address.
  /// @return requestIds Returns the array of ids for each created request
  function requestWithdrawals(
    uint256[] calldata _amounts,
    address _owner
  ) external returns (uint256[] calldata requestIds);

  /// @dev See {IERC721-safeTransferFrom}.
  function safeTransferFrom(address _from, address _to, uint256 _requestId) external;

  /// @notice Claims the _requestId withdrawal request, sending ether to msg.sender address.
  function claimWithdrawal(uint256 _requestId) external;

  function getWithdrawalStatus(uint256[] calldata _requestIds) external view returns (WithdrawalRequestStatus[] calldata statuses);
}
