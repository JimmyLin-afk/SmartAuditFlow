// SPDX-License-Identifier: MIT

pragma solidity 0.8.18;

import "@openzeppelin/contracts/utils/math/Math.sol";

import "../interfaces/ILidoWithdrawalQueueERC721.sol";

import "./MockLido.sol";

contract MockLidoWithdrawalQueue is ILidoWithdrawalQueueERC721 {
  using Math for uint256;

  MockLido public lido;

  uint256 requestId = 1;
  bool returnEmptyRequestIdsForWithdrawal = false;
  bool returnZeroForClaim = false;

  mapping(uint256 => uint256) public withdrawalAmounts;
  mapping(uint256 => ILidoWithdrawalQueueERC721.WithdrawalRequestStatus) public withdrawalStatuses;

  function initialize(address _lido) external {
    requestId = 1;
    returnEmptyRequestIdsForWithdrawal = false;
    returnZeroForClaim = false;
    lido = MockLido(payable(_lido));
  }

  function setReturnEmptyRequestIdsForWithdrawal(bool _return) external {
    returnEmptyRequestIdsForWithdrawal = _return;
  }

  function setReturnZeroForClaim(bool _returnZero) external {
    returnZeroForClaim = _returnZero;
  }

  // withdrawal queue

  function requestWithdrawals(uint256[] calldata _amounts, address _owner) external returns (uint256[] memory ids) {
    if (returnEmptyRequestIdsForWithdrawal) {
      uint256[] memory empty;
      return empty;
    }

    uint256 amount;
    uint256[] memory requestIds = new uint256[](_amounts.length);
    for (uint i = 0; i < _amounts.length; i++) {
      amount += _amounts[i];
      withdrawalAmounts[requestId] = _amounts[i];
      requestIds[i] = requestId;
      requestId += 1;
    }
    uint256 shares = lido.getSharesByPooledEth(amount);

    lido.updateShares((shares), amount, msg.sender);

    return requestIds;
  }

  function getWithdrawalStatus(uint256[] calldata _requestIds) external view returns (ILidoWithdrawalQueueERC721.WithdrawalRequestStatus[] memory statuses) {
    ILidoWithdrawalQueueERC721.WithdrawalRequestStatus[] memory _statuses = new ILidoWithdrawalQueueERC721.WithdrawalRequestStatus[](_requestIds.length);
    for (uint i = 0; i < _requestIds.length; i++) {
      _statuses[i] = withdrawalStatuses[_requestIds[i]];
    }
    return _statuses;
  }

  function safeTransferFrom(address _from, address _to, uint256 _requestId) external {}

  function claimWithdrawalsTo(uint256[] calldata _requestIds, uint256[] calldata _hints, address _recipient) external {}

  function claimWithdrawals(uint256[] calldata _requestIds, uint256[] calldata _hints) external {}

  function claimWithdrawal(uint256 _requestId) external {
    if (returnZeroForClaim) {
      return;
    }
    uint256 withdrawalAmount = withdrawalAmounts[_requestId];
    (bool sent, ) = msg.sender.call{value: withdrawalAmount}("");
    require(sent, "ETF");
  }

  // Function to receive Ether. msg.data must be empty
  receive() external payable {
    // React to receiving ether
  }
}
