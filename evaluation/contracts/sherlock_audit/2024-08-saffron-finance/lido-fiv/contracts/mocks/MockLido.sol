// SPDX-License-Identifier: MIT

pragma solidity 0.8.18;

import "@openzeppelin/contracts/utils/math/Math.sol";

import "../interfaces/ILido.sol";

contract MockLido is ILido {
  using Math for uint256;

  mapping(address => uint256) public sharesBalances;
  uint256 public totalPooledEther = 1000 ether;
  uint256 public totalShares = 1000 ether;

  bool returnZeroForSubmit = false;
  bool returnFalseForApprove = false;
  bool returnFalseForTransfer = false;
  uint256 returnBalance;

  function initialize() external {
    totalPooledEther = 1000 ether;
    totalShares = 1000 ether;
    returnZeroForSubmit = false;
    returnFalseForTransfer = false;
    returnFalseForApprove = false;
  }

  function setReturnZeroForSubmit(bool _returnZero) external {
    returnZeroForSubmit = _returnZero;
  }

  function setReturnFalseForTransfer(bool _return) external {
    returnFalseForTransfer = _return;
  }

  function setReturnFalseForApprove(bool _return) external {
    returnFalseForApprove = _return;
  }

  function setReturnBalance(uint256 _balance) external {
    returnBalance = _balance;
  }

  // simulate staking earnings by increasing balance
  function addStakingEarnings(uint256 amount) external {
    totalPooledEther += amount;
  }

  // calculate the amount of earnings needed to add to the totalPooledEther in order to meet the target ETH balance for the passed in account
  function addStakingEarningsForTargetETH(uint256 targetETHBalance, address user) external {
    uint256 userShares = sharesBalances[user];

    require(userShares > 0 && (targetETHBalance * totalShares) > (userShares * totalPooledEther), "Invalid Shares");

    uint256 earnings = ((targetETHBalance * totalShares) - (userShares * totalPooledEther)) / userShares;
    totalPooledEther += earnings;
  }

  // simulate staking loss by decreasing balance
  function subtractStakingEarnings(uint256 amount) external {
    totalPooledEther -= amount;
  }

  // lido / stETH
  function submit(address _referral) external payable returns (uint256) {
    if (returnZeroForSubmit) {
      return 0;
    }

    uint256 shares = getSharesByPooledEth(msg.value);

    totalShares += shares;
    totalPooledEther += msg.value;
    sharesBalances[msg.sender] += shares;

    return shares;
  }

  function transfer(address recipient, uint256 amount) external returns (bool) {
    if (returnFalseForTransfer) {
      return false;
    }
    return true;
  }

  function approve(address spender, uint256 amount) external returns (bool) {
    if (returnFalseForApprove) {
      return false;
    }
    return true;
  }

  function balanceOf(address account) external view returns (uint256) {
    uint256 pooledETH = getPooledEthByShares(sharesBalances[account]);

    if (pooledETH != 0 && returnBalance != 0) {
      return returnBalance;
    }
    return pooledETH;
  }

  function sharesOf(address account) external view returns (uint256) {
    return sharesBalances[account];
  }

  function getSharesByPooledEth(uint256 _ethAmount) public view returns (uint256) {
    return _ethAmount.mulDiv(totalShares, totalPooledEther);
  }

  function getPooledEthByShares(uint256 _sharesAmount) public view returns (uint256) {
    return _sharesAmount.mulDiv(totalPooledEther, totalShares);
  }

  function updateShares(uint256 shares, uint256 ethAmount, address user) public {
    totalShares -= shares;
    totalPooledEther -= ethAmount;
    sharesBalances[user] -= shares;
  }

  // Function to receive Ether. msg.data must be empty
  receive() external payable {
    // React to receiving ether
  }
}
