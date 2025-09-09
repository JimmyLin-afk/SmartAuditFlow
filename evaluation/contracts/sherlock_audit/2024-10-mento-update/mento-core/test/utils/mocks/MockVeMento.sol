// SPDX-License-Identifier: GPL-3.0-or-later

pragma solidity ^0.8.0;

import "openzeppelin-contracts-upgradeable/contracts/token/ERC20/ERC20Upgradeable.sol";

contract MockVeMento is ERC20Upgradeable {
  function mint(address to, uint256 amount) external {
    _mint(to, amount);
  }

  /**
   * @dev Returns the current amount of votes that `account` has.
   */
  function getVotes(address account) external view returns (uint256) {
    return balanceOf(account);
  }

  /**
   * @dev Returns the amount of votes that `account` had
   * at the end of the last period
   */
  function getPastVotes(address account, uint256) external view returns (uint256) {
    return balanceOf(account);
  }

  /**
   * @dev Returns the total supply of votes available
   * at the end of the last period
   */
  function getPastTotalSupply(uint256) external view returns (uint256) {
    return totalSupply();
  }
}
