// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

import { IERC20 } from "./IERC20.sol";

interface IGoldToken is IERC20 {
  function mint(address, uint256) external returns (bool);

  function burn(uint256) external returns (bool);

  /**
   * @notice Transfer token for a specified address
   * @param to The address to transfer to.
   * @param value The amount to be transferred.
   * @param comment The transfer comment.
   * @return True if the transaction succeeds.
   */
  function transferWithComment(address to, uint256 value, string calldata comment) external returns (bool);

  function permit(
    address owner,
    address spender,
    uint256 value,
    uint256 deadline,
    uint8 v,
    bytes32 r,
    bytes32 s
  ) external;

  function initialize(address registryAddress) external;
}
