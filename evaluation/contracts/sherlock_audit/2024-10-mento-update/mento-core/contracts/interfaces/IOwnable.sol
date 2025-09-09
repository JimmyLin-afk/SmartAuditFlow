// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity >0.5.13 <0.9;

interface IOwnable {
  function transferOwnership(address newOwner) external;

  function renounceOwnership() external;

  function owner() external view returns (address);
}
