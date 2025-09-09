// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity >=0.5.13 <0.8.19;

interface ITransparentProxy {
  function implementation() external view returns (address);

  function changeAdmin(address) external;

  function upgradeTo(address) external;

  function upgradeToAndCall(address, bytes calldata) external payable;
}
