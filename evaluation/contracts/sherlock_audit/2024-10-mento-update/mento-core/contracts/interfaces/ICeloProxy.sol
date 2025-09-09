// SPDX-License-Identifier: GPL-3.0-or-later
// slither-disable-start naming-convention
pragma solidity >=0.5.17 <0.8.19;

interface ICeloProxy {
  function _getImplementation() external view returns (address);

  function _getOwner() external view returns (address);

  function _setImplementation(address implementation) external;

  function _setOwner(address owner) external;

  function _transferOwnership(address newOwner) external;

  function _setAndInitializeImplementation(address implementation, bytes calldata data) external;
}
// slither-disable-end naming-convention
