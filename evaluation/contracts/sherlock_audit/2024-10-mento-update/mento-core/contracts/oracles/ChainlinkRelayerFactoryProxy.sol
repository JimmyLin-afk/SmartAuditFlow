// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8.18;

// solhint-disable max-line-length
import {
  TransparentUpgradeableProxy
} from "openzeppelin-contracts-next/contracts/proxy/transparent/TransparentUpgradeableProxy.sol";

contract ChainlinkRelayerFactoryProxy is TransparentUpgradeableProxy {
  constructor(
    address _logic,
    address admin_,
    bytes memory _data
  ) payable TransparentUpgradeableProxy(_logic, admin_, _data) {}
}
