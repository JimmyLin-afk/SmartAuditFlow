// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

import { Test } from "mento-std/Test.sol";
import { CELO_REGISTRY_ADDRESS } from "mento-std/Constants.sol";

import { IRegistry } from "celo/contracts/common/interfaces/IRegistry.sol";

interface IRegistryInit {
  function initialize() external;
}

contract WithRegistry is Test {
  IRegistry public registry = IRegistry(CELO_REGISTRY_ADDRESS);

  constructor() {
    deployCodeTo("Registry", abi.encode(true), CELO_REGISTRY_ADDRESS);
    IRegistryInit(CELO_REGISTRY_ADDRESS).initialize();
  }
}
