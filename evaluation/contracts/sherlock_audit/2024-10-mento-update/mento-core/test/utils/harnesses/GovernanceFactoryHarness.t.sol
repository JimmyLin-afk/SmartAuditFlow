// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;
// solhint-disable func-name-mixedcase

import { GovernanceFactory } from "contracts/governance/GovernanceFactory.sol";

contract GovernanceFactoryHarness is GovernanceFactory {
  constructor(address owner_) GovernanceFactory(owner_) {}

  function exposed_addressForNonce(uint256 nonce) external view returns (address) {
    return addressForNonce(nonce);
  }
}
