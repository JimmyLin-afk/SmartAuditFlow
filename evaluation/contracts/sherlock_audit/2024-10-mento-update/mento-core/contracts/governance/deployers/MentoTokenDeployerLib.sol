// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;
// solhint-disable max-line-length

import { MentoToken } from "../MentoToken.sol";

library MentoTokenDeployerLib {
  /**
   * @notice Deploys a new MentoToken contract
   * @param allocationRecipients The addresses of the initial token recipients
   * @param allocationAmounts The percentage of tokens to be allocated to each recipient
   * @param emission The address of the emission contract
   * @param locking The address of the locking contract
   * @return The address of the new MentoToken contract
   */
  function deploy(
    address[] memory allocationRecipients,
    uint256[] memory allocationAmounts,
    address emission,
    address locking
  ) external returns (MentoToken) {
    return new MentoToken(allocationRecipients, allocationAmounts, emission, locking);
  }
}
