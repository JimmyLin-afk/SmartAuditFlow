// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;
// solhint-disable max-line-length

import { MentoGovernor } from "../MentoGovernor.sol";

library MentoGovernorDeployerLib {
  /**
   * @notice Deploys a new MentoGovernor contract
   * @return The address of the new MentoGovernor contract
   */
  function deploy() external returns (MentoGovernor) {
    return new MentoGovernor();
  }
}
