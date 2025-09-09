// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;
// solhint-disable max-line-length

import { TimelockController } from "../TimelockController.sol";

library TimelockControllerDeployerLib {
  /**
   * @notice Deploys a new TimelockController contract
   * @return The address of the new TimelockController contract
   */
  function deploy() external returns (TimelockController) {
    return new TimelockController();
  }
}
