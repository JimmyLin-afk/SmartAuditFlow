// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;
// solhint-disable max-line-length

import { Emission } from "../Emission.sol";

library EmissionDeployerLib {
  /**
   * @notice Deploys a new Emission contract
   * @return The address of the new Emission contract
   */
  function deploy() external returns (Emission) {
    return new Emission(true);
  }
}
