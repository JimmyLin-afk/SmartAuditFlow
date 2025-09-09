// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;
// solhint-disable max-line-length

import { Locking } from "../locking/Locking.sol";

library LockingDeployerLib {
  /**
   * @notice Deploys a new Locking contract
   * @return The address of the new Locking contract
   */
  function deploy() external returns (Locking) {
    return new Locking();
  }
}
