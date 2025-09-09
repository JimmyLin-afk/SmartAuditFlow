// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;

//solhint-disable-next-line max-line-length
import {
  TransparentUpgradeableProxy
} from "openzeppelin-contracts-next/contracts/proxy/transparent/TransparentUpgradeableProxy.sol";
import { ProxyAdmin } from "openzeppelin-contracts-next/contracts/proxy/transparent/ProxyAdmin.sol";

library ProxyDeployerLib {
  /**
   * @notice Deploys a new ProxyAdmin contract
   * @return admin The address of the new ProxyAdmin contract
   */
  function deployAdmin() external returns (ProxyAdmin admin) {
    admin = new ProxyAdmin();
  }

  /**
   * @notice Deploys a new TransparentUpgradeableProxy contract
   * @param implementation The address of the implementation contract
   * @param admin The address of the admin contract
   * @param initializer The data to be passed to the implementation contract's constructor
   * @return proxy The address of the new TransparentUpgradeableProxy contract
   */
  function deployProxy(
    address implementation,
    address admin,
    bytes calldata initializer
  ) external returns (TransparentUpgradeableProxy proxy) {
    proxy = new TransparentUpgradeableProxy(implementation, admin, initializer);
  }
}
