// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.5.17;

/**
 * @dev In order for foundry to properly deploy and link contracts via `vm.getCode`
 * they must be imported in the `src` (or `contracts`, in our case) folder of the project.
 * If we would have this file in the `test` folder, everything builds, but
 * `vm.getCode` will complain that it can't find the artifact.
 */
import "celo/contracts/common/Registry.sol";
import "celo/contracts/common/Freezer.sol";
import "celo/contracts/stability/SortedOracles.sol";
import "test/utils/harnesses/WithThresholdHarness.sol";
import "test/utils/harnesses/WithCooldownHarness.sol";
