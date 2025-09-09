// SPDX-License-Identifier: GPL-3.0-or-later
// solhint-disable func-name-mixedcase, var-name-mixedcase, state-visibility
// solhint-disable const-name-snakecase, max-states-count, contract-name-camelcase
pragma solidity ^0.8;

import { Test } from "mento-std/Test.sol";

import { IPricingModule } from "contracts/interfaces/IPricingModule.sol";

contract ConstantProductPricingModuleTest is Test {
  IPricingModule constantProduct;

  uint256 pc1 = 1 * 1e22;
  uint256 pc5 = 5 * 1e22;
  uint256 pc10 = 1e23;

  function setUp() public {
    vm.warp(24 * 60 * 60);
  }

  // TODO: Add tests that don't rely on the legacy exchange.

  // function test_getAmountOut_compareWithLegacyExchange_t1() public {
  //   uint256 expectedAmountOut = legacyExchange.getAmountOut(1e24, 2e24, pc1, 1e23);
  //   uint256 newAmountOut = constantProduct.getAmountOut(1e24, 2e24, pc1, 1e23);

  //   assertEq(newAmountOut, expectedAmountOut);
  // }

  // function test_getAmountOut_compareWithLegacyExchange_t2() public {
  //   uint256 expectedAmountOut = legacyExchange.getAmountOut(11e24, 23e24, pc5, 3e23);
  //   uint256 newAmountOut = constantProduct.getAmountOut(11e24, 23e24, pc5, 3e23);

  //   assertEq(newAmountOut, expectedAmountOut);
  // }

  // function test_getAmountIn_compareWithLegacyExchange_t1() public {
  //   uint256 expectedAmountIn = legacyExchange.getAmountIn(1e24, 2e24, pc1, 1e23);
  //   uint256 newAmountIn = constantProduct.getAmountIn(1e24, 2e24, pc1, 1e23);

  //   assertEq(newAmountIn, expectedAmountIn);
  // }

  // function test_getAmountIn_compareWithLegacyExchange_t2() public {
  //   uint256 expectedAmountIn = legacyExchange.getAmountIn(11e24, 23e24, pc5, 3e23);
  //   uint256 newAmountIn = constantProduct.getAmountIn(11e24, 23e24, pc5, 3e23);

  //   assertEq(newAmountIn, expectedAmountIn);
  // }
}
