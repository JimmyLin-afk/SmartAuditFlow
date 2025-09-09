// SPDX-License-Identifier: GPL-3.0-or-later
// solhint-disable func-name-mixedcase, var-name-mixedcase, state-visibility
// solhint-disable const-name-snakecase, max-states-count, contract-name-camelcase
pragma solidity ^0.8;

import { Test } from "mento-std/Test.sol";

import { IPricingModule } from "contracts/interfaces/IPricingModule.sol";

contract ConstantSumPricingModuleTest is Test {
  IPricingModule constantSum;

  function setUp() public {
    constantSum = IPricingModule(deployCode("ConstantSumPricingModule"));
  }

  /* ---------- getAmountOut ---------- */

  function test_getAmountOut_whenAmountInZero_shouldReturnZero() public view {
    assertEq(constantSum.getAmountOut(10e24, 10e24, 1e23, 0), 0);
  }

  function test_getAmountOut_whenSpreadZero_shouldReturnAmounInInTokenOutValue() public view {
    uint256 tokenInBucketSize = 10e24;
    uint256 tokenOutBucketSize = 20e24;
    uint256 spread = 0;
    uint256 amountIn = 10e18;
    uint256 amountOut = constantSum.getAmountOut(tokenInBucketSize, tokenOutBucketSize, spread, amountIn);
    assertEq(amountOut, amountIn * 2);
  }

  function test_getAmountOut_whenSmallAmounts_shouldReturnCorrectCalculation() public view {
    uint256 amountOut = constantSum.getAmountOut(20e24, 10e24, 0, 1); // ≈ 0.5 Wei rounded down
    assertEq(amountOut, 0);

    amountOut = constantSum.getAmountOut(20e24, 10e24, 0, 2);
    assertEq(amountOut, 1);

    amountOut = constantSum.getAmountOut(10e24, 20e24, 0, 1);
    assertEq(amountOut, 2);
  }

  //Testing concrete Case
  //amountOut = (1 - spread) * amountIn * tokenOutBucketSize) / tokenInBucketSize
  //          = (1-0.1) * 10e18 * 10e24 / 20e24  = 0.9 * 10e18 * 1/2 = 4500000000000000000 Wei
  function test_getAmountOut_whenValidInput_shouldReturnCorrectCalculation() public view {
    uint256 amountOut = constantSum.getAmountOut(20e24, 10e24, 1e23, 10e18);
    assertEq(amountOut, 4500000000000000000);
  }

  /* ---------- getAmountIn ---------- */

  function test_getAmountIn_whenAmountOutZero_shouldReturnZero() public view {
    assertEq(constantSum.getAmountIn(10e24, 10e24, 1e23, 0), 0);
  }

  function test_getAmountIn_whenSpreadIsZero_shouldReturnAmountOutInTokenInValue() public view {
    uint256 tokenInBucketSize = 10e24;
    uint256 tokenOutBucketSize = 20e24;
    uint256 spread = 0;
    uint256 amountOut = 10e18;
    uint256 amountIn = constantSum.getAmountIn(tokenInBucketSize, tokenOutBucketSize, spread, amountOut);
    assertEq(amountIn, amountOut / 2);
  }

  function test_getAmountIn_whenSmallAmounts_shouldReturnCorrectCalculation() public view {
    uint256 amountIn = constantSum.getAmountIn(20e24, 10e24, 0, 1);
    assertEq(amountIn, 2);

    amountIn = constantSum.getAmountIn(10e24, 20e24, 0, 1); // ≈ 0.5 Wei rounded down
    assertEq(amountIn, 0);

    amountIn = constantSum.getAmountIn(10e24, 20e24, 0, 2);
    assertEq(amountIn, 1);
  }

  //Testing concrete Case
  //amountIn = (amountOut  * tokenInBucketSize) / (tokenOutBucketSize * (1 - spread))
  //         = 10e18 * 20e24 / (10e24 * (1 - 0.1))
  //         = 10e18 * 20e24 / (10e24 * 0.9) ≈ 22222222222222222222.22222222222222222
  //         = 22222222222222222222 Wei
  function test_getAmountIn_whenValidInput_shouldReturnCorrectCalculation() public view {
    uint256 amountIn = constantSum.getAmountIn(20e24, 10e24, 1e23, 10e18);
    assertEq(amountIn, 22222222222222222222);
  }

  /* ---------- name ---------- */

  function test_name_shouldReturnCorrectName() public view {
    assertEq(constantSum.name(), "ConstantSum");
  }
}
