// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.5.13;
pragma experimental ABIEncoderV2;

import { SafeMath } from "openzeppelin-solidity/contracts/math/SafeMath.sol";
import { FixidityLib } from "celo/contracts/common/FixidityLib.sol";

import { IPricingModule } from "../interfaces/IPricingModule.sol";

/**
 * @title ConstantSumPricingModule
 * @notice The ConstantSumPricingModule calculates the amount in and the amount out for a constant sum AMM.
 */

contract ConstantSumPricingModule is IPricingModule {
  using SafeMath for uint256;
  using FixidityLib for FixidityLib.Fraction;

  /* ==================== View Functions ==================== */
  /**
   * @notice Calculates the amount of tokens that should be received based on the given parameters
   * @dev amountOut = (1 - spread) * amountIn * tokenOutBucketSize) / tokenInBucketSize
   * @param tokenInBucketSize The bucket size of the token swapped in.
   * @param tokenOutBucketSize The bucket size of the token swapped out.
   * @param spread The spread that is applied to a swap.
   * @param amountIn The amount of tokens in wei that is swapped in.
   * @return amountOut The amount of tokens in wei that should be received.
   */
  function getAmountOut(
    uint256 tokenInBucketSize,
    uint256 tokenOutBucketSize,
    uint256 spread,
    uint256 amountIn
  ) external view returns (uint256 amountOut) {
    if (amountIn == 0) return 0;

    FixidityLib.Fraction memory spreadFraction = FixidityLib.fixed1().subtract(FixidityLib.wrap(spread));
    FixidityLib.Fraction memory netAmountIn = spreadFraction.multiply(FixidityLib.newFixed(amountIn));
    FixidityLib.Fraction memory numerator = netAmountIn.multiply(FixidityLib.newFixed(tokenOutBucketSize));
    FixidityLib.Fraction memory denominator = FixidityLib.newFixed(tokenInBucketSize);

    // Can't use FixidityLib.divide because numerator can easily be greater
    // than maxFixedDivisor.
    // Fortunately, we expect an integer result, so integer division gives us as
    // much precision as we could hope for.
    amountOut = numerator.unwrap().div(denominator.unwrap());
    return amountOut;
  }

  /**
   * @notice Calculates the amount of tokens that should be provided in order to receive the desired amount out.
   * @dev amountIn = (amountOut  * tokenInBucketSize) / (tokenOutBucketSize * (1 - spread))
   * @param tokenInBucketSize The bucket size of the token swapped in.
   * @param tokenOutBucketSize The bucket size of the token swapped out.
   * @param spread The spread that is applied to a swap.
   * @param amountOut The amount of tokens in wei that should be swapped out.
   * @return amountIn The amount of tokens in wei that should be provided.
   */
  function getAmountIn(
    uint256 tokenInBucketSize,
    uint256 tokenOutBucketSize,
    uint256 spread,
    uint256 amountOut
  ) external view returns (uint256 amountIn) {
    if (amountOut == 0) return 0;

    FixidityLib.Fraction memory spreadFraction = FixidityLib.fixed1().subtract(FixidityLib.wrap(spread));
    FixidityLib.Fraction memory numerator = FixidityLib.newFixed(amountOut).multiply(
      FixidityLib.newFixed(tokenInBucketSize)
    );
    FixidityLib.Fraction memory denominator = FixidityLib.newFixed(tokenOutBucketSize).multiply(spreadFraction);

    // Can't use FixidityLib.divide because numerator can easily be greater
    // than maxFixedDivisor.
    // Fortunately, we expect an integer result, so integer division gives us as
    // much precision as we could hope for.
    return numerator.unwrap().div(denominator.unwrap());
  }

  /**
   * @notice Returns the AMM that the IPricingModule implements
   * @return Constant Sum.
   */
  function name() external view returns (string memory) {
    return "ConstantSum";
  }
}
