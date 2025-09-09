// SPDX-License-Identifier: MIT
pragma solidity 0.8.26;

import { UD60x18, unwrap, convert, exp } from "@prb/math/src/UD60x18.sol";

/**
 * @title LMSR
 * @dev This library implements the Logarithmic Market Scoring Rule (LMSR) mechanism to calculate
 *      odds and costs in a prediction market setting using a fixed-point arithmetic format
 *      (UD60x18) for stable and precise computations.
 *
 * The LMSR model sets the cost of a given state based on the current quantities ("votes") of two
 * opposing positions—like "yes" vs. "no." The relative prices adjust automatically as votes
 * change, ensuring that no participant can gain a risk-free profit through arbitrage.
 *
 * Core formulas:
 *
 *     odds = e^(yes/b) / [ e^(yes/b) + e^(no/b) ]
 *
 *     cost = b * ln(e^(yes/b) + e^(no/b))
 *
 * - b is the liquidity parameter (higher b = more stable prices, but higher capital requirements)
 * - yes and no are the count of "yes" and "no" votes respectively
 *
 * All returned values are UD60x18 fixed-point numbers, i.e., scaled by 1e18. These are dimensionless
 * ratios. To convert them into a currency unit (like wei), you must multiply by a base price or
 * scaling factor outside this library.
 *
 * MATHEMATICAL BOUNDS AND SAFETY:
 * - The exp() function reverts if its input > ~133.084.
 * - Therefore, (votes / b) must not exceed ~133.
 * - In practice: votes ≤ 133 * b
 *
 * Example: If b = 1000, then max safe votes ≈ 133,000 per side.
 *
 * LIQUIDITY PARAMETER (b) GUIDELINES:
 * - Recommended range: ~100-1,000,000
 * - Smaller b: More volatility, lower capital requirements, risk of overflow with fewer votes.
 * - Larger b: More stability, higher capital requirements, safer from overflow.
 */

library LMSR {
  error VotesExceedSafeLimit(uint256 votes, uint256 liquidityParameter, uint256 maxSafe);

  /**
   * @notice Computes the current odds (yes or no) given the market state.
   *
   * The formula:
   *     odds = e^(yes/b) / [ e^(yes/b) + e^(no/b) ]
   *
   * @param yesVotes Total number of "yes" votes currently in the market.
   * @param noVotes Total number of "no" votes currently in the market.
   * @param liquidityParameter The market's liquidity parameter (b).
   * @param isYes If true, returns the odds for the "yes" side; if false, for the "no" side.
   * @return ratio The current odds in a percentage from 0 to 1 (scaled up by 1e18).
   */
  function getOdds(
    uint256 yesVotes,
    uint256 noVotes,
    uint256 liquidityParameter,
    bool isYes
  ) public pure returns (uint256 ratio) {
    // Compute exponentials e^(yes/b) and e^(no/b)
    (UD60x18 yesExp, UD60x18 noExp) = _getExponentials(yesVotes, noVotes, liquidityParameter);

    // sumExp = e^(yes/b) + e^(no/b)
    UD60x18 sumExp = yesExp.add(noExp);

    // priceRatio = e^(yes/b)/(sumExp) if isYes, else e^(no/b)/(sumExp)
    UD60x18 priceRatio = isYes ? yesExp.div(sumExp) : noExp.div(sumExp);

    // Unwrap to get  scaled ratio
    ratio = unwrap(priceRatio);
  }

  /**
   * @notice Computes the incremental cost of changing the market state from
   *         (currentYesVotes, currentNoVotes) to (outcomeYesVotes, outcomeNoVotes).
   *
   * This is computed as:
   *     costDiff = cost(outcomeYesVotes, outcomeNoVotes) - cost(currentYesVotes, currentNoVotes)
   *
   * where:
   *     cost = b * ln(e^(yes/b) + e^(no/b))
   *
   * @param currentYesVotes The current number of "yes" votes.
   * @param currentNoVotes The current number of "no" votes.
   * @param outcomeYesVotes The target number of "yes" votes after the trade.
   * @param outcomeNoVotes The target number of "no" votes after the trade.
   * @param liquidityParameter The market's liquidity parameter (b).
   * @return costDiff The incremental cost difference, as an integral of the odds percentage (scaled up by 1e18).
   */
  function getCost(
    uint256 currentYesVotes,
    uint256 currentNoVotes,
    uint256 outcomeYesVotes,
    uint256 outcomeNoVotes,
    uint256 liquidityParameter
  ) public pure returns (int256 costDiff) {
    uint256 oldCost = _cost(currentYesVotes, currentNoVotes, liquidityParameter);
    uint256 newCost = _cost(outcomeYesVotes, outcomeNoVotes, liquidityParameter);
    costDiff = int256(newCost) - int256(oldCost);
  }

  /**
   * @dev Internal function implementing the LMSR cost formula:
   *
   *     cost = b * ln(e^(yes/b) + e^(no/b))
   *
   * Returns the cost in UD60x18 format (scaled by 1e18).
   *
   * @param yesVotes The number of "yes" votes.
   * @param noVotes The number of "no" votes.
   * @param liquidityParameter The market's liquidity parameter (b).
   * @return costResult The cost value in UD60x18 format (scaled by 1e18).
   */
  function _cost(
    uint256 yesVotes,
    uint256 noVotes,
    uint256 liquidityParameter
  ) public pure returns (uint256 costResult) {
    // Compute e^(yes/b) and e^(no/b)
    (UD60x18 yesExp, UD60x18 noExp) = _getExponentials(yesVotes, noVotes, liquidityParameter);

    // sumExp = e^(yes/b) + e^(no/b)
    UD60x18 sumExp = yesExp.add(noExp);

    // lnVal = ln(e^(yes/b) + e^(no/b))
    UD60x18 lnVal = sumExp.ln();

    // Unwrap lnVal and multiply by b (also in UD60x18) to get cost
    uint256 lnValUnwrapped = unwrap(lnVal);
    costResult = lnValUnwrapped * liquidityParameter;
  }

  /**
   * @dev Computes e^(yes/b) and e^(no/b) using UD60x18 arithmetic. Reverts if
   *      votes exceed the safe limit, which could cause overflow in exp().
   *
   * @param yesVotes The number of "yes" votes.
   * @param noVotes The number of "no" votes.
   * @param liquidityParameter The market's liquidity parameter (b).
   * @return yesExp e^(yes/b) in UD60x18 format.
   * @return noExp e^(no/b) in UD60x18 format.
   *
   * Reverts with VotesExceedSafeLimit if either vote count > 133 * b.
   */
  function _getExponentials(
    uint256 yesVotes,
    uint256 noVotes,
    uint256 liquidityParameter
  ) public pure returns (UD60x18 yesExp, UD60x18 noExp) {
    // maxSafeRatio chosen as 133 since exp input ~133.084 causes overflow
    uint256 maxSafeRatio = 133;

    if (
      yesVotes > maxSafeRatio * liquidityParameter || noVotes > maxSafeRatio * liquidityParameter
    ) {
      revert VotesExceedSafeLimit(
        yesVotes > noVotes ? yesVotes : noVotes,
        liquidityParameter,
        maxSafeRatio * liquidityParameter
      );
    }

    // Convert to UD60x18
    UD60x18 yesUD = convert(yesVotes);
    UD60x18 noUD = convert(noVotes);
    UD60x18 b = convert(liquidityParameter);

    // Compute ratios
    UD60x18 yesRatio = yesUD.div(b);
    UD60x18 noRatio = noUD.div(b);

    // Compute exponentials
    yesExp = exp(yesRatio);
    noExp = exp(noRatio);
  }
}
