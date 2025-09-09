// SPDX-License-Identifier: GPL-3.0-or-later
// solhint-disable gas-custom-errors
pragma solidity 0.8.18;

import { OwnableUpgradeable } from "openzeppelin-contracts-upgradeable/contracts/access/OwnableUpgradeable.sol";
import { MentoToken } from "./MentoToken.sol";

/**
 * @title Emission Contract for Mento Token
 * @author Mento Labs
 * @notice This contract handles the emission of Mento Tokens in an exponentially decaying manner.
 */
contract Emission is OwnableUpgradeable {
  /// @notice Pre-calculated constant = EMISSION_HALF_LIFE / LN2.
  uint256 public constant A = 454968308;

  /// @notice Used to not lose precision in calculations.
  uint256 public constant SCALER = 1e18;

  /// @notice The timestamp when the emission process started.
  uint256 public emissionStartTime;

  /// @notice The MentoToken contract reference.
  MentoToken public mentoToken;

  /// @notice The max amount that will be minted through emission
  uint256 public emissionSupply;

  /// @notice The target address where emitted tokens are sent.
  address public emissionTarget;

  /// @notice The cumulative amount of tokens that have been emitted so far.
  uint256 public totalEmittedAmount;

  event EmissionTargetSet(address newTargetAddress);
  event TokensEmitted(address indexed target, uint256 amount);

  /**
   * @dev Should be called with disable=true in deployments when
   * it's accessed through a Proxy.
   * Call this with disable=false during testing, when used
   * without a proxy.
   * @param disable Set to true to run `_disableInitializers()` inherited from
   * openzeppelin-contracts-upgradeable/Initializable.sol
   */
  constructor(bool disable) {
    if (disable) {
      _disableInitializers();
    }
  }

  /**
   * @notice Initialize the Emission contract.
   * @param mentoToken_ The address of the MentoToken contract.
   * @param emissionTarget_ The address of the emission target.
   * @param emissionSupply_ The total amount of tokens that can be emitted.
   */
  function initialize(address mentoToken_, address emissionTarget_, uint256 emissionSupply_) public initializer {
    emissionStartTime = block.timestamp;
    mentoToken = MentoToken(mentoToken_);
    // slither-disable-next-line missing-zero-check
    emissionTarget = emissionTarget_;
    emissionSupply = emissionSupply_;
    __Ownable_init();
  }

  /**
   * @notice Set the recipient address for the emitted tokens.
   * @param emissionTarget_ Address of the emission target.
   */
  function setEmissionTarget(address emissionTarget_) external onlyOwner {
    // slither-disable-next-line missing-zero-check
    emissionTarget = emissionTarget_;
    emit EmissionTargetSet(emissionTarget_);
  }

  /**
   * @notice Emit tokens based on the exponential decay formula.
   * @return amount The number of tokens emitted.
   */
  function emitTokens() external returns (uint256 amount) {
    amount = calculateEmission();
    require(amount > 0, "Emission: no tokens to emit");

    totalEmittedAmount += amount;

    emit TokensEmitted(emissionTarget, amount);
    mentoToken.mint(emissionTarget, amount);
  }

  /**
   * @dev Calculate the releasable token amount using a predefined formula.
   * The Maclaurin series is used to create a simpler approximation of the exponential decay formula.
   * Original formula: E(t) = supply * exp(-A * t)
   * Approximation: E(t) = supply * (1 - (t / A) + (t^2 / 2A^2) - (t^3 / 6A^3) + (t^4 / 24A^4))
   * where A = HALF_LIFE / ln(2)
   * @dev A 5th term (t^5 / 120A^5) is added to ensure the entire supply is minted within around 31.5 years.
   * @return amount Number of tokens that can be emitted.
   */
  function calculateEmission() public view returns (uint256 amount) {
    uint256 t = (block.timestamp - emissionStartTime);

    // slither-disable-start divide-before-multiply
    uint256 term1 = (SCALER * t) / A;
    uint256 term2 = (term1 * t) / (2 * A);
    uint256 term3 = (term2 * t) / (3 * A);
    uint256 term4 = (term3 * t) / (4 * A);
    uint256 term5 = (term4 * t) / (5 * A);
    // slither-disable-end divide-before-multiply

    uint256 positiveAggregate = SCALER + term2 + term4;
    uint256 negativeAggregate = term1 + term3 + term5;

    // Avoiding underflow in case the scheduled amount is bigger than the total supply
    if (positiveAggregate < negativeAggregate) {
      return emissionSupply - totalEmittedAmount;
    }

    uint256 scheduledRemainingSupply = (emissionSupply * (positiveAggregate - negativeAggregate)) / SCALER;

    amount = emissionSupply - scheduledRemainingSupply - totalEmittedAmount;
  }
}
