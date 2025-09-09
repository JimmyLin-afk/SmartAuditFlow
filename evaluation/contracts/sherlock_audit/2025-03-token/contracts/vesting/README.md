# Overview of Vesting in Symmio Foundation

The Symmio Foundation governs the distribution and vesting of SYMM tokens to encourage long-term commitment and ecosystem stability. Detailed in the [official documentation](https://docs.symmio.foundation/token-related/tokenomics/vesting-and-early-unlock), the tokenomics ensure fairness by applying uniform vesting rules to all stakeholders, including the team. This document explains the vesting mechanics, the SymmVesting contract, user interactions, and key features.

**Tokenomics Summary**

* **Token Generation Event (TGE)**: 30% of SYMM tokens unlock immediately.
* **Vesting Period**: The remaining 70% vests linearly over 9 months (or 6 months if audits delay deployment by 3 months, starting from contract deployment, e.g., February 23, 2025).
* **Early Unlock Option**: Users can unlock vested tokens early with a 50% penalty—half is received, and half is redistributed (80% to non-unlockers, 20% to ecosystem incentives, subject to DAO vote).
* **Liquidity Incentive**: Users can avoid penalties by providing liquidity to an 80/20 SYMM/USDC pool, receiving vested SYMM LP tokens instead.

# SymmVesting Contract Overview

The SymmVesting contract extends the base Vesting contract, integrating Symmio’s vesting policies with liquidity provisioning capabilities. It uses:

* **Inheritance**: Vesting (itself inheriting from OpenZeppelin’s Initializable, AccessControlEnumerableUpgradeable, PausableUpgradeable, ReentrancyGuardUpgradeable), integrating role-based access, pausability, and non-reentrancy.
* **Libraries**: VestingPlanOps for vesting plan calculations and updates.
* **Interfaces**: IPool and IRouter for liquidity interactions, alongside IERC20 for token operations.

Key features include:

* **Vesting Management**: Tracks SYMM and SYMM LP token vesting plans per user, allowing claims of unlocked tokens and penalized early claims of locked tokens.
* **Liquidity Addition**: Enables converting locked SYMM into SYMM LP tokens by pairing with USDC in a pool, with LP tokens re-locked in vesting.
* **Penalty Redistribution**: Implements the 50% early unlock fee, sent to a designated receiver.

## **User Flow**

This section outlines how users interact with the SymmVesting contract, from setup to token claims and liquidity provision.

### Vesting Plan Setup (Admin Only)

* Function: setupVestingPlans(address token, uint256 startTime, uint256 endTime, address[] users, uint256[] amounts)
* Mechanics: Called by SETTER_ROLE to assign SYMM vesting plans. For each user, sets amount, startTime, and endTime via VestingPlanOps.setup.
* Flow:

  * 30% of SYMM is assumed distributed at TGE (off-chain or via separate mechanism).
  * 70% (e.g., 700,000 SYMM) vests linearly from startTime (e.g., February 23, 2025) to endTime (e.g., August 23, 2025, for 6 months).

### Claiming Unlocked Tokens

* Function: claimUnlockedToken(address token)
* Mechanics:

  * Calculates claimable amount via VestingPlanOps.claimable (linear vesting: (amount * elapsed) / duration - claimedAmount).
  * Transfers tokens (e.g., SYMM) to user.
* Flow:

  * After 1 month (e.g., March 23, 2025), 1/6 of 700,000 SYMM (~116,667) unlocks.
  * User claims this, updating claimedAmount.

### Early Unlock of Locked Tokens

* Function: claimLockedToken(address token, uint256 amount)
* Mechanics:

  * Claims unlocked tokens first via _claimUnlockedToken.
  * Reduces lockedAmount by amount, applies 50% penalty (e.g., 100,000 SYMM claimed = 50,000 to user, 50,000 to lockedClaimPenaltyReceiver).
* Flow:

  * User with 583,333 locked SYMM after 1 month claims 200,000 early.
  * Receives 100,000 SYMM; 100,000 SYMM goes to the lockedClaimPenaltyReceiver.

### Adding Liquidity

* Function: addLiquidity(uint256 amount, uint256 minLpAmount)
* Mechanics:

  * Claims unlocked SYMM first.
  * Reduces SYMM lockedAmount by amount.
  * Calculates required USDC via neededUSDCForLiquidity (proportional to pool balances).
  * Pulls USDC from user, approves VAULT, and calls ROUTER.addLiquidityProportional.
  * Locks received SYMM LP tokens in vesting.
* Flow:

  * User supplies 200,000 SYMM and enough USDC regarding the current state of balancer pool.
  * Receives SYMM LP tokens, locked in vestingPlans[SYMM_LP].

## **Roles**

* **DEFAULT_ADMIN_ROLE**: Overall contract administration, manages all roles.
* **SETTER_ROLE**: Configures vesting plans (**setupVestingPlans**, **resetVestingPlans**).
* **OPERATOR_ROLE**: Executes claims on behalf of users (e.g., **claimUnlockedTokenFor**).
* **PAUSER_ROLE**: Can pause the contract to halt state changes.
* **UNPAUSER_ROLE**: Can unpause the contract to resume operations.

## Events

* **VestingPlanSetup**: When vesting plans are configured for users.
* **VestingPlanReset**: When vesting amounts are adjusted (e.g., post-liquidity).
* **UnlockedTokenClaimed**: When vested tokens are claimed.
* **LockedTokenClaimed**: When locked tokens are claimed early with penalty.
* **LiquidityAdded**: When SYMM is converted to SYMM LP via liquidity provision.
