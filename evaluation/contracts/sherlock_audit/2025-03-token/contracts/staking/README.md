# Symm Staking

Staking is a core component of the Symmio ecosystem, designed to reward long-term participation and support for the platform. By staking the SYMM token, token holders can directly contribute to the network’s security and liquidity, while earning a portion of the settlement fees generated from derivatives trading. Symmio will use this contract to distribute a portion of protocol fees to Symmio holders. As Symmio is deployed on multiple chains, there might be multiple reward tokens.

## Overview of contract

SymmStaking is an upgradeable, multi-token reward staking contract. It allows users to stake a designated token (SYMM) to earn rewards in multiple whitelisted reward tokens. The contract incorporates security features like access control, reentrancy protection, and pausability.

In the SymmStaking contract, a user begins by depositing SYMM tokens using the `deposit` function, specifying an amount and receiver address, which increases their staked balance and the total supply. They earn rewards over time based on their staked amount and the reward rate of whitelisted tokens, which can be checked using `earned`. To withdraw their staked tokens, users call `withdraw`, specifying an amount and recipient address, reducing their balance. Finally, users can claim accumulated rewards anytime with `claimRewards`, transferring all earned reward tokens to their address, while the contract updates reward states throughout these actions to ensure accurate calculations.

### Staking Process

* **Depositing Tokens:**
  * **Action:** When a user calls the `deposit` function, they specify an amount of SYMM tokens to stake along with a receiver address.
  * **Internal Updates:** Before any token balance is adjusted, the contract calls the internal `_updateRewardsStates` function. This ensures that the rewards accrued so far are properly calculated for the receiver—preventing any dilution or unfairness once the staking balance changes.
  * **Token Transfer & State Change:** The SYMM tokens are then transferred from the user to the contract. As a result, both the total staked supply and the individual balance for the receiver are increased.
  * **Event Logging:** A `Deposit` event is emitted, recording the staker, the amount, and the receiver of the stake.
* **Withdrawing Tokens:**
  * **Action:** To retrieve staked tokens, a user invokes the `withdraw` function, specifying the amount and a recipient address.
  * **Reward Update:** Similar to depositing, the contract first updates the reward state for the user to capture any rewards accumulated up to that moment.
  * **Token Transfer & Validation:** The function checks if the user has sufficient staked balance, then transfers the tokens from the contract back to the user’s chosen address while updating the total supply and the user’s individual balance.
  * **Event Logging:** A `Withdraw` event is emitted to log the withdrawal.

### Rewards Calculation

* **Continuous Accrual:**

  * The contract calculates rewards on a per-token basis using the `rewardPerToken` function. This function computes how many rewards each staked token earns over time, factoring in the elapsed time since the last update and the reward rate for that token.
* **Individual Earnings:**

  * For each user, the `earned` function calculates total rewards by combining:
    * The rewards already accrued (stored in the contract),
    * The product of the user's staked balance and the difference between the current cumulative reward per token and what the user has already been credited for.
* **State Updates:**

  * The internal function `_updateRewardsStates` loops through all whitelisted reward tokens to update:
    * The cumulative reward per token (`perTokenStored`),
    * The `lastUpdated` timestamp,
    * And, if applicable, the individual user's earned rewards and the checkpoint of rewards per token they have already been paid for.

  This ensures that every change in staking (whether deposit or withdrawal) is immediately reflected in the rewards calculations.

### Notifying and Adjusting Rewards

* **Adding Rewards:**
  * New rewards are introduced via the `notifyRewardAmount` function. The caller supplies arrays of reward tokens and corresponding amounts.
  * **Token Verification & Transfer:** For each reward token, the contract checks if it’s whitelisted. If so, it transfers the reward amount from the sender to the contract and updates the pending rewards.
* **Reward Rate Calculation:**
  * The `_addRewardsForToken` function recalculates the reward rate for each token:
    * **Expired Period:** If the previous reward period has ended, the new reward rate is simply the reward amount divided by the default duration (e.g., one week).
    * **Ongoing Period:** If the reward period is still active, any leftover rewards (from the remaining time) are added to the new rewards. The combined amount is then used to set a new reward rate, ensuring a smooth transition without disrupting the rewards already in motion.
* **Duration Management:**
  * Each reward token has an associated duration (set to a default value, e.g., one week). This duration defines the period over which the notified rewards will be distributed.

### Claiming Rewards

* **User Claims:**
  * When users are ready to receive their accumulated rewards, they call the `claimRewards` function.
  * **State Update Before Claiming:** As with deposits and withdrawals, the contract updates the user’s reward state to capture all rewards accrued up to that moment.
  * **Reward Transfer:** For each reward token, the function:
    * Transfers the due reward from the contract to the user,
    * Resets the user’s accrued reward balance for that token,
    * And deducts the claimed amount from the pending rewards.
* **Event Logging:**
  * Every reward distribution is recorded with a `RewardClaimed` event that logs the user, the reward token, and the amount claimed.

### Roles

* DEFAULT_ADMIN_ROLE: Overall contract administration
* REWARD_MANAGER_ROLE: Manages reward tokens and claims
* PAUSER_ROLE: Can pause the contract
* UNPAUSER_ROLE: Can unpause the contract

### Events

* RewardNotified: When new rewards are added
* Deposit: When tokens are staked
* Withdraw: When tokens are withdrawn
* RewardClaimed: When rewards are claimed
* UpdateWhitelist: When token whitelist status changes
* RescueToken: When tokens are rescued by admin
