
# Saffron Finance contest details

- Join [Sherlock Discord](https://discord.gg/MABEWyASkp)
- Submit findings using the issue page in your private contest repo (label issues as med or high)
- [Read for more details](https://docs.sherlock.xyz/audits/watsons)

# Q&A

### Q: On what chains are the smart contracts going to be deployed?
Ethereum
___

### Q: If you are integrating tokens, are you allowing only whitelisted tokens to work with the codebase or any complying with the standard? Are they assumed to have certain properties, e.g. be non-reentrant? Are there any types of [weird tokens](https://github.com/d-xo/weird-erc20) you want to integrate?
The contract uses stETH to calculate most operations, but all users can only deposit ETH and receive withdrawals in ETH. The contract doesn't have its tokens.

The specification of stETH is on the official site of Lido Liquid Staking protocol. The contract uses standard stETH without any modifications.
___

### Q: Are there any limitations on values set by admins (or other roles) in the codebase, including restrictions on array lengths?
There are some parameters’ limitations in the factory and contract constructors.
This contract has a web interface with additional constraints on all parameters (fixedSideCapacity, duration, and variableSideCapacity) used for the function createVault in VaultFactory. Those constraints ensure that variable and fixed users receive income from their deposits. The contract itself doesn't constrain these parameters by design.

The contract contains arrays with undefined lengths that are iterated by their length. Specifically, there is an array with a length equal to the number of variable users with a minimal deposit of 0.01 ETH. It is unlikely that the number of variable users will exceed the safe limit for array length.

The contract indirectly constrains the number of fixed users (maximum 20) due to the minimumFixedDepositBps. 

There is an array with an undefined length for fixed users withdrawal. It is the array fixedOngoingWithdrawalUsers. In line 577 (and in line 670), there is iteration through this array with undefined length. A new element is pushed into this array on each valid request for withdrawal from a fixed user (and only before the vault ends). A fixed user can have only one such element in this array until they finalize their withdrawal. Fixed users can't make additional withdrawal requests after finalizing, as they can only withdraw their entire deposit or nothing.
This means that, while it is an array with an undefined length, the code logic limits its length to a maximum of N elements if there are N fixed users. However, because of the minimumFixedDepoistBps, N <= 20.

There is also a second type of array with undefined length—withdrawal requests (vaultEndedWithdrawalRequestIds, variableToVaultOngoingWithdrawalRequestIds, fixedToVaultNotStartedWithdrawalRequestIds, fixedToVaultOngoingWithdrawalRequestIds, variableToVaultOngoingWithdrawalRequestIds). These are due to limits in the Lido Staking Withdrawal queue. Withdrawal is limited because there is a maximum stETH withdrawal amount of 1000 stETH. This means that if a withdrawal exceeds 1000 stETH, it is split into multiple withdrawal requests and placed in these arrays with undefined length. This happens in the function calculateWithdrawals. Given this situation, we have set the fixedSideCapacity to 100,000 ETH. This ensures that any transaction involving withdrawals from the Lido Liquid Staking protocol will be sufficiently sized to fit within a single Ethereum block.

___

### Q: Are there any limitations on values set by admins (or other roles) in protocols you integrate with, including restrictions on array lengths?
The contract is integrated with Lido Liquid Staking protocol functionality. It has restrictions on the minimum/maximum deposit amount to the Lido Liquid Staking protocol and no restrictions on array lengths.
___

### Q: For permissioned functions, please list all checks and requirements that will be made before calling the function.
The only functions that need permissions are functions inside VaultFactory for changing parameters set during VaultFactory creation. The function checks that the Factory owner calls it. There are no other checks.
___

### Q: Is the codebase expected to comply with any EIPs? Can there be/are there any deviations from the specification?
The contract must strictly complies with all current final EIPs.
___

### Q: Are there any off-chain mechanisms for the protocol (keeper bots, arbitrage bots, etc.)? We assume they won't misbehave, delay, or go offline unless specified otherwise.
No
___

### Q: If the codebase is to be deployed on an L2, what should be the behavior of the protocol in case of sequencer issues (if applicable)? Should Sherlock assume that the Sequencer won't misbehave, including going offline?
This contract works explicitly only on the main Ethereum blockchain due to the Lido Liquid Staking protocol being native to the Ethereum mainnet.
While it is possible to modify a contract to work on L2, it is outside the current plan for this contract.

___

### Q: What properties/invariants do you want to hold even if breaking them has a low/unknown impact?
Yes. The main assumptions described in point 16, should hold in any situation.
___

### Q: Please discuss any design choices you made.
We address some of our design choices in point 16.

___

### Q: Please list any known issues and explicitly state the acceptable risks for each known issue.
The Lido Liquid Staking protocol can experience slashing incidents (such as this https://blog.lido.fi/post-mortem-launchnodes-slashing-incident/). These incidents will decrease income from deposits to the Lido Liquid Staking protocol and could decrease the stETH balance. The contract must be operational after it, but it is acceptable for users to lose part of their income/deposit (for example, a fixed user receives less at the end of the Vault than he deposited at the start).
We assume that fixed and variable users understand the logic of the contract and this system. It is acceptable to create contracts where, for example, Fixed capacity is much smaller than Variable Capacity. Contracts with those parameters will never be profitable for Variable users, and we assume no one will deposit in them. However, the contract can be part of a more sophisticated instrument between parties, for example, with a fixed interest rate offset by other parameters determined by the vault creator. We address this issue by providing warnings in our UIs that a specific vault configuration has a high probability of capital loss. 
ETH/stETH ratio is assumed to always equal 1:1 and any issue related to the ETH/stETH ratio != 1:1 is considered invalid, if the Lido Staking protocol returns ETH at a different ratio, contract will distribute ETH based on the percentage of the claim from the initial deposit.
___

### Q: We will report issues where the core protocol functionality is inaccessible for at least 7 days. Would you like to override this value?
No, we are OK with this.
___

### Q: Please list any relevant protocol resources.
There is no whitepaper for this project at the moment. Point 16 describes a short summary of the contract's main points.

Readme
https://github.com/saffron-finance/lido-fiv/blob/main/README.md
List of error messages and their meanings
https://github.com/saffron-finance/lido-fiv/blob/main/revert.md

Lido Liquid Staking protocol documentation
https://docs.lido.fi/guides/lido-tokens-integration-guide/

Official Lido Liquid Staking protocol Apr chart. This is mostly financial information for understanding the fact that the amount of stETH on the contract balance will always increase.
https://dune.com/queries/570874/1464690

This auto-generated document describes all functions and parameters inside the contract.
https://github.com/saffron-finance/lido-fiv/blob/main/docs/index.md

___

### Q: Additional audit information.
Point 16:

This contract is a reverse zero coupon swap between multiple parties on each side of the contract. The fixed side receives upfront an amount equal to a fixed rate of return on deposit paid by the variable side. On the other hand, the variable side receives the yield generated by the fixed side capital deployed to an underlying asset during the contract's validity. At the end of the contract, the fixed side will receive its principal back.

An equivalent way to understand this protocol is for fixed users to deploy X ETH to receive a fixed amount of premium Y ETH when the vault starts. This value is calculated based on the difference between fixedCapacity and variableCapacity set at the Vault creation. At the end of the vault, he can withdraw the X ETH that he deposited at the start of the vault.

When the contract starts, those X ETH are converted to X' stETH using the Lido Liquid Staking protocol. Consequently, the contract receives X stETH equivalent to Z Saffron Lido Vault stakes (a contract’s internal parameter). During the vault lifetime, the number of stakes stays the same (If no one withdraws). Still, the number of stETH increases each day due to the rebase mechanism from Lido Liquid Staking protocol, the underlying asset in the contract. 

The user on the variable side deploys Y ETH, and in compensation, they receive the acquired stETH from the principal X stETH from the fixed side during the contract's lifetime. It is important to notice that the variable side users do not receive their initial Y deposit back at the end of the Vault.

When both the fixed and variable side withdraws from Saffron Lido Vault, all stETH is converted 1:1 into ETH, and the contract always pays the users in ETH.

If there are multiple fixed-side users, the Y ETH premium is split among the fixed-side users proportionally to the amount of capital they deploy relative to all the capital deployed on the fixed side. Likewise, when there are multiple variable-side users, they each receive a portion of the Saffron Lido Vault yield based on the amount of capital deployed compared to the sum of all capital deployed on the variable side.

A contract starts only when users fill both fixed and variable capacity.

The contract ends after a "duration" of seconds after it starts, and one of the users triggers a withdrawal function. Technically, until someone calls a withdrawal function, the contract will never know it has ended.

Users on both fixed and variable sides can withdraw their initial capital before the Vault starts.

After the vault starts but before it ends, users from the fixed side can withdraw their initial capital by paying an earlyExitFee. The fee decreases with the time of withdrawal, meaning that the earlier the user withdraws, the higher the earlyExitFee they must pay. Users from the variable side receive this earlyExitFee as compensation for a reduced rate of income due to the diminished capital deployed to the underlying asset, in this case, Lido Liquid Staking protocol. Users on the variable side can withdraw their income from Fixed deposit in Lido Liquid Staking protocol anytime after Vault starts. They can withdraw multiple times or all income in one swoop by waiting for the vault to end.

After the vault ends, Fixed users can withdraw their initial capital in full size.

Variable users receive their income from the deposits of fixed users. Income is accumulated daily through rebasing the amount of stETH on the Saffron Lido Vault’s stETH balance. Daily income is based on the amount of stETH on the balance. If no variable users withdraw until the last day, they will receive the highest cumulative income after the Saffron Lido Vault ends.

On the other hand, if a variable user withdraws their income before the Saffron Lido Vault ends, the vault needs to withdraw a portion of its stETH balance from the Lido Liquid Staking protocol to pay the variable user in ETH. This withdrawal decreases the vault's daily income, reducing the income for all remaining users. To address this, we ensured that variable users receive the same amount at the end of the Saffron Lido Vault, regardless of whether anyone withdraws before the end. This was achieved by using ‘stakes’ in our calculations for variable users' income.

___



# Audit scope

[lido-fiv @ 7246b6651c8affffe17faa4d2984975102a65d81](https://github.com/saffron-finance/lido-fiv/tree/7246b6651c8affffe17faa4d2984975102a65d81)
- [lido-fiv/contracts/LidoVault.sol](lido-fiv/contracts/LidoVault.sol)
- [lido-fiv/contracts/VaultFactory.sol](lido-fiv/contracts/VaultFactory.sol)
- [lido-fiv/contracts/interfaces/ILido.sol](lido-fiv/contracts/interfaces/ILido.sol)
- [lido-fiv/contracts/interfaces/ILidoVault.sol](lido-fiv/contracts/interfaces/ILidoVault.sol)
- [lido-fiv/contracts/interfaces/ILidoVaultInitializer.sol](lido-fiv/contracts/interfaces/ILidoVaultInitializer.sol)
- [lido-fiv/contracts/interfaces/ILidoWithdrawalQueueERC721.sol](lido-fiv/contracts/interfaces/ILidoWithdrawalQueueERC721.sol)


