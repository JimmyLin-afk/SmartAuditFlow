# Perennial V2 Update #4 contest details

- Join [Sherlock Discord](https://discord.gg/MABEWyASkp)
- Submit findings using the **Issues** page in your private contest repo (label issues as **Medium** or **High**)
- [Read for more details](https://docs.sherlock.xyz/audits/watsons)

# Q&A

### Q: On what chains are the smart contracts going to be deployed?
Arbitrum, Base
___

### Q: If you are integrating tokens, are you allowing only whitelisted tokens to work with the codebase or any complying with the standard? Are they assumed to have certain properties, e.g. be non-reentrant? Are there any types of [weird tokens](https://github.com/d-xo/weird-erc20) you want to integrate?
ERC20 only, particularly DSU and (native) USDC stablecoins.
___

### Q: Are there any limitations on values set by admins (or other roles) in the codebase, including restrictions on array lengths?
Yes:
- Owner is trusted and controls limits in ProtocolParameters (see packages/core/contracts/types/ProtocolParameter.sol).  ProtocolParameters place limitations on MarketParameters and RiskParameters.  Owner also controls MarketParameters (see packages/core/contracts/types/MarketParameter.sol) and OracleParameters (see packages/oracle/contracts/types/OracleParameter.sol).  MarketParameters place limitations on trading activity.  This includes limiting array length (maxPendingGlobal and maxPendingLocal).  OracleParameters limit oracle fees and cap update intervals configured in KeeperOracleParameter.sol.
- Coordinator is partially trusted and controls RiskParameters (see packages/core/contracts/types/RiskParameter.sol).  RiskParameters also place limitations on trading activity.
- Vault Coordinator is a trusted role for the new SolverVault.  This role has authority to trade on behalf of the Vault but not to withdraw funds.  Aside from extraction of funds via self-trading, any elevation of privilege issue is in-scope.

Owners and coordinators may adjust configuration on the running protocol as appropriate for market and chain conditions.
Hardcoded parameters for L2 gas measurement may be adjusted prior to release based on chain conditions.
___

### Q: Are there any limitations on values set by admins (or other roles) in protocols you integrate with, including restrictions on array lengths?
No
___

### Q: Is the codebase expected to comply with any specific EIPs?
Our codebase uses EIP-712 to verify signed messages.
___

### Q: Are there any off-chain mechanisms involved in the protocol (e.g., keeper bots, arbitrage bots, etc.)? We assume these mechanisms will not misbehave, delay, or go offline unless otherwise specified.
Yes; keeper bots are employed to:
- Commit oracle updates and settle markets
- Liquidate users under their maintenance requirement
- Process certain order types and market interactions, such as those used for intents, collateral accounts, and trigger orders

___

### Q: What properties/invariants do you want to hold even if breaking them has a low/unknown impact?
Fine-grained invariants are documented in natspec comments.  We have neither documented coarse-grained invariants, nor created automated invariant tests to check them.  Watsons are encouraged to raise an issue at the appropriate severity if a perceived invariant is violated.
___

### Q: Please discuss any design choices you made.
Perennial is a complex codebase and each release involves evaluating tradeoffs to determine the best design to achieve our goals.  As such, we encourage auditors to thoroughly read documentation, and inquire about specific design decisions they find curious.
___

### Q: Please provide links to previous audits (if any).
https://github.com/equilibria-xyz/perennial-v2/tree/main/audits
___

### Q: Please list any relevant protocol resources.
There are no documentation changes for this release, however these materials remain relevant to the protocol in general:
V2 Docs: https://docs.perennial.finance
V2 Mechanism 1-pager: https://docs.google.com/document/d/1f-V_byFYkJdJAHMXxN2NiiDqysYhoqKzZXteee8BuIQ
V2.3 Intents Doc: https://docs.google.com/document/d/12nCBzhwYRCwAvNESxV0HHYVTFsS2-yNu5BsD9BRjI9M
___

### Q: Additional audit information.
The base of this audit should be this commit: https://github.com/equilibria-xyz/perennial-v2/commit/9c70add56887628ecd999d927c7e0351a3524de8

This release makes (relatively) minor changes intended to improve user experience and reduce transaction fees.

Perennial Intents: https://docs.google.com/document/d/1byG6wleAmR4LAK0x4qFNfmOaXk0Vef__tyYUxF4a4bE/edit?usp=drivesdk


# Audit scope

[perennial-v2 @ 1beb10a3fe23a8a594b4275d376e261dffa811c2](https://github.com/equilibria-xyz/perennial-v2/tree/1beb10a3fe23a8a594b4275d376e261dffa811c2)
- [perennial-v2/packages/core/contracts/Market.sol](perennial-v2/packages/core/contracts/Market.sol)
- [perennial-v2/packages/core/contracts/Verifier.sol](perennial-v2/packages/core/contracts/Verifier.sol)
- [perennial-v2/packages/core/contracts/libs/CheckpointLib.sol](perennial-v2/packages/core/contracts/libs/CheckpointLib.sol)
- [perennial-v2/packages/core/contracts/libs/InvariantLib.sol](perennial-v2/packages/core/contracts/libs/InvariantLib.sol)
- [perennial-v2/packages/core/contracts/libs/MagicValueLib.sol](perennial-v2/packages/core/contracts/libs/MagicValueLib.sol)
- [perennial-v2/packages/core/contracts/types/Fill.sol](perennial-v2/packages/core/contracts/types/Fill.sol)
- [perennial-v2/packages/core/contracts/types/Guarantee.sol](perennial-v2/packages/core/contracts/types/Guarantee.sol)
- [perennial-v2/packages/core/contracts/types/Order.sol](perennial-v2/packages/core/contracts/types/Order.sol)
- [perennial-v2/packages/core/contracts/types/Position.sol](perennial-v2/packages/core/contracts/types/Position.sol)
- [perennial-v2/packages/core/contracts/types/Take.sol](perennial-v2/packages/core/contracts/types/Take.sol)
- [perennial-v2/packages/oracle/contracts/keeper/KeeperOracle.sol](perennial-v2/packages/oracle/contracts/keeper/KeeperOracle.sol)
- [perennial-v2/packages/periphery/contracts/CollateralAccounts/AccountVerifier.sol](perennial-v2/packages/periphery/contracts/CollateralAccounts/AccountVerifier.sol)
- [perennial-v2/packages/periphery/contracts/CollateralAccounts/Controller.sol](perennial-v2/packages/periphery/contracts/CollateralAccounts/Controller.sol)
- [perennial-v2/packages/periphery/contracts/CollateralAccounts/Controller_Incentivized.sol](perennial-v2/packages/periphery/contracts/CollateralAccounts/Controller_Incentivized.sol)
- [perennial-v2/packages/periphery/contracts/CollateralAccounts/Controller_Optimism.sol](perennial-v2/packages/periphery/contracts/CollateralAccounts/Controller_Optimism.sol)
- [perennial-v2/packages/periphery/contracts/CollateralAccounts/types/RelayedTake.sol](perennial-v2/packages/periphery/contracts/CollateralAccounts/types/RelayedTake.sol)
- [perennial-v2/packages/periphery/contracts/MultiInvoker/MultiInvoker.sol](perennial-v2/packages/periphery/contracts/MultiInvoker/MultiInvoker.sol)
- [perennial-v2/packages/periphery/contracts/MultiInvoker/MultiInvoker_Arbitrum.sol](perennial-v2/packages/periphery/contracts/MultiInvoker/MultiInvoker_Arbitrum.sol)
- [perennial-v2/packages/periphery/contracts/MultiInvoker/MultiInvoker_Optimism.sol](perennial-v2/packages/periphery/contracts/MultiInvoker/MultiInvoker_Optimism.sol)
- [perennial-v2/packages/periphery/contracts/TriggerOrders/Manager.sol](perennial-v2/packages/periphery/contracts/TriggerOrders/Manager.sol)
- [perennial-v2/packages/periphery/contracts/TriggerOrders/Manager_Arbitrum.sol](perennial-v2/packages/periphery/contracts/TriggerOrders/Manager_Arbitrum.sol)
- [perennial-v2/packages/periphery/contracts/TriggerOrders/Manager_Optimism.sol](perennial-v2/packages/periphery/contracts/TriggerOrders/Manager_Optimism.sol)
- [perennial-v2/packages/vault/contracts/MakerVault.sol](perennial-v2/packages/vault/contracts/MakerVault.sol)
- [perennial-v2/packages/vault/contracts/SolverVault.sol](perennial-v2/packages/vault/contracts/SolverVault.sol)
- [perennial-v2/packages/vault/contracts/Vault.sol](perennial-v2/packages/vault/contracts/Vault.sol)
- [perennial-v2/packages/vault/contracts/libs/MakerStrategyLib.sol](perennial-v2/packages/vault/contracts/libs/MakerStrategyLib.sol)
- [perennial-v2/packages/vault/contracts/libs/SolverStrategyLib.sol](perennial-v2/packages/vault/contracts/libs/SolverStrategyLib.sol)
- [perennial-v2/packages/vault/contracts/types/Checkpoint.sol](perennial-v2/packages/vault/contracts/types/Checkpoint.sol)
- [perennial-v2/packages/vault/contracts/types/Target.sol](perennial-v2/packages/vault/contracts/types/Target.sol)
- [perennial-v2/packages/vault/contracts/types/VaultParameter.sol](perennial-v2/packages/vault/contracts/types/VaultParameter.sol)


