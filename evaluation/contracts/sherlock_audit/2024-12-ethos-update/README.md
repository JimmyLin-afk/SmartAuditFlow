
# Ethos Update contest details

- Join [Sherlock Discord](https://discord.gg/MABEWyASkp)
- Submit findings using the issue page in your private contest repo (label issues as med or high)
- [Read for more details](https://docs.sherlock.xyz/audits/watsons)

# Q&A

### Q: On what chains are the smart contracts going to be deployed?
Base L2 only
___

### Q: If you are integrating tokens, are you allowing only whitelisted tokens to work with the codebase or any complying with the standard? Are they assumed to have certain properties, e.g. be non-reentrant? Are there any types of [weird tokens](https://github.com/d-xo/weird-erc20) you want to integrate?
We are not integrating ANY tokens. We will only be handling native Ethereum. 
___

### Q: Are there any limitations on values set by admins (or other roles) in the codebase, including restrictions on array lengths?
Owner is trusted. Admin is trusted.
Graduate_Withdraw contracts will also be deployed by and owned by Ethos and trusted.

Reputation Market:
- Cannot remove all market configs (must keep at least 1)
- Base price must be >= MINIMUM_BASE_PRICE (0.0001 ether)
- Must maintain LMSR invariant (yes + no price sum to 1)
- A user cannot sell more votes than they own
- A graduated market cannot accept new trades or be recreated.
- Total contract balance must be greater or equal to all active (non-graduated) market funds

___

### Q: Are there any limitations on values set by admins (or other roles) in protocols you integrate with, including restrictions on array lengths?
These contracts rely on the settings and configuration covered in Ethos Network non-financial contracts, audited here:
https://audits.sherlock.xyz/contests/584
___

### Q: Is the codebase expected to comply with any specific EIPs?
Currently graduating reputation markets are not yet implemented. However, they are expected to be on-chain once implemented.
___

### Q: Are there any off-chain mechanisms involved in the protocol (e.g., keeper bots, arbitrage bots, etc.)? We assume these mechanisms will not misbehave, delay, or go offline unless otherwise specified.
There are no off-chain mechanisms involved in the Reputation Market protocol. We do not use the information in Reputation Markets to impact Ethos credibility scores.
___

### Q: What properties/invariants do you want to hold even if breaking them has a low/unknown impact?
The contract must never pay out the initial liquidity deposited as part of trading. The only way to access those funds is to graduate the market.
___

### Q: Please discuss any design choices you made.
We opted to use an LMSR algorithm for the bonding curve, as it's proven to operate well with AMMs like Polymarket. This was in response to a critical finding in our previous audit.
___

### Q: Please provide links to previous audits (if any).
https://github.com/sherlock-audit/2024-10-ethos-network
https://github.com/sherlock-audit/2024-11-ethos-network-ii
___

### Q: Please list any relevant protocol resources.
whitepaper: https://whitepaper.ethos.network
website: https://ethos.network 
testnet app: https://dev.ethos.markets


___

### Q: Additional audit information.
The major changes are in the addition and integration with the LMSR library.
___



# Audit scope


[ethos @ 9c41a570cb87fc19989a6880ae2d6a21b3d36c1b](https://github.com/trust-ethos/ethos/tree/9c41a570cb87fc19989a6880ae2d6a21b3d36c1b)
- [ethos/packages/contracts/contracts/ReputationMarket.sol](ethos/packages/contracts/contracts/ReputationMarket.sol)
- [ethos/packages/contracts/contracts/errors/ReputationMarketErrors.sol](ethos/packages/contracts/contracts/errors/ReputationMarketErrors.sol)
- [ethos/packages/contracts/contracts/utils/LMSR.sol](ethos/packages/contracts/contracts/utils/LMSR.sol)


