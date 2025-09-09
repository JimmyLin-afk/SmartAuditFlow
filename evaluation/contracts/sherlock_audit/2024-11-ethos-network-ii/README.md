
# Ethos Network II contest details

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
Graduate and Slasher are assumed to be contracts, also deployed and owned by Ethos.
Fee receiver should not have any additional access (beyond standard users), though if the fee receiver is set to owner/admin that's fine.

For both contracts:
- Maximum total fees cannot exceed 10%

Vouch:
- Maximum total slash cannot exceed 10%
- Minimum vouch amount must be >= ABSOLUTE_MINIMUM_VOUCH_AMOUNT (0.0001 ether)
- Maximum number of _active_ vouches received by a profile cannot exceed 256
- Maximum number of _active_ vouches given by a profile cannot exceed 256

Reputation Market:
- Cannot remove all market configs (must keep at least 1)
- Base price must be >= MINIMUM_BASE_PRICE (0.0001 ether)
- Initial liquidity must be >= DEFAULT_PRICE (0.01 ether)
- Initial votes cannot be 0 (would cause divide by zero)
___

### Q: Are there any limitations on values set by admins (or other roles) in protocols you integrate with, including restrictions on array lengths?
These contracts rely on the settings and configuration covered in Ethos Network non-financial contracts, audited here:
https://audits.sherlock.xyz/dashboard/ce54133aa7c357e64f460e6f53d3f2a5
___

### Q: Is the codebase expected to comply with any specific EIPs?
No compliance with EIPs
___

### Q: Are there any off-chain mechanisms involved in the protocol (e.g., keeper bots, arbitrage bots, etc.)? We assume these mechanisms will not misbehave, delay, or go offline unless otherwise specified.
Currently slashing and graduating reputation markets are not yet implemented. However, they are expected to be on-chain once implemented.
___

### Q: What properties/invariants do you want to hold even if breaking them has a low/unknown impact?
The vouch and vault contracts must never revert a transaction due to running out of funds.

Reputation Markets must never sell the initial votes. They must never pay out the initial liquidity deposited. The only way to access those funds is to graduate the market.
___

### Q: Please discuss any design choices you made.
We chose to go with a maximum 256 vouches because of constraints imposed by gas limits. We initially investigated a shares/asset split that allowed unlimited vouches but it was too error prone and we went back to a much simpler model. 
___

### Q: Please provide links to previous audits (if any).
Not these same contracts, but some of the related contracts on which these depend: https://audits.sherlock.xyz/dashboard/ce54133aa7c357e64f460e6f53d3f2a5/
___

### Q: Please list any relevant protocol resources.
whitepaper: whitepaper.ethos.network
website: ethos.network
testnet app: sepolia.ethos.network
___



# Audit scope


[ethos @ 12e6a9d3b813040463266483733a84218a35847f](https://github.com/trust-ethos/ethos/tree/12e6a9d3b813040463266483733a84218a35847f)
- [ethos/packages/contracts/contracts/EthosVouch.sol](ethos/packages/contracts/contracts/EthosVouch.sol)
- [ethos/packages/contracts/contracts/ReputationMarket.sol](ethos/packages/contracts/contracts/ReputationMarket.sol)
- [ethos/packages/contracts/contracts/errors/ReputationMarketErrors.sol](ethos/packages/contracts/contracts/errors/ReputationMarketErrors.sol)
- [ethos/packages/contracts/contracts/utils/Common.sol](ethos/packages/contracts/contracts/utils/Common.sol)

