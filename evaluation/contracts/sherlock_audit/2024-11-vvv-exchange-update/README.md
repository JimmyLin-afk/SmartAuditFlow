
# vVv Exchange Update contest details

- Join [Sherlock Discord](https://discord.gg/MABEWyASkp)
- Submit findings using the issue page in your private contest repo (label issues as med or high)
- [Read for more details](https://docs.sherlock.xyz/audits/watsons)

# Q&A

### Q: On what chains are the smart contracts going to be deployed?
Eth, base, bnb, avalanche, polkadot, arbitrum
___

### Q: If you are integrating tokens, are you allowing only whitelisted tokens to work with the codebase or any complying with the standard? Are they assumed to have certain properties, e.g. be non-reentrant? Are there any types of [weird tokens](https://github.com/d-xo/weird-erc20) you want to integrate?
Yes only certain ERC20 tokens (i.e. specific ERC20 such as USDC for investment, and another specific ERC20 for claims) will interact with the codebase. It's expected that none are "weird" and would have proper protections such as being non-reentrant.

In general, the contracts will facilitate transfer of ERC20 tokens. In the case of VVVVCInvestmentLedger.sol, the contract handles users' deposits of ERC20 tokens as well as permissioned withdrawal of these tokens. VVVVCTokenDistributor.sol handles the transfer of ERC20 tokens from approved wallets to callers of the claim()  function, given all other checks are passed.
___

### Q: Are there any limitations on values set by admins (or other roles) in the codebase, including restrictions on array lengths?
No
___

### Q: Are there any limitations on values set by admins (or other roles) in protocols you integrate with, including restrictions on array lengths?
No
___

### Q: Is the codebase expected to comply with any specific EIPs?
Both VVVVCInvestmentLedger.sol and VVVVCTokenDistributor.sol define EIP-712 domain components and utilize EIP-712 structured data formats in validating signature used to validate calls to VVVVCInvestmentLedger:invest() and VVVVCTokenDistributor:claim().
___

### Q: Are there any off-chain mechanisms involved in the protocol (e.g., keeper bots, arbitrage bots, etc.)? We assume these mechanisms will not misbehave, delay, or go offline unless otherwise specified.
Yes. A trusted off-chain centralized system handles creating the signatures which validate calls to VVVVCInvestmentLedger:invest() and VVVVCTokenDistributor:claim() and maintaining the wallet ERC20 token balances from which VVVVCTokenDistributor:claim() transfers. In the case of VVVVCInvestmentLedger:invest() the centralized system generates approved input parameters, and the contract ensures that these validated parameters aren't exceeded (see check for investment round start and end timestamp on line 151 of VVVVCInvestmentLedger.sol). In the case of VVVVCTokenDistributor:claim(), the centralized system is the sole source of approval for transactions/input parameters.
___

### Q: What properties/invariants do you want to hold even if breaking them has a low/unknown impact?
N/A
___

### Q: Please discuss any design choices you made.
- VVVVCInvestmentLedger:FEE_DENOMINATOR value of 10_000 was deemed to provide sufficient precision for any fees applied to deposit of ERC20 tokens to the VVVVCInvestmentLedger contract.

- Successful function of VVVVCInvestmentLedger:invest() depends on the successful management of the centralized system for creating signatures.

- Successful function of VVVVCTokenDistributor:claim() depends on successful management of the centralized system which handles both signature creation and ensuring all wallets from which tokens will be claimed have the expected balances to be withdrawn.
___

### Q: Please provide links to previous audits (if any).
N/A
___

### Q: Please list any relevant protocol resources.
N/A
___

### Q: Additional audit information.
Prior audit in which VVVAuthorizationRegistry.sol and VVVAuthorizationRegistryChecker.sol were in scope: https://github.com/sherlock-audit/2024-03-vvv-vesting-staking

Additional prior Context Q&A answered/formatted here if helpful: https://gist.github.com/curi0n-s/b89d869d637e703bbdda8899bb8e978b
___



# Audit scope


[vvv-platform-smart-contracts @ 29fdceaeed9a4174039b66d85a5d4ce5d0ed14bf](https://github.com/vvvdevs/vvv-platform-smart-contracts/tree/29fdceaeed9a4174039b66d85a5d4ce5d0ed14bf)
- [vvv-platform-smart-contracts/contracts/vc/VVVVCInvestmentLedger.sol](vvv-platform-smart-contracts/contracts/vc/VVVVCInvestmentLedger.sol)
- [vvv-platform-smart-contracts/contracts/vc/VVVVCTokenDistributor.sol](vvv-platform-smart-contracts/contracts/vc/VVVVCTokenDistributor.sol)


