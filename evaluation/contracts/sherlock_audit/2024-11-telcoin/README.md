
# Telcoin contest details

- Join [Sherlock Discord](https://discord.gg/MABEWyASkp)
- Submit findings using the issue page in your private contest repo (label issues as med or high)
- [Read for more details](https://docs.sherlock.xyz/audits/watsons)

# Q&A

### Q: On what chains are the smart contracts going to be deployed?
Polygon
___

### Q: If you are integrating tokens, are you allowing only whitelisted tokens to work with the codebase or any complying with the standard? Are they assumed to have certain properties, e.g. be non-reentrant? Are there any types of [weird tokens](https://github.com/d-xo/weird-erc20) you want to integrate?
Stablecoins listed in the StablecoinHandler.sol are the same as the stablecoin contract being evaluated in this audit and are expected to behave as such, not simply "any stablecoin". Besides this tokens are expected to behave like normal ERC20s. Tokens that spend off fee portions should be allowed, and if a portion of the fee sent off breaks the contract, than this would be an issue. However issues raised simply stating "user's might get less than expected" is not valid if the source of this issue is simply lacking checks for return values.  We will also work with USDT and USDC.
___

### Q: Are there any limitations on values set by admins (or other roles) in the codebase, including restrictions on array lengths?
No
___

### Q: Are there any limitations on values set by admins (or other roles) in protocols you integrate with, including restrictions on array lengths?
No
___

### Q: For permissioned functions, please list all checks and requirements that will be made before calling the function.
Potentially multiple trades are prepared durning a single swap. Permissioned functions rely on swap data provided from an aggregator for the user's swap and buyback swap. Some of these payloads need to be signed by the user and appropriate checks are made for approvals, ensuring that the user's have approved the tokens to swap, or tokens taken from safes already have approvals to do so. FX rates are provided for stablecoin swaps and these rates are verifies off chain and their values align. 
___

### Q: Is the codebase expected to comply with any EIPs? Can there be/are there any deviations from the specification?
Most all contracts will be behind proxies and should comply with eip-1967. That said, it is not the auditor's responsibility to verify that the contract is currently backwards compatible with an existing deployment. 
___

### Q: Are there any off-chain mechanisms for the protocol (keeper bots, arbitrage bots, etc.)? We assume they won't misbehave, delay, or go offline unless specified otherwise.
An executor system is used to perform transactions on the user's behalf. It is assumed that this system will not go offline. 
___

### Q: If the codebase is to be deployed on an L2, what should be the behavior of the protocol in case of sequencer issues (if applicable)? Should Sherlock assume that the Sequencer won't misbehave, including going offline?
The system should assume that so long as the network is running, bridge related issues should not affect the workings of this contract. 
___

### Q: What properties/invariants do you want to hold even if breaking them has a low/unknown impact?
N/A
___

### Q: Please discuss any design choices you made.
The contract is constructed so that most inputs can be ignored in nearly any flow unless this would lead to a loss of funds. For example, the protocol may chose not to perform a buyback, are accept a fee, etc. However is the value for a safe is not passed in tokens should not be able to be sent into space or the zero address. 
Tokens received in stablecoin swaps do not need to back the currencies that are minted. 
___

### Q: Please list any known issues and explicitly state the acceptable risks for each known issue.
Defi trades in different portions of the swap may fail, causing the entire transaction to fail. This is acceptable. Actions are expected to be atomic and if portions are not successful the entire transaction should not succeed. 
___

### Q: We will report issues where the core protocol functionality is inaccessible for at least 7 days. Would you like to override this value?
No
___

### Q: Please provide links to previous audits (if any).
https://audits.sherlock.xyz/contests/196
___

### Q: Please list any relevant protocol resources.
N/A
___

### Q: Additional audit information.
N/A
___



# Audit scope


[telcoin-audit @ 15c5381f16f6a7febd9a07cba2f1f77fbce2184f](https://github.com/telcoin/telcoin-audit/tree/15c5381f16f6a7febd9a07cba2f1f77fbce2184f)
- [telcoin-audit/contracts/stablecoin/Stablecoin.sol](telcoin-audit/contracts/stablecoin/Stablecoin.sol)
- [telcoin-audit/contracts/stablecoin/StablecoinHandler.sol](telcoin-audit/contracts/stablecoin/StablecoinHandler.sol)
- [telcoin-audit/contracts/swap/AmirX.sol](telcoin-audit/contracts/swap/AmirX.sol)
- [telcoin-audit/contracts/swap/interfaces/ISimplePlugin.sol](telcoin-audit/contracts/swap/interfaces/ISimplePlugin.sol)
- [telcoin-audit/contracts/util/abstract/Blacklist.sol](telcoin-audit/contracts/util/abstract/Blacklist.sol)


