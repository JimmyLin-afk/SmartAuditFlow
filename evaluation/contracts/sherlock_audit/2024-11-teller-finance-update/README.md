
# Teller Finance Update contest details

- Join [Sherlock Discord](https://discord.gg/MABEWyASkp)
- Submit findings using the issue page in your private contest repo (label issues as med or high)
- [Read for more details](https://docs.sherlock.xyz/audits/watsons)

# Q&A

### Q: On what chains are the smart contracts going to be deployed?
Ethereum Mainnet, Polygon PoS, Arbitrum One, Base 
___

### Q: If you are integrating tokens, are you allowing only whitelisted tokens to work with the codebase or any complying with the standard? Are they assumed to have certain properties, e.g. be non-reentrant? Are there any types of [weird tokens](https://github.com/d-xo/weird-erc20) you want to integrate?
For LenderGroups (our primary concern for this audit - LenderCommitmentGroup_Smart), we are allowing any token complying with the ERC20 standard except for fee-on-transfer tokens.   Pools will be deployed with the principal token and collateral token types ( both must be ERC20)  already defined and visible on the frontend so borrowers/lenders will be able to vet tokens for weirdness that way and avoid them if needed.  Therefore we do not especially care if a 'weird' token breaks the contract since pools with tokens that are obviously weird will be naturally avoided by users but we would like to know if weird tokens affects do break our pools .  We absolutely do want to fully support all mainstream tokens like USDC,  USDT, WETH, MOG, PEPE, etc.  and we already took special consideration to make sure USDT works with our contracts. 
___

### Q: Are there any limitations on values set by admins (or other roles) in the codebase, including restrictions on array lengths?
Since the first round of this audit in Q2 of 2024, there have been significant updates with regards to access restrictions.  

ProtocolOwner - can set roles such as ProtocolFeeRecipient, Pausers  (in practice, is a multisig wallet).  
ProtocolFeeRecipient - a singleton role that can be set by the ProtocolOwner, this address is where protocol fee funds go.  Should never be a smart contract as this may cause reverts of loans / softpause the protocol.  This is known.
PauserRole - There can be infinite pausers, and these have the power to run pause functions: PauseProtocol, PauseLiquidations, PauseSmartCommitmentForwarder 


Pausers may pause individual LenderGroupPools.
The owner of a LenderGroupPool may set the 'principalPerCollateral' number but this is also Math.min'ed with the uniswap oracle price AND it is ignored if 0 (null) so the pool owner can choose to set this or leave it at zero and allow only the uniswap price to be used for this purpose. 



___

### Q: Are there any limitations on values set by admins (or other roles) in protocols you integrate with, including restrictions on array lengths?
no
___

### Q: Is the codebase expected to comply with any specific EIPs?
Lender Pools principal and collateral tokens are expected to comply with ERC20.  The shares token that is automatically deployed with a pool is also expected to comply with ERC20.   
___

### Q: Are there any off-chain mechanisms involved in the protocol (e.g., keeper bots, arbitrage bots, etc.)? We assume these mechanisms will not misbehave, delay, or go offline unless otherwise specified.
At this time, not keeper bots specifically.  However, each LenderGroups contract does have a liquidation auction that expects the network to not misbehave in order to reach the optimal / fair price of principal tokens paid to claim the collateral tokens for the liquidated loan . 
___

### Q: What properties/invariants do you want to hold even if breaking them has a low/unknown impact?
This is the same as during the last audit:   Very importantly, we expect the following invariant:  the 'EstimatedTotalValue' of a LenderGroupPool should accurately track the value of principal tokens in a pool.  Theoretically, the value of a pool should only be able to become 'less' than the (deposits - withdrawls) of lenders if liquidation auctions occur AND that auction went below net "0" input (int) meaning that the value of principal tokens given is expected to be worth less than the collateral was originally worth.   This also means that lenders should only be able to lose money (in terms of principal tokens) if/when this occurs and no other way.   (If there are other minor ways such as with 'sandwich attacks' , that is OK but we would like to know -- however this likely wont disrupt or prevent day to day operation or invalidate the design as these types of attacks are non-preventable and also present in other similar systems such as Uniswap) 
___

### Q: Please discuss any design choices you made.
We chose to reset liquidation auctions back to the beginning of the timer when any aspect of the protocol is paused in order to make sure the lender group liquidation auctions result in a 'fair market' price as these are ever-decreasing auctions based on time.

We chose to add reentrancy-guard to quite a number of our functions to mitigate unforeseeable exploits and to add a 'hypernative oracle' firewall guard as well which is similar in that it is just a modifier that can revert based on certain criteria (allowlist/denylist). 

We chose to replace the OZ Pauseable contract dependency on TellerV2 with one that is custom so that we can instead use a 'pausing manager' contract that uses roles for Hypernative. This should not break the storage slots on TellerV2 ( it does not as per test upgrade on polygon.)  

We chose to allow for 'multi-hop' price oracle routing, with code taken directly from our already-audited LenderCommitmentForwarder contract.  In practice, it is quite complicated to construct these uniswap pool routing structs manually however it is quite similar to any other uniswap routing scheme in solidity.  This means that the price oracles , when hardcoded into a new LenderGroupPool, are specified with either a single hop or a dual-hop.  Dual hops are very nice when there are two pools with high liquidity that link two assets and the singular pool that links the two assets has low liquidity, for example as how a dual route of PEPE-WETH to WETH-MOG would be a much better price oracle as compared to the singular route of PEPE-MOG whose pool im sure has zero to no liquidity and thus could be easily manipulated -- thus not a good idea to use.   This dual routing option allows us to more safely have a LenderGroupsPool with , for example, PEPE as principal and MOG as collateral.  

The max protocol fee should be no more than 10% 
The max market fee should be no more than 10% 
___

### Q: Please provide links to previous audits (if any).
( Sherlock audit in 2024 Q2 )  https://audits.sherlock.xyz/contests/295?filter=results
( 0xAdri audit in 2024 Q4 ) 

These are issues that were flagged in a private audit as 'potential issues' but which we have reviewed and decided are 'wont fix' as they are either avoidable based on configuration, intended, or otherwise known. 

1. Fee-on-transfer tokens are not supported 
2. Lenders participating in a lender group pool (LenderCommitment_Smart) have to fully re-prepare all of their shares [for withdraw] if they transfer out any of their own shares from their account 
3.  Uniswap Oracle TWAP is being queried from block-N to block-1 on purpose to mitigate flash loan attacks but this causes slightly outdated price information (one block) 
4. Lender pool owners have the option to set the TWAP to any value, including 0, which would be unsafe so it is recommended they use at least a value of 5 and this depends on the quality of the pair pool on Uniswap (ability to push price / influence oracle).

___

### Q: Please list any relevant protocol resources.
Teller docs: https://docs.teller.org/teller-lite 

Audit repo: https://github.com/teller-protocol/teller-protocol-v2-audit-2024    (see readme  for setup instructions - requires nodejs and  rust for foundry+forge - run tests with 'npm contracts test' in the root) 
___

### Q: Additional audit information.
We chose to provide repayment information to the LenderGroups contracts in a new 'hook' from TellerV2 which is a somewhat complex pattern which we want to make sure is resilient and well-built as it is a somewhat experimental, rare and not very tried/true pattern in solidity especially since it is in a try/catch which i personally dislike in solidity.  

We realize that by using the uniswap as a price oracle (even WITH configurable twap) for lender group pools 'principalPerCollateral' for borrowing, there is always some degree of financial risk that a user could manipulate the uniswap pool oracle and then borrow from a lender group pool using that 'ficticious' oracle price ratio.  The goal is not for this to be impossible, but for it to be non-profitable and thus we expect that as long as  a LenderGroupPool has significantly less TVL than the uniswap pool(s) that it uses for oracles, and TWAP is configured appropriately, this should not be profitable.  




___



# Audit scope


[teller-protocol-v2-audit-2024 @ d829864a1ddf408218692449b228af36c23001ef](https://github.com/teller-protocol/teller-protocol-v2-audit-2024/tree/d829864a1ddf408218692449b228af36c23001ef)
- [teller-protocol-v2-audit-2024/packages/contracts/contracts/CollateralManager.sol](teller-protocol-v2-audit-2024/packages/contracts/contracts/CollateralManager.sol)
- [teller-protocol-v2-audit-2024/packages/contracts/contracts/LenderCommitmentForwarder/SmartCommitmentForwarder.sol](teller-protocol-v2-audit-2024/packages/contracts/contracts/LenderCommitmentForwarder/SmartCommitmentForwarder.sol)
- [teller-protocol-v2-audit-2024/packages/contracts/contracts/LenderCommitmentForwarder/extensions/LenderCommitmentGroup/LenderCommitmentGroupShares.sol](teller-protocol-v2-audit-2024/packages/contracts/contracts/LenderCommitmentForwarder/extensions/LenderCommitmentGroup/LenderCommitmentGroupShares.sol)
- [teller-protocol-v2-audit-2024/packages/contracts/contracts/LenderCommitmentForwarder/extensions/LenderCommitmentGroup/LenderCommitmentGroup_Smart.sol](teller-protocol-v2-audit-2024/packages/contracts/contracts/LenderCommitmentForwarder/extensions/LenderCommitmentGroup/LenderCommitmentGroup_Smart.sol)
- [teller-protocol-v2-audit-2024/packages/contracts/contracts/MarketRegistry.sol](teller-protocol-v2-audit-2024/packages/contracts/contracts/MarketRegistry.sol)
- [teller-protocol-v2-audit-2024/packages/contracts/contracts/TellerV2.sol](teller-protocol-v2-audit-2024/packages/contracts/contracts/TellerV2.sol)
- [teller-protocol-v2-audit-2024/packages/contracts/contracts/TellerV2Context.sol](teller-protocol-v2-audit-2024/packages/contracts/contracts/TellerV2Context.sol)
- [teller-protocol-v2-audit-2024/packages/contracts/contracts/TellerV2Storage.sol](teller-protocol-v2-audit-2024/packages/contracts/contracts/TellerV2Storage.sol)
- [teller-protocol-v2-audit-2024/packages/contracts/contracts/escrow/CollateralEscrowV1.sol](teller-protocol-v2-audit-2024/packages/contracts/contracts/escrow/CollateralEscrowV1.sol)
- [teller-protocol-v2-audit-2024/packages/contracts/contracts/libraries/UniswapPricingLibrary.sol](teller-protocol-v2-audit-2024/packages/contracts/contracts/libraries/UniswapPricingLibrary.sol)
- [teller-protocol-v2-audit-2024/packages/contracts/contracts/libraries/V2Calculations.sol](teller-protocol-v2-audit-2024/packages/contracts/contracts/libraries/V2Calculations.sol)


