
# Mach Finance contest details

- Join [Sherlock Discord](https://discord.gg/MABEWyASkp)
- Submit findings using the issue page in your private contest repo (label issues as med or high)
- [Read for more details](https://docs.sherlock.xyz/audits/watsons)

# Q&A

### Q: On what chains are the smart contracts going to be deployed?
Sonic (https://www.soniclabs.com/) -> Cancun support
___

### Q: If you are integrating tokens, are you allowing only whitelisted tokens to work with the codebase or any complying with the standard? Are they assumed to have certain properties, e.g. be non-reentrant? Are there any types of [weird tokens](https://github.com/d-xo/weird-erc20) you want to integrate?
Only whitelisted ERC20 tokens will be supported by Mach Finance to be supplied / borrowed
These whitelisted tokens are expected to follow one of these two behaviours
- Return true if transfer succeeds
- Not return anything, if transfer suceeds

Tokens are expected to have a maximum of 36 decimal places for precision oracle price reasons
We do not support ERC777 tokens

ERC20 tokens we plan to integrate
- USDC (6 decimals)
- USDT
  - Not return anything if transfer succeeds 
- SolvBTC, 
- Layer Zero wrapped tokens such as
  - lz.WBTC, lz.WETH
- Wormhole wrapped tokens on Sonic such as 
  - Wormhole WETH, Wormhole WBTC


In the future we are looking to support:
- $S (SONIC) Liquid Staking Token 

___

### Q: Are there any limitations on values set by admins (or other roles) in the codebase, including restrictions on array lengths?
Comptroller admin is trusted in this case. After the protocol has been launched and the situation stabilizes, the Comptroller admin will be relinquished to a timelock contract. Timelock contract will be managed by Safe Multisig, managed by the Mach Finance contributors as signers.

CToken admin is trusted too in this case, similar to above, admin will be reliquinshed to a Timelock contract that is managed by a Safe Multisig.

There are 3 different guardian roles that ensure the safety of the protocol. These guardians do not need to go through a Timelock to call the Comptroller only for these actions:
1. Pause guardian (Compound implementation)
  1. Pause mint 
  2. Pause borrow
  3. Pause transfer
  4. Pause seize 
2. Supply cap for each cToken
  1. Set the maximum amount (less than) of cToken that can be supplied (cash + borrows - reserves)
3. Borrow cap for each cToken
  1. Set the maximum amount (less than) of cToken that can be borrowed (borrows)
___

### Q: Are there any limitations on values set by admins (or other roles) in protocols you integrate with, including restrictions on array lengths?
No
___

### Q: Is the codebase expected to comply with any specific EIPs?
No
___

### Q: Are there any off-chain mechanisms involved in the protocol (e.g., keeper bots, arbitrage bots, etc.)? We assume these mechanisms will not misbehave, delay, or go offline unless otherwise specified.
Similar to Compound and other borrow/lending protocols, there are two off-chain components required:
- Price feed oracles -> Pyth, API3
- Liquidation bots 

Price feed oracles will submit prices at every interval or if there is a percentage change in price for the particular asset
- Pyth -> https://api-reference.pyth.network/price-feeds/evm/getPriceUnsafe
- API3 -> https://docs.api3.org/dapps/integration/

There is an implicit trust that these oracle providers would honour their commitment to provide the best feed prices, if not the protocol may go into bad debt due to arbitrage

Liquidation bots are needed to keep the protocol healthy (no bad debt), similar to Compound v2
- Liquidation bots will monitor if there is an account that has shortfall > 0 
- Shortfall is defined as 
  - $$\text{Shortfall} = \sum_{i} \left( \text{cTokenBorrowAmountInUsd}_i - \text{collateralAmountInUsd}_i \times \text{cTokenCollateralFactor} \right)$$
- Liquidation bot will repay the loan's shortfall on behalf of the account 
- Then the liquidation bot can seize the equivalent shortfall's collateral asset in USD with an additional incentive 
 
___

### Q: What properties/invariants do you want to hold even if breaking them has a low/unknown impact?

For all individuals:
- Sum of each token borrow amount in USD value < Sum of each token deposit amount in USD value
This invariant must hold true to prevent bad debt, assuming liquidators do their job well

Total supply and total borrow balances at the market level must always match the sum of all individual user balances.
___

### Q: Please discuss any design choices you made.
Mach Finance is a Compound v2 Fork, so the design choices made were based on Compound v2. 

Details on the differences between Mach Finance & Compound v2
https://github.com/Mach-Finance/contracts/blob/main/audit/brief.md

Some differences with Compound v2 are highlighted here:

block.timestamp is used instead of block.number for interest accrual
- For Sonic, the block times on testnet vary around ~0.3-0.5 seconds from our observation. Varying block times impact the actual interest accrued. 
- The team wanted to ensure the interest rates are independent of the actual block times, so it won't be affected by the differing block times. Timestamp is more consistent as it depends on the block builder's timestamp. So block.timestamp is used instead of block.number for interest accrual
- We acknowledge possible manipulation of block.timestamp by the block builders. However, in the long term, we felt the effects would not be significant. If there are possible issues with this approach that could impact the protocol severely, it may be a valid finding. 

Token Distribution
In Compound v2, the accrual and distribution of rewards are done in the Comptroller contract. The admin can configure how fast the reward tokens are distributed to borrowers / lenders for specific markets.

Mach Finance aims to extract out the calculation & distribution logic for rewards. Interface is implemented similarly to Compound v2 to ensure similar logic can be implemented for the Distributor contracts. The vanilla Compound v2 emissions / distribution calculation logic is removed from Comptroller contract

The aim is to build the Distributor contracts similar to Compound, but instead of only supporting a single reward token, the Distributor contract can reward the borrowers / lenders with several reward tokens.  Currently, the Distributor contract is not implemented in the codebase yet, so the distribution & calculations are skipped during EVM execution.

A reference that we are looking at is Moonwell MultiRewardDistributor contract here
https://github.com/moonwell-fi/moonwell-contracts-v2/blob/main/src/rewards/MultiRewardDistributor.sol

Once the team has implemented the Distributor contracts, there will be another round of audits to ensure security when updating the protocol to support the Distributor contracts.

Oracles
Oracles used will be Pyth & API3 for Sonic assets supplied / borrowed on Mach Finance

Pyth & API3 contracts can & should be scrutinized deeper to see what can go likely wrong and how Mach Finance can get affected

PriceOracleAggregator.sol is an upgradeable contract that has a list of oracles for each token, this will be used by the Comptroller to fetch underlying prices

There is a priority list of oracles that are used to get the price of the token. The first oracle that returns a price is used as the price feed. Admin can add, remove and re-order the priority list of oracles. If the price feed is invalid for a particular provider, the next provider on the priority list is used.

mintAsCollateral 

Instead of two function calls to supply and set it as collateral, Mach Finance introduces a new function that supplies and sets the supplied token as collateral in a single EVM transaction

This new function may need more scrutiny for possible security issues such as front-running attacks

There are other differences that are described here
https://github.com/Mach-Finance/contracts/blob/main/audit/brief.md

___

### Q: Please provide links to previous audits (if any).
N/A
___

### Q: Please list any relevant protocol resources.
- https://machfi.gitbook.io/
- https://www.machfi.xyz/
- https://github.com/Mach-Finance/contracts/blob/main/audit/brief.md


___

### Q: Additional audit information.
Sonic block times are much shorter than Ethereum, averaging around ~0.3-0.5s seen in the link below
https://testnet.soniclabs.com/

Sonic supports EVM Cancun version, Solidity 0.8.22 is used to compile (see foundry.toml)

Details on the audit brief can be found here:
https://github.com/Mach-Finance/contracts/blob/main/audit/brief.md

Mach Finance is a fork of Compound v2 from this commit -> a3214f67b73310d547e00fc578e8355911c9d376 
https://github.com/compound-finance/compound-protocol/commit/a3214f67b73310d547e00fc578e8355911c9d376

forge fmt was run against the Compound v2 codebase here
https://github.com/Mach-Finance/contracts/pull/2 

Git Diff between Compound v2 & Mach Finance can be found here (excluding forge fmt)
https://github.com/Mach-Finance/contracts/compare/37a4bd855cabc3c3e402c59a138ba555731b7f52...7c617b97ee65acc85d2d45d8f4ed70c16a6a0d83

For the diff above, please look at the files that are part of the audit scope as defined by Sherlock's audit contest, this should be the one that is audited

Ignored Findings:

With regards to the Reward token distribution, the implementation will be out of scope for this audit
Security issues that pertain to the implementation of the Distributor contract such as but not limited to:
- Distributor can revert, making contract unusable
- Distributor can be infinite gas guzzle, making contract usable

The issues above (Distributor contract) will not be in the scope of the audit itself, as they are not implemented. Another future audit will be done when reward distribution is to be implemented to be integrated with the protocol.

The team will ignore any findings that are present in Compound v2, as the focus of the audit contest is to discover bugs / issues that may arise from the git diff against Compound v2
___



# Audit scope


[contracts @ 60e06b7bc8d59055cf399d5c8b09f77482550a85](https://github.com/Mach-Finance/contracts/tree/60e06b7bc8d59055cf399d5c8b09f77482550a85)
- [contracts/src/BaseJumpRateModelV2.sol](contracts/src/BaseJumpRateModelV2.sol)
- [contracts/src/CErc20.sol](contracts/src/CErc20.sol)
- [contracts/src/CErc20Delegate.sol](contracts/src/CErc20Delegate.sol)
- [contracts/src/CErc20Delegator.sol](contracts/src/CErc20Delegator.sol)
- [contracts/src/CSonic.sol](contracts/src/CSonic.sol)
- [contracts/src/CToken.sol](contracts/src/CToken.sol)
- [contracts/src/CTokenInterfaces.sol](contracts/src/CTokenInterfaces.sol)
- [contracts/src/Comptroller.sol](contracts/src/Comptroller.sol)
- [contracts/src/ComptrollerInterface.sol](contracts/src/ComptrollerInterface.sol)
- [contracts/src/ComptrollerStorage.sol](contracts/src/ComptrollerStorage.sol)
- [contracts/src/EIP20Interface.sol](contracts/src/EIP20Interface.sol)
- [contracts/src/EIP20NonStandardInterface.sol](contracts/src/EIP20NonStandardInterface.sol)
- [contracts/src/ErrorReporter.sol](contracts/src/ErrorReporter.sol)
- [contracts/src/ExponentialNoError.sol](contracts/src/ExponentialNoError.sol)
- [contracts/src/InterestRateModel.sol](contracts/src/InterestRateModel.sol)
- [contracts/src/JumpRateModelV2.sol](contracts/src/JumpRateModelV2.sol)
- [contracts/src/Maximillion.sol](contracts/src/Maximillion.sol)
- [contracts/src/Oracles/API3/API3Oracle.sol](contracts/src/Oracles/API3/API3Oracle.sol)
- [contracts/src/Oracles/IOracleSource.sol](contracts/src/Oracles/IOracleSource.sol)
- [contracts/src/Oracles/PriceOracleAggregator.sol](contracts/src/Oracles/PriceOracleAggregator.sol)
- [contracts/src/Oracles/Pyth/PythOracle.sol](contracts/src/Oracles/Pyth/PythOracle.sol)
- [contracts/src/PriceOracle.sol](contracts/src/PriceOracle.sol)
- [contracts/src/Rewards/IRewardDistributor.sol](contracts/src/Rewards/IRewardDistributor.sol)
- [contracts/src/Unitroller.sol](contracts/src/Unitroller.sol)


