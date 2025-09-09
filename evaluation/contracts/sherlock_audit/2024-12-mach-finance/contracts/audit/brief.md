# Audit

Mach Finance is a Compound v2 fork with few changes highlighted here:
1. Supply caps for `cToken`
2. Upgradeable Price Oracle Aggregator that consists of Pyth & API3 oracles
3. `block.timestamp` instead of `block.number` when modelling interest rates
4. Extract out accrual & distribution that was previously in `Comptroller` to external `Distributor` contract
5. `mintAsCollateral` function that mints & enables the asset as collateral

# Files to review
- `src/Oracles/API3/*.sol`
- `src/Oracles/Pyth/*.sol`
- `src/Oracles/IOracleSource.sol`
- `src/Oracles/PriceOracleAggregator.sol`
- `src/Rewards/IRewardDistributor.sol`
- `src/**/BaseJumpRateModelV2.sol`
- `src/BaseJumpRateModelV2.sol`
- `src/CErc20.sol`
- `src/CErc20Delegate.sol`
- `src/CErc20Delegator.sol`
- `src/CErc20Immutable.sol`
- `src/Comptroller.sol`
- `src/ComptrollerInterface.sol`
- `src/ComptrollerStorage.sol`
- `src/CSonic.sol`
- `src/CToken.sol`
- `src/CTokenInterfaces.sol`
- `src/ErrorReporter.sol`
- `src/ExponentialNoError.sol`
- `src/InterestRateModel.sol`
- `src/JumpRateModelV2.sol`
- `src/Maximillion.sol`
- `src/Unitroller.sol`


# Difference with Compound v2
This PR serves the implementation differences between Mach Finance and Compound v2 

Compound v2 (forked from commit) -> a3214f67b73310d547e00fc578e8355911c9d376
Mach Finance [Code Changes]
- https://github.com/Mach-Finance/contracts/pull/1 
- https://github.com/Mach-Finance/contracts/pull/3 
- https://github.com/Mach-Finance/contracts/pull/5

`forge fmt` was run against Compound v2 
- https://github.com/Mach-Finance/contracts/pull/2


# Specification for added features

## Price Oracle
In the beginning we plan to list *safe* assets that are supported by **Pyth** + **API3** Oracles.
The price feed used for the wrapped / bridged assets is the price feed of the underlying asset.
These assets include, which will be retrieved from the respective price feeds
- $USDC (Wormhole / LayerZero)
- $FTM / $S
- $ETH (Wormhole / LayerZero)
- $wBTC (Wormhole / LayerZero)


For the next phase of audits, we plan to list other assets such as:
- $S LST
- Yield Bearing Tokens
- Governance tokens of major Sonic Players
- $SolvBTC
- $ONE

These assets will be retrieved from the respective price feeds, that most likely depend on custom price feed implementations from different liquidity pools.

`PriceOracleAggregator.sol` is an upgradeable contract that has a list of oracles for each token. There is a priority list of oracles that are used to get the price of the token. The first oracle that returns a price is used as the price feed. Admin can add, remove and re-order the priority list of oracles. If the price feed is invalid for particular provider, the next provider will be used. 

The priority list is (highest priority first):
1. Pyth
2. API3

### Oracle Sources 
- Pyth: `src/Oracles/Pyth/PythOracle.sol` - 
    - https://api-reference.pyth.network/price-feeds/evm/getPriceUnsafe
    - The team will select the price feed id for each token, and set it in the `PriceOracleAggregator.sol` contract

- API3: `src/Oracles/API3/API3Oracle.sol` - 
    - https://docs.api3.org/dapps/integration/contract-integration.html
    - The team will select the API3 proxy address for each token, and set it in the `PriceOracleAggregator.sol` contract
    - As per the API3 docs, the team will purchase the plan to let API3 update the price every 0.25% movement in price. 

## Supply Caps
Similar to Compound v2's borrow cap, we have supply caps for each `cToken`. 

Amount supplied to the protocol is capped by the `supplyCap` of the `cToken`. It is governed by the following equation:
- `totalCash + totalBorrows - totalReserves < supplyCap`

The protocol will not allow any more `mint` / `mintAsCollateral` to be done if the supply cap is reached.

Supply caps will be used to minimise risk, especially for long tail assets, to prevent draining the protocol of liquidity if one provides a large amount of liquidity.

Supply guardian will be assigned to a different address from the `admin` address. The supply guardian will have the ability to update the supply cap of the `cToken`.

## `block.timestamp` instead of `block.number`

For interest rate calculations, `block.timestamp` is used instead of `block.number`. 
The change was made to make interest rate calculations more accurate, independent of different block times on Sonic chain. 

Ensure that `block.timestamp` interest rate calculations are not prone to `accrueInterest` manipulation calls that may spike up the interest rate to unsustainably high levels, especially given Sonic chain's fast block times of ~0.3-0.5 seconds.

The interest rate model used will be `JumpRateModelV2` from Compound v2, with differing parameters for each `cToken`.

## Accrual & Distribution

In Compound v2, the accrual and distribution of rewards are done in the `Comptroller` contract. In Mach Finance, the accrual and distribution of rewards are done in the `Distributor` contract. 

This Distributor contract is not implemented in the codebase yet. In later audits, the team will implement the `Distributor` contract, to calculate and distribute token rewards to suppliers and borrowers.

## `mintAsCollateral`

`mint` from Compound v2 only allows users to deposit `cToken`. They need to call `enterMarkets` to include it in liquidity calculations as collateral. 

To simplify user experience, the team introduces `mintAsCollateral` function that allows users to mint `cToken`, and include it in their liquidity calculations as collateral.

Case 1:
- `mintAsCollateral` for 100 USDC
- 100 USDC in liquidity calculations as collateral

Case 2:
- `mint` for 100 USDC (not included in liquidity calculations as collateral)
- `mintAsCollateral` for 100 USDC
- 200 USDC in liquidity calculations as collateral

Case 3:
- `mint` for 100 USDC
- `enterMarkets` for USDC
- `mintAsCollateral` for 100 USDC
- 200 USDC in liquidity calculations as collateral

Check possible front-running attacks on `mintAsCollateral`, what are the possible implications?
