# Mach Finance

Mach Finance is a fork of Compound Finance v2 with additional features such as supply caps and one click mint to use as collateral.
The protocol will be deployed as a native application on the upcoming Sonic Network. 

## What is different about Mach Finance?
- **Supply Caps**: The protocol will have a maximum supply for each cToken.
- **Mint and use asset as collateral**: Users will be able to mint cTokens and use them as collateral for borrowing in a single function call for better UX.
- **block.timestamp vs block.number**: The protocol will use `block.timestamp` for all interest rate calculations, instead of `block.number`.
- **Reward Distribution**: Instead of the `Comptroller` updating & distributing rewards, a separate contract `RewardDistributor` (WIP) will be responsible for distributing rewards.
- **Price Oracle**: Via an Upgradable `PriceOracleAggregator.sol`, the protocol has a priority list of price feeds such as Pyth and Band to fetch price data.

## Usage

### Build

```shell
$ forge build
```

### Test

```shell
$ forge test --force
```

### Format

```shell
$ forge fmt
```

### Deploy on SONIC Testnet

```shell
$ forge script ./script/DeployMachFi.s.sol --broadcast --rpc-url sonic_testnet --force 
```
