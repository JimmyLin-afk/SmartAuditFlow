# Saffron Lido Fixed Income Vaults

## Overview

Saffron Lido Fixed Income Vaults enable fixed and variable yield exchange. Yield for Saffron Lido Vaults is produced from [Lido ETH Staking](https://lido.fi/faq). Fixed side participants are paid an up-front fixed amount for depositing ETH into the vault, which is then staked on Lido. The variable side likewise pays a fixed amount to earn all staking earnings generated over the duration of the vault's lifetime. Parties agree ahead of time on terms, including the lockup duration, total asset value, and fixed payment amount.

Vaults facilitate the above interaction. Each vault, when created, is initialized with parameters defining the agreement between the fixed and variable side participants. Parameters include:

    duration
    fixed side deposit capacity
    variable side deposit capacity - calculated from fixed side premimum basis points
    fixed side early exit fee basis points
    fee basis points
    fee receiver (Saffron Governance)

Vaults handle deposits from users and execute the accounting for their gains and losses. The adapters handle interactions with the Lido staking platform.

Vaults are deployed from a VaultFactory. The VaultFactory stores bytecode for the vault and adapter types.

## Development

* Requires node 16, we suggest using nvm: `nvm use 16`

1. Create a `.env` file
```bash
cp default.env .env
# update the contents of the .env
```

2. To compile the contracts, use this command:

```bash
yarn build
```

## Test

To test the contracts, use this command:

```bash
yarn test
# or to bail on the first failure
yarn test:bail
```

### Test Coverage

* Executed with [Solidity Coverage](https://www.npmjs.com/package/solidity-coverage)
* If the coverage does not match the one below, please clear out the artifacts and cache folders (`rm -rf cache`, `rm -rf artifacts`). There's a bug that sometimes presents and causes a lot of functionality to be incorrectly reported as not covered

```bash
yarn coverage
---------------------------------------|----------|----------|----------|----------|----------------|
File                                   |  % Stmts | % Branch |  % Funcs |  % Lines |Uncovered Lines |
---------------------------------------|----------|----------|----------|----------|----------------|
 contracts/                            |      100 |    95.43 |      100 |      100 |                |
  AdminLidoAdapter.sol                 |      100 |       90 |      100 |      100 |                |
  Initialized.sol                      |      100 |      100 |      100 |      100 |                |
  LidoAdapter.sol                      |      100 |    98.61 |      100 |      100 |                |
  LidoVault.sol                        |      100 |       94 |      100 |      100 |                |
  NonTransferrableVaultBearerToken.sol |      100 |      100 |      100 |      100 |                |
  VaultBearerToken.sol                 |      100 |      100 |      100 |      100 |                |
  VaultFactory.sol                     |      100 |    96.43 |      100 |      100 |                |
 contracts/interfaces/                 |      100 |      100 |      100 |      100 |                |
  IAdminLidoAdapter.sol                |      100 |      100 |      100 |      100 |                |
  ILido.sol                            |      100 |      100 |      100 |      100 |                |
  ILidoAdapter.sol                     |      100 |      100 |      100 |      100 |                |
  ILidoVault.sol                       |      100 |      100 |      100 |      100 |                |
  ILidoVaultInitializer.sol            |      100 |      100 |      100 |      100 |                |
  IVaultBearerToken.sol                |      100 |      100 |      100 |      100 |                |
---------------------------------------|----------|----------|----------|----------|----------------|
All files                              |      100 |    95.43 |      100 |      100 |                |
---------------------------------------|----------|----------|----------|----------|----------------|
```

### Echidna Fuzzy Testing

* Executed with [Echidna](https://github.com/crytic/echidna)

```bash
# replace `EchidnaLidoVault` with whatever file you want to test
echidna . --contract EchidnaLidoVault --config echidna.config.yaml
```

## Documentation

[Generated docs](docs/index.md)

[Error codes](revert.md)
