![Seaport](img/Mento-banner.png)

[![Foundry][foundry-badge]][foundry]
[![Mento Core CI][ci-badge]][ci-link]

# Mento Core

This repo contains the source code of the core smart contracts for the Mento protocol. The repository is built with foundry which is used for the compilation and testing of the smart contracts.

## What is Mento?

The Mento protocol is a smart contract platform built on the Celo blockchain that enables the creation of stable value digital assets. Stable assets created with Mento can be classified as 'Hybrid stable assets' as they are algorithmic, transparent and backed by a over-collateralized, diversified portfolio of exogenous crypto assets([Mento Reserve](https://reserve.mento.org/)).

## Documentation

- [Protocol Documentation](https://docs.mento.org/mento/mento-protocol/readme)
- [Stability Whitepaper](https://celo.org/papers/stability)

## Getting Started

```bash
# Get the latest code
git clone git@github.com:mento-protocol/mento-core.git

# Change directory to the the newly cloned repo
cd mento-core

# Install dev dependencies with yarn
yarn

# Install submodule dependencies with forge
forge install

# Compile the smart contracts with forge
forge build

# Run all tests with forge
forge test
```

#### Slither

Install slither, if not installed.

```bash
pip3 install slither-analyzer
```

Running slither locally requires you to build only a subset of packages:

```bash
forge clean
forge build --build-info --skip tests
slither . --foundry-ignore-compile
```

If you want to ignore a slither warning run:

```bash
slither . --foundry-ignore-compile --triage-mode
```

For triage mode, in which you can choose to ignore warnings which are added to `slither.db.json`.

[ci-link]: https://github.com/mento-protocol/mento-core/actions/workflows/ci.yml
[ci-badge]: https://github.com/mento-protocol/mento-core/actions/workflows/ci.yml/badge.svg
[foundry]: https://getfoundry.sh/
[foundry-badge]: https://img.shields.io/badge/Built%20with-Foundry-FFDB1C.svg
