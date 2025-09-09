# Seamless Leverage Tokens

The Seamless Leverage Token protocol is comprised of smart contracts that provide ERC20 tokenized representations of any leveraged position between 2 assets on a blockchain. If a lending market exists between 2 assets then a Leverage Token can be permissionlessly created for the leveraged position.

## Getting Started

This project uses [Foundry](https://book.getfoundry.sh/) for development.

Install dependencies: `forge install`

Run forge tests: `forge test`

Build: `forge build`

## Architecture

The protocol is composed of modular contracts that abstract away the complexities of managing leveraged positions. These components are designed to be flexible, composable, and upgradeable, making it easy to integrate new lending markets, rebalancing strategies, or token configurations.

### LeverageManager

The LeverageManager is the core singleton contract used for creating and maintaining leverage positions involving two assets: one serving as collateral and the other as debt. It does not handle any swapping or other asset-trading logic; instead, it only relies on price data (i.e., the relative price of the two assets) to validate and oversee the leverage position.

### LeverageToken

LeverageTokens are ERC20 tokens that represent a share of a specific leveraged position. These are deployed permissionlessly via the LeverageManager.

Each LeverageToken has the following properties:

- ERC20-compliant: Balances represent proportional ownership of a leveraged position.
- Immutable configuration: Once deployed, the position’s parameters cannot be modified.
- Configured with adapters:
  - A LendingAdapter, which handles interaction with the external lending protocol.
  - A RebalanceAdapter, which defines rebalance conditions and logic.

### LendingAdapter

LendingAdapters are periphery contracts responsible for interfacing with external lending protocols (e.g., Morpho). Each LeverageToken is configured with a LendingAdapter at the time of creation and must implement the minimal interface `ILendingAdapter`.

Key responsibilities:

- Own the underlying debt and collateral positions.
- Abstract lending protocol-specific logic behind a standard interface for the LeverageManager to consume to manage the positions.

### RebalanceAdapter

RebalanceAdapters are periphery contracts that facilitate rebalance actions and rules for LeverageTokens. Each LeverageToken is configured with a RebalanceAdapter at the time of creation. The RebalanceAdapter must implement the minimal interface `IRebalanceAdapterBase` which includes the following:

- isEligibleForRebalance: Returns true if the token meets conditions to trigger a rebalance
- isStateValidAfterRebalance: Returns true if the LeverageToken’s state is valid after a rebalance
- getLeverageTokenInitialCollateralRatio: Returns the collateral ratio that should be used when the LeverageToken is empty / total shares == 0
- postLeverageTokenCreation: Post LeverageToken creation hook. Executed during LeverageToken creation in LeverageManager

## Upgrades

All contracts are immutable except LeverageManager, LeverageToken, RebalanceAdapter, and BeaconProxyFactory. During deployment and upgrade of these contracts, OZ's foundry-upgrades plugin is used. For safe upgrades, the custom directive `@custom:oz-upgrades-from <reference contract>` must be added on the new implementation. More details on how to use OZ's upgrades plugin can be found [here](https://docs.openzeppelin.com/upgrades-plugins/foundry-upgrades#upgrade_a_proxy_or_beacon).

## Audits

Audits reports can be found in the [audits](./audits/) folder.

## Security

To report any issues, please contact security@seamlessprotocol.com.
