# Seamless Leverage Tokens

The Seamless Leverage Token protocol is comprised of smart contracts that provide ERC20 tokenized representations of any leveraged position between 2 assets on a blockchain. If a lending market exists between 2 assets then a Leverage Token can be permissionlessly created for the leveraged position.

## Getting Started

This project uses [Foundry](https://book.getfoundry.sh/) for development.

Install dependencies: `forge install`

Run forge tests: `forge test`

Build: `forge build`

## Architecture

The protocol is composed of modular contracts that abstract away the complexities of managing leveraged positions. These components are designed to be flexible, composable, and upgradeable, making it easy to integrate new lending markets, rebalancing strategies, or token configurations.

## Permissionless Creation
Leverage Tokens within the Seamless Protocol are designed to be fully permissionless. This means:

- **Anyone can create a new Leverage Token:** There are no restrictions or allowlists for token creation. Users, developers, or even external parties can deploy new Leverage Tokens at any time.

- **Potential for Malicious Tokens:** Because creation is open, it is possible for malicious actors to deploy Leverage Tokens with configurations or parameters intended to deceive users or exploit vulnerabilities. Participants should exercise caution and verify the legitimacy and configuration of any Leverage Token before interacting with it.

## Inherited Risks from Underlying Platforms
Leverage Tokens are built to interact with various DeFi protocols through adapters. As a result:

- **Exposure to Adapter Risks:** Each Leverage Token inherits the technical and economic risks of any underlying lending or DeFi platform it integrates with via adapters. If an adapter connects to a platform with vulnerabilities, those risks are passed through to the Leverage Token and its users.

- **No Isolation from Platform Failures:** Issues such as smart contract bugs, oracle failures, insolvency, or governance attacks on underlying platforms can directly impact the safety and value of the Leverage Token.

- **Dynamic Risk Profile:** The risk profile of each Leverage Token may change over time as underlying protocols are upgraded, attacked, or experience market volatility.

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

## Deployed Contracts (Base)

### Core

- LeverageToken implementation: `0x057A2a1CC13A9Af430976af912A27A05DE537673`
- LeverageToken factory proxy: `0xE0b2e40EDeb53B96C923381509a25a615c1Abe57`
- LeverageToken factory implementation: `0x057a2a1cc13a9af430976af912a27a05de537673`
- LeverageManager implementation: `0xeb0221bf6cdaa74c94129771d5b0c9a994bb2b7c`
- LeverageManager proxy: `0x38Ba21C6Bf31dF1b1798FCEd07B4e9b07C5ec3a8`
- MorphoLendingAdapterFactory: `0xDd33419F0c01879a23051edbcdA997A0f9E68e61`
- MorphoLendingAdapter implementation: `0x585cc1c8AF5C8aD79C64ac66D264590A3Ff65C51`
- RebalanceAdapter implementation: `0xD923b2522E1f369e207d151cFE6A1BCd8EC24912`

### Periphery

- SwapAdapter: `0xfdF71E2c2B34cD1f8bb852e7c55E8b926Fe38942`
- LeverageRouter: `0xDbA92fC3dc10a17b96b6E807a908155C389A887C`
- PricingAdapter: `0x94E337608d6F3AFD4E5e1dE9Aaa0804e3ab92C4c`
