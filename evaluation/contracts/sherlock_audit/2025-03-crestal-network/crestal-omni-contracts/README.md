# crestal-omni-contracts
Crestal Omnichain Smart Contracts

## Requirements

### Dependencies

Install [Foundry](https://book.getfoundry.sh/getting-started/installation).

Update Foundry to latest version:
```bash
foundryup
```
> Add `~/.foundry/bin` to `$PATH` if command is not found.

Install OpenZeppelin contracts:
```bash
forge install foundry-rs/forge-std
forge install OpenZeppelin/openzeppelin-foundry-upgrades
forge install OpenZeppelin/openzeppelin-contracts-upgradeable
```

Update dependencies (for an existing project):
```bash
forge update
```

Install [OpenZeppelin Upgrades CLI](https://docs.openzeppelin.com/upgrades-plugins/1.x/api-core):
```bash
npm install @openzeppelin/upgrades-core
```

### Setup (one-time)

This only needs to be [set up](https://docs.openzeppelin.com/upgrades-plugins/1.x/foundry-upgrades) during initialization for each Foundry project. Keeping it as reference.

### Tools

(Optional) Recommend installing [solc-select](https://github.com/crytic/solc-select) to manage Solidity compiler versions.

(Optional) Security tools

#### Dependencies

Install [pipx](https://github.com/pypa/pipx?tab=readme-ov-file#install-pipx).

#### Slither

```bash
pipx install slither-analyzer
```

#### Mythril
```bash
pipx install mythril
```

## Usage

### Development

Format source files:
```bash
forge fmt
```

Build contracts:
```bash
forge clean
forge build
```

Test contracts:
```bash
forge test
```

Security checks:
```bash
make slither
make mythril
```

Generate abi (for external access):
```bash
make abi
```

### Deployment

#### Local

Start local node in a separate window:
```bash
anvil
```

Deploy (copy private key from `anvil` output):
```bash
PRIVATE_KEY=xxx make deploy
```

Upgrade to latest version (copy proxy address from deployed output):
```bash
PRIVATE_KEY=xxx PROXY_ADDRESS=xxx make upgrade
```

Upgrade to particular version, one V+ at a time (copy proxy address from deployed output):
```bash
PRIVATE_KEY=xxx PROXY_ADDRESS=xxx UPGRADE_TO=Vx make upgrade
```

Sanity check:
```bash
PROXY_ADDRESS=xxx make check
```

#### Live Networks

Deploy:
```bash
PRIVATE_KEY=xxx RPC_URL=https://xxx make deploy
```

Upgrade (copy proxy address from deployed output):
```bash
PRIVATE_KEY=xxx RPC_URL=https://xxx PROXY_ADDRESS=xxx make upgrade
```

Sanity check:
```bash
PROXY_ADDRESS=xxx RPC_URL=https://xxx make check
```

#### Tips

Put all env-related variables in a per chain `.env.chain` file then use the following:
```bash
ENV_FILE=.env.chain make deploy
```
