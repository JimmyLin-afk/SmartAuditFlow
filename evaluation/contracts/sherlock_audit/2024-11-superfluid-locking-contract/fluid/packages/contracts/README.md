## Address Registry :

Latest deployed contract address

### BASE SEPOLIA :

| Contract              | Address                                    | Explorer                                                                        |
| --------------------- | ------------------------------------------ | ------------------------------------------------------------------------------- |
| FluidToken            | 0x3A193aC8FcaCCDa817c174D04081C105154a8441 | https://sepolia.basescan.org/address/0x3A193aC8FcaCCDa817c174D04081C105154a8441 |
| FluidEPProgramManager | 0xe175053257B6A0Fe7EAE537499Ba4465065DdD88 | https://sepolia.basescan.org/address/0xe175053257B6A0Fe7EAE537499Ba4465065DdD88 |
| PenaltyManager        | 0xA6c9DCED06c378a0548D7d4a880e2036A03696CE | https://sepolia.basescan.org/address/0xA6c9DCED06c378a0548D7d4a880e2036A03696CE |
| FluidLocker (Logic)   | 0x783Fcb21077a1b243c915ed639F9c7f5bE9111BB | https://sepolia.basescan.org/address/0x783Fcb21077a1b243c915ed639F9c7f5bE9111BB |
| Fontaine (Logic)      | 0x9a2D08fE4fe75D33990085A436B4aB215Bf9e524 | https://sepolia.basescan.org/address/0x9a2D08fE4fe75D33990085A436B4aB215Bf9e524 |
| FluidLockerFactory    | 0x8DaF7BF1a2052B6BDA0eC46619855Cec77DfbC76 | https://sepolia.basescan.org/address/0x8DaF7BF1a2052B6BDA0eC46619855Cec77DfbC76 |

## Test Coverage :

Current test coverage is as follow :

| File                            | % Lines         | % Statements    | % Branches      | % Funcs         |
| ------------------------------- | --------------- | --------------- | --------------- | --------------- |
| src/EPProgramManager.sol        | 100.00% (42/42) | 100.00% (56/56) | 100.00% (14/14) | 100.00% (13/13) |
| src/FluidEPProgramManager.sol   | 98.72% (77/78)  | 96.97% (96/99)  | 90.00% (18/20)  | 92.86% (13/14)  |
| src/FluidLocker.sol             | 100.00% (75/75) | 97.92% (94/96)  | 84.62% (11/13)  | 100.00% (21/21) |
| src/FluidLockerFactory.sol      | 85.00% (17/20)  | 80.95% (17/21)  | 0.00% (0/2)     | 81.82% (9/11)   |
| src/Fontaine.sol                | 100.00% (6/6)   | 85.71% (6/7)    | 0.00% (0/1)     | 100.00% (2/2)   |
| src/StakingRewardController.sol | 92.86% (13/14)  | 93.75% (15/16)  | 100.00% (2/2)   | 87.50% (7/8)    |

## Foundry

**Foundry is a blazing fast, portable and modular toolkit for Ethereum application development written in Rust.**

Foundry consists of:

- **Forge**: Ethereum testing framework (like Truffle, Hardhat and DappTools).
- **Cast**: Swiss army knife for interacting with EVM smart contracts, sending transactions and getting chain data.
- **Anvil**: Local Ethereum node, akin to Ganache, Hardhat Network.
- **Chisel**: Fast, utilitarian, and verbose solidity REPL.

## Documentation

https://book.getfoundry.sh/

## Usage

### Build

```shell
$ forge build
```

### Test

```shell
$ forge test
```

### Format

```shell
$ forge fmt
```

### Gas Snapshots

```shell
$ forge snapshot
```

### Anvil

```shell
$ anvil
```

### Deploy

To deploy the contract suite, fill in the `.env` file using `.env.example` as reference.
The sections `Private Keys`, `RPCs`, and `Deployment Settings` must be complete to deploy.

### References

```
HOST_ADDRESS : Superfluid Host address
FLUID_ADDRESS : SuperToken to be distributed
GOVERNOR_ADDRESS : Contract owner address
TREASURY_ADDRESS : Treasury address holding the SuperToken to be distributed
STACK_SIGNER_ADDRESS : Signer address to be verified in order to grant units
PAUSE_FACTORY_LOCKER_CREATION : Whether the Factory allows Lockers to be created or not
FLUID_UNLOCK_STATUS : Whether the Lockers allow the SuperToken to be withdrawn or not
```

```shell
$ forge script script/Deploy.s.sol:DeployScript --ffi --rpc-url $BASE_SEPOLIA_RPC_URL --broadcast
```
