
# Superfluid Locking Contract  contest details

- Join [Sherlock Discord](https://discord.gg/MABEWyASkp)
- Submit findings using the issue page in your private contest repo (label issues as med or high)
- [Read for more details](https://docs.sherlock.xyz/audits/watsons)

# Q&A

### Q: On what chains are the smart contracts going to be deployed?
This smart-contract suite is intended to be deployed on the exhaustive list of networks below :
- Ethereum
- Optimism
- BNB Smart Chain
- Gnosis
- Polygon
- Base
- Arbitrum One
- Celo
- Avalanche
- Scroll
- Degen Chain

___

### Q: If you are integrating tokens, are you allowing only whitelisted tokens to work with the codebase or any complying with the standard? Are they assumed to have certain properties, e.g. be non-reentrant? Are there any types of [weird tokens](https://github.com/d-xo/weird-erc20) you want to integrate?
- The project WILL ONLY integrate a Super Token that Superfluid Core Team will deploy.
- The token will be deployed as a plain ERC20 on Ethereum Mainnet, will then be wrapped as SuperToken and bridged to other chains / L2 (as Super Token).  
- The Super Token that will be used with these contracts is similar to the Super Token at the following address :
https://basescan.org/token/0x2112b92A4f6496B7b2f10850857FfA270464d054

- For more info regarding SuperTokens, see https://docs.superfluid.finance/docs/category/super-tokens

___

### Q: Are there any limitations on values set by admins (or other roles) in the codebase, including restrictions on array lengths?
- FluidEPProgramManager.sol :

  - Owner is trusted
  - Signers are trusted
  - `createProgram(...)` :
    - it is assumed that the given `programAdmin` and `signer` are valid and correct addresses
  - `startFunding(...)` :
    - it is assumed that the `FluidTreasury` will have granted allowance to this contract for the desired amount prior to this call
    - it is assumed that users (specifically lockers) will have units in the corresponding program distribution pool prior to this call
    - similarly, when subsidy switch is turned ON, it is assumed that users (specifically lockers) will have units in the corresponding tax distribution pool prior to this call
  - For reference, during the initial phase, the`subsidyFundingRate` will be set to`0`. After some time, the `subsidyFundingRate` will be increased to`500` (i.e. 5%). It is known that the subsidies will only start flowing for the program funded after the rate is updated.
  - `EARLY_PROGRAM_END` is currently set to 7 days. This value could potentially be reduced (to 5 or 3 days).  

- FluidLocker.sol :

  - Each locker is "owned" by an individual account (`lockerOwner`).
  - `UNLOCK_AVAILABLE` will initially be set to `false`. After some time, the beacon proxy will be updated and this valus will be set to `true`.

- FluidLockerFactory.sol :

  - Governor is trusted

- Fontaine.sol

  - `initialize(...)` :
    - it is assumed that users (specifically lockers) will have units in the corresponding tax distribution pool prior to this call.

- StakingRewardController.sol

  - Owner is trusted
  - It is assumed that the correct `lockerFactory` will be set during deployment

___

### Q: Are there any limitations on values set by admins (or other roles) in protocols you integrate with, including restrictions on array lengths?
No.
___

### Q: Is the codebase expected to comply with any specific EIPs?

The codebase is expected to comply with EIP-1271 (simple signature verification).

**Context**:

In order to grant units in Superfluid GDA pools, we rely on a third party tool : Stack (see stack.so).
Stack essentially allocates "points" to users based on onchain activity.
On user requests (i.e. when users need to claim these points that are representing pool units), Stack generate a signature that contains user related details and the amount of units to be claimed.
The contract verifies that the signature is originated from a Stack whitelisted signer and allocate the units once the signature is verified.

___

### Q: Are there any off-chain mechanisms involved in the protocol (e.g., keeper bots, arbitrage bots, etc.)? We assume these mechanisms will not misbehave, delay, or go offline unless otherwise specified.
Yes, as mentioned above, the protocol rely on off-chain signature from a third party to grant GDA pool units to users.

Additionally, Superfluid will run some offchain monitoring system to ensure that programs do not overrun their duration (see `PROGRAM_DURATION` in FluidEPProgramManager). 
Upon detecting an approaching end of a program a transaction call to `stopFunding` will be performed, ensuring that the different streams inherent to a program are closed on time. It is assumed that these calls will be performed on time.
 
___

### Q: What properties/invariants do you want to hold even if breaking them has a low/unknown impact?
- While the boolean `UNLOCK_AVAILABLE` is set to `false`  (initial phase), the sum of all tokens distributed through `FluidEPProgramManager` should be equal to the sum of all tokens inside the `FluidLocker` instances plus the undistributed amount (still held in the `FluidEPProgramManager` contract)

- Token distributed through the `FluidEPProgramManager` should always land in a `FluidLocker` instance (until withdrawn).

- The sum of all the token distributed to `FluidLocker` instance and distributed to the tax distribution pool (if `subsidyFundingRate` is not null) should be equal to the sum of all token distributed through the `FluidEPProgramManager`.

___

### Q: Please discuss any design choices you made.
- `FluidEPProgramManager.stopFunding` function permissionless.

  - This ensure that any actor is able to close the streams to the program pool and the tax pool when a program reaches its completion. This is important as funds from multiple program will be held by the `FluidEPProgramManager` contract and stream will not necessarily be liquidated if there are still funds to provision them.
  - Superfluid will necessarily monitor the program and their end date but having this method permissionless also allow external actors to participate in the healthiness of the protocol.

- `FluidEPProgramManager.stopFunding` can create dust.

  - Superfluid streams involve flow rates which imply calculation inaccuracy due to Solidity lack of floating points number supports. This means that when stopping program (although early stops compensation are in place) some dust loss may incur to users. The design accepts and acknowledge this fact it would require an extra intensive calculation for a very minimal amount of token per user (less than a gwei in order of magnitude). These funds will not necessarily be lost forever as other program will essentially stream these funds at some point in time.

- Stakers units in the tax distribution pools :
    - Due to Superfluid General Distribution Agreement implementation and technical rules, the units held by stakers are downscaled. GDA pool do not allow the total amount of units in a GDA pool to be higher than the flow rate. Therefore, the down scaling factor is 0.01 Token staked is equal 1 units. This imply that staking less that 0.01 Token will not yield any returns.

___

### Q: Please provide links to previous audits (if any).
N/A
___

### Q: Please list any relevant protocol resources.
Superfluid Docs : 
https://docs.superfluid.finance/

Superfluid SuperToken Docs :
https://docs.superfluid.finance/docs/protocol/super-tokens/overview

Superfluid GDA Docs :
https://docs.superfluid.finance/docs/protocol/distributions/overview


___

### Q: Additional audit information.
It is suggested to look carefully at below functions :

- In `FluidEPProgramManager`:

  - `startFunding`
  - `stopFunding`
  - `cancelProgram`

- In `FluidLocker` :
  - `unlock`

It is also suggested to ensure that the Token cannot be extracted from `FluidLocker`  instances in any other way than the `FluidLocker.unlock`  function (it can be assumed that beacon upgrades will not offer such possibility). 
___



# Audit scope


[fluid @ 8aa3402429c5e994abfde319d95c8115dc00c1ed](https://github.com/superfluid-finance/fluid/tree/8aa3402429c5e994abfde319d95c8115dc00c1ed)
- [fluid/packages/contracts/src/EPProgramManager.sol](fluid/packages/contracts/src/EPProgramManager.sol)
- [fluid/packages/contracts/src/FluidEPProgramManager.sol](fluid/packages/contracts/src/FluidEPProgramManager.sol)
- [fluid/packages/contracts/src/FluidLocker.sol](fluid/packages/contracts/src/FluidLocker.sol)
- [fluid/packages/contracts/src/FluidLockerFactory.sol](fluid/packages/contracts/src/FluidLockerFactory.sol)
- [fluid/packages/contracts/src/FluidToken.sol](fluid/packages/contracts/src/FluidToken.sol)
- [fluid/packages/contracts/src/Fontaine.sol](fluid/packages/contracts/src/Fontaine.sol)
- [fluid/packages/contracts/src/StakingRewardController.sol](fluid/packages/contracts/src/StakingRewardController.sol)

[protocol-monorepo @ c8795f8db446761279fa4a8aee0a48f2eb374d52](https://github.com/superfluid-finance/protocol-monorepo/tree/c8795f8db446761279fa4a8aee0a48f2eb374d52)
- [protocol-monorepo/packages/ethereum-contracts/contracts/apps/SuperTokenV1Library.sol](protocol-monorepo/packages/ethereum-contracts/contracts/apps/SuperTokenV1Library.sol)




[fluid @ 8aa3402429c5e994abfde319d95c8115dc00c1ed](https://github.com/superfluid-finance/fluid/tree/8aa3402429c5e994abfde319d95c8115dc00c1ed)
- [fluid/packages/contracts/src/EPProgramManager.sol](fluid/packages/contracts/src/EPProgramManager.sol)
- [fluid/packages/contracts/src/FluidEPProgramManager.sol](fluid/packages/contracts/src/FluidEPProgramManager.sol)
- [fluid/packages/contracts/src/FluidLocker.sol](fluid/packages/contracts/src/FluidLocker.sol)
- [fluid/packages/contracts/src/FluidLockerFactory.sol](fluid/packages/contracts/src/FluidLockerFactory.sol)
- [fluid/packages/contracts/src/FluidToken.sol](fluid/packages/contracts/src/FluidToken.sol)
- [fluid/packages/contracts/src/Fontaine.sol](fluid/packages/contracts/src/Fontaine.sol)
- [fluid/packages/contracts/src/StakingRewardController.sol](fluid/packages/contracts/src/StakingRewardController.sol)

