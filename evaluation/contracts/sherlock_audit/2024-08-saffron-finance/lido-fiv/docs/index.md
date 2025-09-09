# Solidity API

## Initialized

This contract extends the Initializable contract with a modifier that requires the contract be initialized

### isInitialized

```solidity
modifier isInitialized()
```

This modifier checks if the contract's initializer function has been called

### initialized

```solidity
function initialized() external view returns (bool)
```

Returns true if the initializer has been called

## LidoAdapter

### id

```solidity
uint256 id
```

Adapter ID set by the factory

### lido

```solidity
contract ILido lido
```

Lido contract

### lidoWithdrawalQueue

```solidity
contract ILidoWithdrawalQueueERC721 lidoWithdrawalQueue
```

Lido withdrawal queue contract

### admin

```solidity
address admin
```

Address that can call admin actions

### vault

```solidity
address vault
```

Vault address

### LIDO_ERROR_TOLERANCE_ETH

```solidity
uint256 LIDO_ERROR_TOLERANCE_ETH
```

ETH diff tolerance either for the expected deposit or withdraw from Lido - some rounding errors of a few wei seem to occur

### VaultSet

```solidity
event VaultSet(address vault, address setter)
```

Emitted when the vault contract address is updated

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| vault | address | New vault address |
| setter | address | Address of setter |

### AdminSet

```solidity
event AdminSet(address admin, address setter)
```

Emitted when the admin address is updated

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| admin | address | New admin address |
| setter | address | Address of setter |

### FundsStaked

```solidity
event FundsStaked(uint256 amount, uint256 shares, address user)
```

Emitted when ETH is submitted to the Lido platform for staking

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| amount | uint256 | ETH amount staked |
| shares | uint256 | Amount of shares received for the ETH staked |
| user | address | User address |

### FundsTransferred

```solidity
event FundsTransferred(uint256 amount, address recipient)
```

Emitted when withdrawn funds are transferred out of this contract

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| amount | uint256 | ETH amount transferred |
| recipient | address | Address of receiver of the funds |

### TransferredStETH

```solidity
event TransferredStETH(uint256 amount, address recipient)
```

Emitted when stETH funds are transferred out of this contract

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| amount | uint256 | ETH amount transferred |
| recipient | address | Address of receiver of the stETH |

### TransferredWithdrawalERC721

```solidity
event TransferredWithdrawalERC721(uint256 requestId, address recipient)
```

Emitted when a Lido withdrawal request ERC721 is transferred out of this contract

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| requestId | uint256 | Request id of the Lido withdrawal request |
| recipient | address | Address of receiver of the ERC721 |

### WithdrawalRequested

```solidity
event WithdrawalRequested(uint256 amount, uint256[] requestIds, address user)
```

Emitted when stakings funds are requested to be withdrawn from Lido

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| amount | uint256 | ETH amount requested to be withdrawn |
| requestIds | uint256[] | Request ids of the Lido withdrawal requests |
| user | address | User address |

### WithdrawalClaimed

```solidity
event WithdrawalClaimed(uint256 amount, uint256[] requestIds, address user)
```

Emitted when a Lido fixed withdrawal is claimed and the ETH is transferred to this contract

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| amount | uint256 | ETH amount withdrawn |
| requestIds | uint256[] | Request ids of the Lido withdrawal requests |
| user | address | User address |

### onlyVault

```solidity
modifier onlyVault()
```

This modifier checks that the sender is the vault address

### onlyAdmin

```solidity
modifier onlyAdmin()
```

This modifier checks that the sender is the admin address

### constructor

```solidity
constructor() public
```

_Vault factory will always be the owner_

### initialize

```solidity
function initialize(uint256 _id, address _admin) external
```

Initializes the adapter

_Make sure this is only callable by the vault creator_

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| _id | uint256 |  |
| _admin | address |  |

### receive

```solidity
receive() external payable
```

Function to receive Ether. msg.data must be empty

### renounceOwnership

```solidity
function renounceOwnership() public
```

Disable ownership renunciation

### owner

```solidity
function owner() public view returns (address)
```

Owner of contract - set in the constructor to the contract creator

### minStETHWithdrawalAmount

```solidity
function minStETHWithdrawalAmount() external pure returns (uint256)
```

The minimum amount of stETH that can be withdrawn from Lido

### maxStETHWithdrawalAmount

```solidity
function maxStETHWithdrawalAmount() external pure returns (uint256)
```

The maximum amount of stETH that can be withdrawn from Lido in a single request
If desired amount to withdraw is greater than the value it has to be broken up into multiple requests

### setAdmin

```solidity
function setAdmin(address _admin) external
```

Updates the admin address

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| _admin | address | Admin address |

### setVault

```solidity
function setVault(address _vault) public virtual
```

Updates the vault contract address

_Make sure this is only callable by the vault creator_

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| _vault | address |  |

### stakingBalance

```solidity
function stakingBalance() public view returns (uint256)
```

Balance of stETH tokens, which is dynamic based off of staking rewards https://docs.lido.fi/contracts/lido#rebase

### stakeFunds

```solidity
function stakeFunds(address user) external payable returns (uint256)
```

Stakes ETH on Lido and returns stETH

_Make sure this is only callable by the vault_

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| user | address | Address of the user that is staking the funds |

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | uint256 | shares amount of Lido shares issued for the staked ETH |

### transferStETH

```solidity
function transferStETH(address recipient, uint256 amount) external
```

Transfers the stETH balance to another address

_Make sure this is only callable by the vault_

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| recipient | address | Address to transfer stETH to |
| amount | uint256 | Amount of stETH to transfer |

### transferWithdrawalERC721

```solidity
function transferWithdrawalERC721(address recipient, uint256 requestId) external
```

Transfers a Lido ERC721 representing a staking withdrawal request

_Make sure this is only callable by the vault_

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| recipient | address | Address to transfer to |
| requestId | uint256 | Lido withdrawal request id |

### transferWithdrawnFunds

```solidity
function transferWithdrawnFunds(address recipient, uint256 amount) external
```

Transfers ETH unstaked from Lido

_Make sure this is only callable by the vault_

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| recipient | address | Address to transfer to |
| amount | uint256 | Amount of ETH |

### requestWithdrawViaShares

```solidity
function requestWithdrawViaShares(address user, uint256 shares) external returns (uint256[])
```

Request a withdrawal on Lido to exchange stETH for ETH via shares amount

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| user | address | Address of the user that is requesting the withdraw |
| shares | uint256 | Amount of staking shares to withdraw. Will be used to calculate the actual stETH amount to withdraw |

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | uint256[] | requestIds Ids of the withdrawal requests |

### requestEntireBalanceWithdraw

```solidity
function requestEntireBalanceWithdraw(address user) external returns (uint256[])
```

Request a withdrawal of the entire stETH balance of the contract

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| user | address | Address of the user that is requesting the withdraw |

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | uint256[] | requestIds Ids of the withdrawal requests |

### requestWithdrawViaETH

```solidity
function requestWithdrawViaETH(address user, uint256 stETHAmount) public returns (uint256[])
```

Request a withdrawal on Lido to exchange stETH for ETH via stETH amount

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| user | address | Address of the user that is requesting the withdraw |
| stETHAmount | uint256 | Amount of stETH to withdraw |

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | uint256[] | requestIds Ids of the withdrawal requests |

### _requestWithdraw

```solidity
function _requestWithdraw(address user, uint256 stETHAmount) internal returns (uint256[])
```

Request a withdrawal on Lido to exchange stETH for ETH

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| user | address |  |
| stETHAmount | uint256 | amount of stETH to withdraw |

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | uint256[] | requestIds Ids of the withdrawal requests |

### claimWithdrawals

```solidity
function claimWithdrawals(address user, uint256[] requestIds) external returns (uint256)
```

Claim a withdrawal request on Lido once it has been finalized. This will transfer ETH to the contract if successful

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| user | address | Address of the user that is claiming the withdraw |
| requestIds | uint256[] | Withdrawal request ids to claim |

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | uint256 | amount of ETH withdrawn |

### _claimWithdrawals

```solidity
function _claimWithdrawals(address user, uint256[] requestIds) internal returns (uint256)
```

Claim a withdrawal request on Lido once it has been finalized

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| user | address |  |
| requestIds | uint256[] | Ids of the withdrawal requests |

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | uint256 | amount of ETH withdrawn |

### calculateWithdrawals

```solidity
function calculateWithdrawals(uint256 stETHAmount) internal pure returns (uint256[])
```

Helper function to calculate how many requests will be needed to unstake from Lido due to the maximum withdrawal amount per request

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| stETHAmount | uint256 | stETH amount requested to be withdrawn |

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | uint256[] | withdrawalAmounts an array of the amounts to withdraw in each withdrawal request to Lido |

## LidoVault

### id

```solidity
uint256 id
```

Vault ID set by the factory

### duration

```solidity
uint256 duration
```

Length of the earning period of the vault in seconds

### startTime

```solidity
uint256 startTime
```

Start of vault

_Calculated when vault starts via block.timestamp_

### endTime

```solidity
uint256 endTime
```

End of duration

_Calculated when vault starts via (block.timestamp + duration)_

### fixedSideCapacity

```solidity
uint256 fixedSideCapacity
```

Total capacity of the fixed side

### variableSideCapacity

```solidity
uint256 variableSideCapacity
```

Total capacity of the variable side

### fixedPremiumBps

```solidity
uint256 fixedPremiumBps
```

Upfront premium of the fixed principal fixed side depositors receive from variable depositors

### protocolFeeBps

```solidity
uint256 protocolFeeBps
```

Saffron protocol fee in basis points

### appliedProtocolFee

```solidity
uint256 appliedProtocolFee
```

ETH amount that tracks the Saffron protocol fee applied to withdrawn Lido staking earnings. It is reset when the fee reciever withdraws

### totalProtocolFee

```solidity
uint256 totalProtocolFee
```

ETH amount that tracks the total Saffron protocol fee from variable withdrawal staking earnings while the vault is still ongoing

### protocolFeeReceiver

```solidity
address protocolFeeReceiver
```

Address that collects the Saffron protocol fee

### earlyExitFeeBps

```solidity
uint256 earlyExitFeeBps
```

Penalty fee in basis points for fixed side early withdrawals that is quadratically scaled based off of the amount of time that has elapsed since the vault started

### admin

```solidity
address admin
```

Address that can call admin actions

### minimumDepositAmount

```solidity
uint256 minimumDepositAmount
```

Minimum amount of ETH that can be deposited for variable or fixed side users

### fixedClaimToken

```solidity
contract IVaultBearerToken fixedClaimToken
```

ERC20 bearer token that entitles owner to a portion of the fixed side bearer tokens and the variable side premium payment
It represents the amount of StETH shares generated from Lido staking submit on fixed side deposits

_If the vault hasn't started, this is used to return the fixed side deposit_

### fixedBearerToken

```solidity
contract IVaultBearerToken fixedBearerToken
```

ERC20 bearer token that entitles owner to a portion of the fixed side deposits after the vault has ended and the ETH has been unstaked from Lido

### fixedETHDepositToken

```solidity
contract IVaultBearerToken fixedETHDepositToken
```

ERC20 bearer token that tracks fixed user's ETH deposits in order to determine when the fixed capacity is reached

### variableBearerToken

```solidity
contract IVaultBearerToken variableBearerToken
```

ERC20 bearer token that entitles owner to a portion of the vault earnings

### lidoAdapter

```solidity
contract ILidoAdapter lidoAdapter
```

Contract that interacts with Lido to stake / unstake

### feeEarnings

```solidity
uint256 feeEarnings
```

Amount of earnings from fixed side early exit fees

### withdrawnFeeEarnings

```solidity
uint256 withdrawnFeeEarnings
```

Fee earnings that have been withdrawn from the variable side

### withdrawnStakingEarnings

```solidity
uint256 withdrawnStakingEarnings
```

Amount of earnings from Lido Staking in ETH already withdrawn by variable depositors

### vaultEndedWithdrawalRequestIds

```solidity
uint256[] vaultEndedWithdrawalRequestIds
```

Lido withdraw request ids for after the vault has ended

### vaultEndedWithdrawalsFinalized

```solidity
bool vaultEndedWithdrawalsFinalized
```

The vault ended Lido withdrawal requests have been claimed and all the staked Lido ETH has been withdrawn

_True if the remaining Lido ETH funds have been withdrawn from Lido_

### vaultEndedStakingEarnings

```solidity
uint256 vaultEndedStakingEarnings
```

Amount of earnings in ETH from Lido Staking after the vault has ended

### vaultEndedFixedDepositsFunds

```solidity
uint256 vaultEndedFixedDepositsFunds
```

Amount of ETH used to cover the returning of fixed user's initial principal

### userToFixedUpfrontPremium

```solidity
mapping(address => uint256) userToFixedUpfrontPremium
```

Mapping from user addresses to the upfront premium a fixed depositor received from the variable side

### fixedToVaultNotStartedWithdrawalRequestIds

```solidity
mapping(address => uint256[]) fixedToVaultNotStartedWithdrawalRequestIds
```

Mapping from user addresses to their withdrawal request ids before the vault has started

### WithdrawalRequest

```solidity
struct WithdrawalRequest {
  uint256 timestamp;
  uint256[] requestIds;
}
```

### fixedToVaultOngoingWithdrawalRequestIds

```solidity
mapping(address => struct LidoVault.WithdrawalRequest) fixedToVaultOngoingWithdrawalRequestIds
```

Mapping from fixed side user addresses to their withdrawal request ids after the vault has started but before the vault has ended

### getFixedOngoingWithdrawalRequestTimestamp

```solidity
function getFixedOngoingWithdrawalRequestTimestamp(address user) public view returns (uint256)
```

Add getter functions for each field in the struct b/c the compiler doesn't know how to construct the "free" getter that usually comes with public

### getFixedOngoingWithdrawalRequestIds

```solidity
function getFixedOngoingWithdrawalRequestIds(address user) public view returns (uint256[])
```

### fixedOngoingWithdrawalUsers

```solidity
address[] fixedOngoingWithdrawalUsers
```

Store array of all fixed users with pending ongoing withdrawals to iterate through fixedToVaultOngoingWithdrawalRequestIds mapping if the vault ends

### fixedToPendingWithdrawalAmount

```solidity
mapping(address => uint256) fixedToPendingWithdrawalAmount
```

When the vault ends ongoing fixed withdrawals may be claimed by another user
this stores the amount they should be given when the user eventually withdraws

### variableToVaultOngoingWithdrawalRequestIds

```solidity
mapping(address => uint256[]) variableToVaultOngoingWithdrawalRequestIds
```

Mapping from variable side user addresses to their withdrawal request ids after the vault has started but before the vault has ended

### variableToWithdrawnStakingEarnings

```solidity
mapping(address => uint256) variableToWithdrawnStakingEarnings
```

Mapping from variable side user addresses to their total withdrawn staking earnings amount right after their Lido withdrawal request is submitted

### variableToWithdrawnFees

```solidity
mapping(address => uint256) variableToWithdrawnFees
```

Mapping from variable side user addresses to their total withdrawn fees amount right after their Lido withdrawal request is claimed

### VaultInitialized

```solidity
event VaultInitialized(uint256 vaultId, uint256 duration, uint256 variableSideCapacity, uint256 fixedSideCapacity, uint256 fixedPremiumBps, uint256 earlyExitFeeBps, uint256 protocolFeeBps, address protocolFeeReceiver, address admin, address adapter)
```

Emitted when a new vault is initialized

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| vaultId | uint256 | ID of vault |
| duration | uint256 | How long the vault will be locked once started, in seconds |
| variableSideCapacity | uint256 | Maximum capacity of variable side |
| fixedSideCapacity | uint256 | Maximum capacity of fixed side |
| fixedPremiumBps | uint256 | Fixed depositor premium in basis points |
| earlyExitFeeBps | uint256 | Fixed depositor early exit fee in basis points |
| protocolFeeBps | uint256 | Protocol fee in basis points |
| protocolFeeReceiver | address | Address that collects protocol fee |
| admin | address | Address of vault admin |
| adapter | address | Address of adapter |

### AdminSet

```solidity
event AdminSet(address admin, address setter)
```

Emitted when the admin address is updated

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| admin | address | New admin address |
| setter | address | Address of setter |

### ProtocolFeeReceiverSet

```solidity
event ProtocolFeeReceiverSet(address protocolFeeReceiver, address setter)
```

Emitted when the protocol fee receiver is updated

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| protocolFeeReceiver | address | New protocol fee receiver |
| setter | address | Address of setter |

### MinimumDepositAmountSet

```solidity
event MinimumDepositAmountSet(uint256 minimumDepositAmount, address setter)
```

Emitted when the minimum deposit amount is updated

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| minimumDepositAmount | uint256 | New minimum deposit amount |
| setter | address | Address of setter |

### VariableFundsDeposited

```solidity
event VariableFundsDeposited(uint256 amount, address user)
```

Emitted when variable funds are deposited into the vault

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| amount | uint256 | ETH amount deposited |
| user | address | Address of user |

### FixedFundsDeposited

```solidity
event FixedFundsDeposited(uint256 amount, uint256 shares, address user)
```

Emitted when fixed funds are deposited into the vault

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| amount | uint256 | ETH amount deposited |
| shares | uint256 | stETH shares issued for Lido submit |
| user | address | Address of user |

### FixedPremiumClaimed

```solidity
event FixedPremiumClaimed(uint256 premium, uint256 shares, address user)
```

Emitted when a fixed user claims their upfront premium

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| premium | uint256 | Fixed upfront premium transferred to the fixed user |
| shares | uint256 | stETH shares claimed |
| user | address | Address of user |

### VariableFundsWithdrawn

```solidity
event VariableFundsWithdrawn(uint256 amount, address user, bool isStarted, bool isEnded)
```

Emitted when variable funds are withdrawn from the vault

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| amount | uint256 | Amount withdrawn |
| user | address | Address of user |
| isStarted | bool | Indicates whether the vault has started when the funds were withdrawn |
| isEnded | bool | Indicates whether the vault has ended when the funds were withdrawn |

### FixedFundsWithdrawn

```solidity
event FixedFundsWithdrawn(uint256 amount, address user, bool isStarted, bool isEnded)
```

Emitted when fixed funds are withdrawn from the vault

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| amount | uint256 | Amount withdrawn |
| user | address | Address of user |
| isStarted | bool | Indicates whether the vault has started when the funds were withdrawn |
| isEnded | bool | Indicates whether the vault has ended when the funds were withdrawn |

### VaultStarted

```solidity
event VaultStarted(uint256 timeStarted, address user)
```

Emitted when the vault capacities have been met and the vault has been moved into the started phase

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| timeStarted | uint256 | Time the vault started |
| user | address | Address of user that triggered the start of the vault |

### VaultEnded

```solidity
event VaultEnded(uint256 timeEnded, address user)
```

Emitted when the vault has passed its end time and moved into the ended phase

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| timeEnded | uint256 | Time the vault ended |
| user | address | Address of user that triggered the end of the vault |

### LidoWithdrawalRequested

```solidity
event LidoWithdrawalRequested(address user, uint256[] requestIds, uint256 side, bool isStarted, bool isEnded)
```

Emitted when stakings funds are requested to be withdrawn from Lido

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| user | address | Address of user |
| requestIds | uint256[] | Request ids of the Lido withdrawal requests |
| side | uint256 | Indicates fixed or variable side requested |
| isStarted | bool | Indicates whether the vault has started when the funds were requested to be withdrawn |
| isEnded | bool | Indicates whether the vault has ended when the funds were requested to be withdrawn |

### LidoWithdrawalFinalized

```solidity
event LidoWithdrawalFinalized(address user, uint256[] requestIds, uint256 side, bool isStarted, bool isEnded)
```

Emitted when a Lido withdrawal is claimed and the ETH is transferred to the adapter contract

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| user | address | Address of user |
| requestIds | uint256[] | Request ids of the Lido withdrawal requests |
| side | uint256 | Indicates fixed or variable side requested |
| isStarted | bool | Indicates whether the vault has started when the funds were requested to be withdrawn |
| isEnded | bool | Indicates whether the vault has ended when the funds were submitted for withdrawal |

### onlyAdmin

```solidity
modifier onlyAdmin()
```

This modifier checks that the sender is the admin address

### constructor

```solidity
constructor() public
```

_Vault factory will always be the owner_

### initialize

```solidity
function initialize(struct ILidoVaultInitializer.InitializationParams params) external
```

Initializes the vault

_make sure this is only callable by the contract creator aka the vault factory_

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| params | struct ILidoVaultInitializer.InitializationParams | Initialization parameters |

### receive

```solidity
receive() external payable
```

Function to receive Ether. msg.data must be empty

### renounceOwnership

```solidity
function renounceOwnership() public
```

Disable ownership renunciation

### isStarted

```solidity
function isStarted() public view returns (bool)
```

True if the vault has started

### isEnded

```solidity
function isEnded() public view returns (bool)
```

True if the vault has ended
or the admin has settled all debts thus effectively ending the vault

### setMinimumDepositAmount

```solidity
function setMinimumDepositAmount(uint256 _minimumDepositAmount) external
```

Updates the minimum deposit amount

### setProtocolFeeReceiver

```solidity
function setProtocolFeeReceiver(address _protocolFeeReceiver) external
```

Updates the protocol fee receiver

### setAdmin

```solidity
function setAdmin(address _admin) external
```

Updates the admin

### deposit

```solidity
function deposit(uint256 side) external payable
```

Deposit ETH into the vault

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| side | uint256 | ID of side to deposit into |

### claimFixedPremium

```solidity
function claimFixedPremium() external
```

Claim fixed side bearer tokens with fixed side claim tokens
and transfer to the user their upfront premium from variable deposits

### withdraw

```solidity
function withdraw(uint256 side) external
```

Withdraw from the vault

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| side | uint256 | ID of side to withdraw from |

### finalizeVaultNotStartedFixedWithdrawals

```solidity
function finalizeVaultNotStartedFixedWithdrawals() external
```

Finalize a fixed withdrawal that was requested before the vault started

### finalizeVaultOngoingFixedWithdrawals

```solidity
function finalizeVaultOngoingFixedWithdrawals() external
```

Finalize a fixed withdrawal that was requested after the vault has started

### finalizeVaultOngoingVariableWithdrawals

```solidity
function finalizeVaultOngoingVariableWithdrawals() external
```

Finalize a variable withdrawal that was requested after the vault has started

### finalizeVaultEndedWithdrawals

```solidity
function finalizeVaultEndedWithdrawals(uint256 side) external
```

Finalize the vault ended withdrawals

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| side | uint256 | ID of side |

### claimOngoingFixedWithdrawals

```solidity
function claimOngoingFixedWithdrawals() internal
```

### applyProtocolFee

```solidity
function applyProtocolFee(uint256 stakingEarnings) internal returns (uint256)
```

Helper function to apply Saffron's protocol fee to Lido staking earnings

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| stakingEarnings | uint256 | the amount of ETH staking earnings on Lido |

### vaultEndedWithdraw

```solidity
function vaultEndedWithdraw(uint256 side) internal
```

Helper function to do the accounting for a withdraw after the vault has ended

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| side | uint256 | ID of side |

### fixedLidoSharesTotalSupply

```solidity
function fixedLidoSharesTotalSupply() internal view returns (uint256)
```

Returns all fixed side Lido shares - claimed or unclaimed

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | uint256 | totalSupply Total supply of the fixed bearer and claim tokens |

### claimFixedVaultOngoingWithdrawal

```solidity
function claimFixedVaultOngoingWithdrawal(address user) internal returns (uint256)
```

Helper function to claim a Lido fixed withdrawal that was requested after the vault has started

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| user | address | User that requested the withdrawal |

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | uint256 | amount Amount of ETH withdrawn from Lido after the withdraw has been claimed minus the early exit fees |

### protocolFeeReceiverWithdraw

```solidity
function protocolFeeReceiverWithdraw() internal
```

Helper function to withdraw any accrued protocol fees

### calculateVariableWithdrawState

```solidity
function calculateVariableWithdrawState(uint256 totalEarnings, uint256 previousWithdrawnAmount) internal view returns (uint256, uint256)
```

Helper function to calculate the ongoing variable withdraw state
The vault must track a variable user's withdrawals during the duration of the vault since withdrawals can be executed at any time

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| totalEarnings | uint256 | Amount of staking or fee earnings |
| previousWithdrawnAmount | uint256 | The total amount of earnings already withdrawn |

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | uint256 | (currentState, amountOwed) The new total amount of earnings withdrawn, the amount to withdraw |
| [1] | uint256 |  |

### calculateVariableFeeEarningsShare

```solidity
function calculateVariableFeeEarningsShare() internal returns (uint256)
```

Helper function to calculate a variable user's proportional amount of fees that they are owed

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | uint256 | amount The amount to give the user |

### calculateFixedEarlyExitFees

```solidity
function calculateFixedEarlyExitFees(uint256 upfrontPremium, uint256 timestampRequested) internal view returns (uint256)
```

Helper function to calculate the total fees to apply to a fixed user's principal that they are requesting to withdraw
The earlier in the vault duration the withdraw request is made the higher the fee
The user must also pay back their upfront premium

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| upfrontPremium | uint256 | Upfront premium the fixed user received |
| timestampRequested | uint256 | When the withdraw was requested |

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | uint256 | earlyExitFee The total amount of early exit fees to apply |

## NonTransferrableVaultBearerToken

### constructor

```solidity
constructor(string name, string symbol, address vault) public
```

### transfer

```solidity
function transfer(address recipient, uint256 amount) public virtual returns (bool)
```

### allowance

```solidity
function allowance(address owner, address spender) public view virtual returns (uint256)
```

### approve

```solidity
function approve(address spender, uint256 amount) public virtual returns (bool)
```

### transferFrom

```solidity
function transferFrom(address sender, address recipient, uint256 amount) public virtual returns (bool)
```

## VaultBearerToken

Vaults create these tokens to give vault participants ownership of their positions

### vault

```solidity
address vault
```

The address of the vault that owns this token

### onlyVault

```solidity
modifier onlyVault()
```

This modifier checks that the sender is the vault address

### constructor

```solidity
constructor(string name, string symbol, address _vault) public
```

### mint

```solidity
function mint(address _to, uint256 _amount) external
```

Mints tokens

_Only the owning vault can do this_

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| _to | address | The address to mint to |
| _amount | uint256 | The amount to mint |

### burn

```solidity
function burn(address _account, uint256 _amount) external
```

Burns tokens

_Only the owning vault can do this_

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| _account | address | The address to burn from |
| _amount | uint256 | The amount to burn |

## ILido

### submit

```solidity
function submit(address _referral) external payable returns (uint256)
```

Send funds to the pool with optional _referral parameter

_This function is alternative way to submit funds. Supports optional referral address._

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | uint256 | Amount of StETH shares generated |

### transfer

```solidity
function transfer(address _recipient, uint256 _amount) external returns (bool)
```

Moves `_amount` tokens from the caller's account to the `_recipient` account.

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | bool | a boolean value indicating whether the operation succeeded. |

### approve

```solidity
function approve(address spender, uint256 amount) external returns (bool)
```

Sets amount as the allowance of spender over the caller’s StETH tokens.

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | bool | bool indicating whether the operation succeeded |

### getPooledEthByShares

```solidity
function getPooledEthByShares(uint256 _sharesAmount) external view returns (uint256)
```

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | uint256 | uint256 Returns the amount of ether that corresponds to _sharesAmount token shares. |

### balanceOf

```solidity
function balanceOf(address account) external view returns (uint256)
```

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | uint256 | uint256 Returns stETH balance of the account |

## ILidoAdapter

Manages interactions with the Lido staking and withdrawal contracts

### initialize

```solidity
function initialize(uint256 id, address admin) external
```

Initializes the adapter

_Make sure this is only callable by the vault creator_

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| id | uint256 | ID of adapter |
| admin | address | address of admin |

### setVault

```solidity
function setVault(address vault) external
```

Updates the vault contract address

_Make sure this is only callable by the vault creator_

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| vault | address | Vault contract address |

### owner

```solidity
function owner() external view returns (address)
```

Owner of contract - set in the constructor to the contract creator

### vault

```solidity
function vault() external view returns (address)
```

Vault contract address

### admin

```solidity
function admin() external view returns (address)
```

Admin of contract

### minStETHWithdrawalAmount

```solidity
function minStETHWithdrawalAmount() external view returns (uint256)
```

The minimum amount of stETH that can be withdrawn from Lido

### maxStETHWithdrawalAmount

```solidity
function maxStETHWithdrawalAmount() external view returns (uint256)
```

The maximum amount of stETH that can be withdrawn from Lido in a single request
If desired amount to withdraw is greater than the value it has to be broken up into multiple requests

### stakingBalance

```solidity
function stakingBalance() external view returns (uint256)
```

Balance of stETH tokens, which is dynamic based off of staking rewards https://docs.lido.fi/contracts/lido#rebase

### stakeFunds

```solidity
function stakeFunds(address user) external payable returns (uint256)
```

Stakes ETH on Lido and returns stETH

_Make sure this is only callable by the vault_

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| user | address | Address of the user that is staking the funds |

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | uint256 | shares amount of Lido shares issued for the staked ETH |

### transferStETH

```solidity
function transferStETH(address recipient, uint256 amount) external
```

Transfers the stETH balance to another address

_Make sure this is only callable by the vault_

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| recipient | address | Address to transfer stETH to |
| amount | uint256 | Amount of stETH to transfer |

### transferWithdrawalERC721

```solidity
function transferWithdrawalERC721(address recipient, uint256 requestId) external
```

Transfers a Lido ERC721 representing a staking withdrawal request

_Make sure this is only callable by the vault_

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| recipient | address | Address to transfer to |
| requestId | uint256 | Lido withdrawal request id |

### transferWithdrawnFunds

```solidity
function transferWithdrawnFunds(address recipient, uint256 amount) external
```

Transfers ETH unstaked from Lido

_Make sure this is only callable by the vault_

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| recipient | address | Address to transfer to |
| amount | uint256 | Amount of ETH |

### requestWithdrawViaShares

```solidity
function requestWithdrawViaShares(address user, uint256 shares) external returns (uint256[])
```

Request a withdrawal on Lido to exchange stETH for ETH via shares amount

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| user | address | Address of the user that is requesting the withdraw |
| shares | uint256 | Amount of staking shares to withdraw. Will be used to calculate the actual stETH amount to withdraw |

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | uint256[] | requestIds Ids of the withdrawal requests |

### requestWithdrawViaETH

```solidity
function requestWithdrawViaETH(address user, uint256 stETHAmount) external returns (uint256[])
```

Request a withdrawal on Lido to exchange stETH for ETH via stETH amount

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| user | address | Address of the user that is requesting the withdraw |
| stETHAmount | uint256 | Amount of stETH to withdraw |

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | uint256[] | requestIds Ids of the withdrawal requests |

### requestEntireBalanceWithdraw

```solidity
function requestEntireBalanceWithdraw(address user) external returns (uint256[])
```

Request a withdrawal of the entire stETH balance of the contract

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| user | address | Address of the user that is requesting the withdraw |

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | uint256[] | requestIds Ids of the withdrawal requests |

### claimWithdrawals

```solidity
function claimWithdrawals(address user, uint256[] requestIds) external returns (uint256)
```

Claim a withdrawal request on Lido once it has been finalized. This will transfer ETH to the contract if successful

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| user | address | Address of the user that is claiming the withdraw |
| requestIds | uint256[] | Withdrawal request ids to claim |

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | uint256 | amount of ETH withdrawn |

## ILidoVault

### initialize

```solidity
function initialize(struct ILidoVaultInitializer.InitializationParams params) external
```

Initializes the vault

_make sure this is only callable by the contract creator aka the vault factory_

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| params | struct ILidoVaultInitializer.InitializationParams | Initialization parameters |

## ILidoVaultInitializer

Defines the parameter struct used during vault initialization

### InitializationParams

```solidity
struct InitializationParams {
  uint256 vaultId;
  uint256 duration;
  uint256 fixedSideCapacity;
  uint256 fixedPremiumBps;
  uint256 earlyExitFeeBps;
  uint256 protocolFeeBps;
  address protocolFeeReceiver;
  address admin;
  address lidoAdapter;
  address fixedClaimToken;
  address fixedETHDepositToken;
  address fixedBearerToken;
  address variableBearerToken;
}
```

## ILidoWithdrawalQueueERC721

### requestWithdrawals

```solidity
function requestWithdrawals(uint256[] _amounts, address _owner) external returns (uint256[] requestIds)
```

Batch request the _amounts of stETH for withdrawal to the _owner address. For each request, the respective amount of stETH is transferred to this contract address, and an unstETH NFT is minted to the _owner address.

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| requestIds | uint256[] | Returns the array of ids for each created request |

### safeTransferFrom

```solidity
function safeTransferFrom(address _from, address _to, uint256 _requestId) external
```

_See {IERC721-safeTransferFrom}._

### claimWithdrawal

```solidity
function claimWithdrawal(uint256 _requestId) external
```

Claims the _requestId withdrawal request, sending ether to msg.sender address.

## IVaultBearerToken

The vault factory creates these tokens and they are used to give vault participants ownership of their positions

### vault

```solidity
function vault() external view returns (address)
```

The address of the vault that owns this token

### mint

```solidity
function mint(address _to, uint256 _amount) external
```

Mints tokens

_Only the owning vault can do this_

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| _to | address | The address to mint to |
| _amount | uint256 | The amount to mint |

### burn

```solidity
function burn(address _account, uint256 _amount) external
```

Burns tokens

_Only the owning vault can do this_

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| _account | address | The address to burn from |
| _amount | uint256 | The amount to burn |

## VaultFactory

Configure and deploy vaults; allow owner to add new adapter types

### nextVaultId

```solidity
uint256 nextVaultId
```

Incrementing vault ID

### nextAdapterId

```solidity
uint256 nextAdapterId
```

Incrementing adapter ID

### nextAdapterTypeId

```solidity
uint256 nextAdapterTypeId
```

Incrementing adapter type ID

### protocolFeeBps

```solidity
uint256 protocolFeeBps
```

Protocol fee taken from earnings in basis points (one basis point = 1/100 of 1%)

### protocolFeeReceiver

```solidity
address protocolFeeReceiver
```

Address that collects protocol fees

### earlyExitFeeBps

```solidity
uint256 earlyExitFeeBps
```

Fixed income early exit fee in basis points (one basis point = 1/100 of 1%) that is quadratically scaled based off of early exit time

### fixedPremiumBps

```solidity
uint256 fixedPremiumBps
```

Fixed upfront premium from variable deposits to fixed depositors in basis points (one basis point = 1/100 of 1%)

### admin

```solidity
address admin
```

Address that can call admin actions on the vault

### vaultByteCode

```solidity
bytes vaultByteCode
```

Lido Vault bytecode

### VaultInfo

```solidity
struct VaultInfo {
  address creator;
  address addr;
  address adapter;
}
```

### vaultInfo

```solidity
mapping(uint256 => struct VaultFactory.VaultInfo) vaultInfo
```

Info about vault, mapped by vault ID

### vaultAddrToId

```solidity
mapping(address => uint256) vaultAddrToId
```

ID of vault, mapped by vault address

### AdapterInfo

```solidity
struct AdapterInfo {
  uint256 adapterTypeId;
  address creator;
  address addr;
}
```

### adapterInfo

```solidity
mapping(uint256 => struct VaultFactory.AdapterInfo) adapterInfo
```

Adapter info, mapped by adapter ID

### adapterAddrToId

```solidity
mapping(address => uint256) adapterAddrToId
```

Adapter ID, mapped by Adapter address

### adapterTypeByteCode

```solidity
mapping(uint256 => bytes) adapterTypeByteCode
```

Adapter bytecode, mapped by Adapter ID

### VaultCodeSet

```solidity
event VaultCodeSet(address creator)
```

Emitted when the vault bytecode is set

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| creator | address | Address of creator |

### AdapterCreated

```solidity
event AdapterCreated(uint256 id, uint256 adapterTypeId, address creator, address adapter)
```

Emitted when an adapter is deployed

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| id | uint256 | ID of adapter |
| adapterTypeId | uint256 | Type ID of adapter |
| creator | address | Address of creator |
| adapter | address | Address of adapter |

### AdapterTypeAdded

```solidity
event AdapterTypeAdded(uint256 id, address creator)
```

Emitted when a new adapter type is added

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| id | uint256 | ID of new adapter type |
| creator | address | Address of creator |

### AdapterTypeRevoked

```solidity
event AdapterTypeRevoked(uint256 id, address revoker)
```

Emitted when an adapter type is revoked

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| id | uint256 | ID of revoked adapter type |
| revoker | address | Address of revoker |

### VaultCreated

```solidity
event VaultCreated(uint256 vaultId, uint256 duration, uint256 fixedSideCapacity, uint256 fixedPremiumBps, uint256 earlyExitFeeBps, uint256 protocolFeeBps, address protocolFeeReceiver, address admin, address adapter, address creator, address vault)
```

Emitted when a new vault is deployed

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| vaultId | uint256 | ID of vault |
| duration | uint256 | How long the vault will be locked once started, in seconds |
| fixedSideCapacity | uint256 | Maximum capacity of fixed side |
| fixedPremiumBps | uint256 | Fixed depositor premium in basis points |
| earlyExitFeeBps | uint256 | Fixed depositor early exit fee in basis points |
| protocolFeeBps | uint256 | Protocol fee in basis points |
| protocolFeeReceiver | address | Address that collects protocol fee |
| admin | address | Address of vault admin |
| adapter | address | Address of adapter |
| creator | address | Address of vault creator |
| vault | address | Address of vault |

### ProtocolFeeBpsSet

```solidity
event ProtocolFeeBpsSet(uint256 protocolFeeBps, address setter)
```

Emitted when the fee is updated

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| protocolFeeBps | uint256 | New fee basis points |
| setter | address | Address of setter |

### ProtocolFeeReceiverSet

```solidity
event ProtocolFeeReceiverSet(address protocolFeeReceiver, address setter)
```

Emitted when the fee receiver is updated

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| protocolFeeReceiver | address | New fee receiver |
| setter | address | Address of setter |

### FixedPremiumBpsSet

```solidity
event FixedPremiumBpsSet(uint256 fixedPremiumBps, address setter)
```

### EarlyExitFeeBpsSet

```solidity
event EarlyExitFeeBpsSet(uint256 earlyExitFeeBps, address setter)
```

### AdminSet

```solidity
event AdminSet(address admin, address setter)
```

Emitted when the admin is updated

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| admin | address | New admin |
| setter | address | Address of setter |

### constructor

```solidity
constructor() public
```

### renounceOwnership

```solidity
function renounceOwnership() public
```

Disable ownership renunciation

### createVault

```solidity
function createVault(address _adapterAddress, uint256 _fixedSideCapacity, uint256 _duration, uint256 _fixedPremiumBps) public virtual
```

Deploys a new vault

### createAdapter

```solidity
function createAdapter(uint256 adapterTypeId, address vaultAddress) external
```

Deploys a new adapter

### addVault

```solidity
function addVault(bytes bytecode) external
```

Adds a the Lido vault bytecode

### addAdapterType

```solidity
function addAdapterType(bytes bytecode) external returns (uint256)
```

Adds a new adapter bytecode, indexed by an auto-incremented adapter type ID

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| bytecode | bytes | Bytecode of new adapter type |

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | uint256 | New adapter type ID |

### revokeAdapterType

```solidity
function revokeAdapterType(uint256 id) external
```

Removes an adapter type, preventing new vault deployments from using this type

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| id | uint256 | ID of adapter type to revoke |

### setProtocolFeeBps

```solidity
function setProtocolFeeBps(uint256 _protocolFeeBps) external
```

Set protocol fee basis points

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| _protocolFeeBps | uint256 | New basis points value to set as protocol fee |

### setProtocolFeeReceiver

```solidity
function setProtocolFeeReceiver(address _protocolFeeReceiver) external
```

Set new address to collect protocol fees

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| _protocolFeeReceiver | address | New address to set as fee receiver |

### setFixedPremiumBps

```solidity
function setFixedPremiumBps(uint256 _fixedPremiumBps) external
```

Set fixed premium basis points

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| _fixedPremiumBps | uint256 | New fixed premium in basis points |

### setEarlyExitFeeBps

```solidity
function setEarlyExitFeeBps(uint256 _earlyExitFeeBps) external
```

Set fixed early exit fee basis points

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| _earlyExitFeeBps | uint256 | New fixed early exit fee in basis points |

### setAdmin

```solidity
function setAdmin(address _admin) external
```

Set new address to be admin on vault

#### Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| _admin | address | New address to set as admin |

### wasDeployedByFactory

```solidity
function wasDeployedByFactory(address addr) external view returns (bool)
```

Check to see if a given vault was deployed by this factory

#### Return Values

| Name | Type | Description |
| ---- | ---- | ----------- |
| [0] | bool | True if address matches a vault deployed by this factory |

