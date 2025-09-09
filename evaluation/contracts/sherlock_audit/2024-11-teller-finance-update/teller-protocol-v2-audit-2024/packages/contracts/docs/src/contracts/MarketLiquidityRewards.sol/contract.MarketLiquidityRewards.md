# MarketLiquidityRewards
[Git Source](https://github.com/teller-protocol/teller-protocol-v2/blob/f4bf5a00ae7113b0344876c13db9b3dd705154f6/contracts/MarketLiquidityRewards.sol)

**Inherits:**
IMarketLiquidityRewards, Initializable


## State Variables
### tellerV2

```solidity
address immutable tellerV2;
```


### marketRegistry

```solidity
address immutable marketRegistry;
```


### collateralManager

```solidity
address immutable collateralManager;
```


### allocationCount

```solidity
uint256 allocationCount;
```


### allocatedRewards

```solidity
mapping(uint256 => RewardAllocation) public allocatedRewards;
```


### rewardClaimedForBid

```solidity
mapping(uint256 => mapping(uint256 => bool)) public rewardClaimedForBid;
```


## Functions
### onlyMarketOwner


```solidity
modifier onlyMarketOwner(uint256 _marketId);
```

### constructor


```solidity
constructor(address _tellerV2, address _marketRegistry, address _collateralManager);
```

### initialize


```solidity
function initialize() external initializer;
```

### allocateRewards

Creates a new token allocation and transfers the token amount into escrow in this contract


```solidity
function allocateRewards(RewardAllocation calldata _allocation) public virtual returns (uint256 allocationId_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_allocation`|`RewardAllocation`|- The RewardAllocation struct data to create|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`allocationId_`|`uint256`|allocationId_|


### updateAllocation

Allows the allocator to update properties of an allocation


```solidity
function updateAllocation(
    uint256 _allocationId,
    uint256 _minimumCollateralPerPrincipalAmount,
    uint256 _rewardPerLoanPrincipalAmount,
    uint32 _bidStartTimeMin,
    uint32 _bidStartTimeMax
) public virtual;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_allocationId`|`uint256`|- The id for the allocation|
|`_minimumCollateralPerPrincipalAmount`|`uint256`|- The required collateralization ratio|
|`_rewardPerLoanPrincipalAmount`|`uint256`|- The reward to give per principal amount|
|`_bidStartTimeMin`|`uint32`|- The block timestamp that loans must have been accepted after to claim rewards|
|`_bidStartTimeMax`|`uint32`|- The block timestamp that loans must have been accepted before to claim rewards|


### increaseAllocationAmount

Allows anyone to add tokens to an allocation


```solidity
function increaseAllocationAmount(uint256 _allocationId, uint256 _tokenAmount) public virtual;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_allocationId`|`uint256`|- The id for the allocation|
|`_tokenAmount`|`uint256`|- The amount of tokens to add|


### deallocateRewards

Allows the allocator to withdraw some or all of the funds within an allocation


```solidity
function deallocateRewards(uint256 _allocationId, uint256 _tokenAmount) public virtual;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_allocationId`|`uint256`|- The id for the allocation|
|`_tokenAmount`|`uint256`|- The amount of tokens to withdraw|


### claimRewards

Allows a borrower or lender to withdraw the allocated ERC20 reward for their loan


```solidity
function claimRewards(uint256 _allocationId, uint256 _bidId) external virtual;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_allocationId`|`uint256`|- The id for the reward allocation|
|`_bidId`|`uint256`|- The id for the loan. Each loan only grants one reward per allocation.|


### _verifyAndReturnRewardRecipient

Verifies that the bid state is appropriate for claiming rewards based on the allocation strategy and then returns the address of the reward recipient(borrower or lender)


```solidity
function _verifyAndReturnRewardRecipient(
    AllocationStrategy _strategy,
    BidState _bidState,
    address _borrower,
    address _lender
) internal virtual returns (address rewardRecipient_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_strategy`|`AllocationStrategy`|- The strategy for the reward allocation.|
|`_bidState`|`BidState`|- The bid state of the loan.|
|`_borrower`|`address`|- The borrower of the loan.|
|`_lender`|`address`|- The lender of the loan.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`rewardRecipient_`|`address`|The address that will receive the rewards. Either the borrower or lender.|


### _decrementAllocatedAmount

Decrements the amount allocated to keep track of tokens in escrow


```solidity
function _decrementAllocatedAmount(uint256 _allocationId, uint256 _amount) internal;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_allocationId`|`uint256`|- The id for the allocation to decrement|
|`_amount`|`uint256`|- The amount of ERC20 to decrement|


### _calculateRewardAmount

Calculates the reward to claim for the allocation


```solidity
function _calculateRewardAmount(
    uint256 _loanPrincipal,
    uint256 _principalTokenDecimals,
    uint256 _rewardPerLoanPrincipalAmount
) internal view returns (uint256);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_loanPrincipal`|`uint256`|- The amount of principal for the loan for which to reward|
|`_principalTokenDecimals`|`uint256`|- The number of decimals of the principal token|
|`_rewardPerLoanPrincipalAmount`|`uint256`|- The amount of reward per loan principal amount, expanded by the principal token decimals|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|The amount of ERC20 to reward|


### _verifyCollateralAmount

Verifies that the collateral ratio for the loan was sufficient based on _minimumCollateralPerPrincipalAmount of the allocation


```solidity
function _verifyCollateralAmount(
    address _collateralTokenAddress,
    uint256 _collateralAmount,
    address _principalTokenAddress,
    uint256 _principalAmount,
    uint256 _minimumCollateralPerPrincipalAmount
) internal virtual;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_collateralTokenAddress`|`address`|- The contract address for the collateral token|
|`_collateralAmount`|`uint256`|- The number of decimals of the collateral token|
|`_principalTokenAddress`|`address`|- The contract address for the principal token|
|`_principalAmount`|`uint256`|- The number of decimals of the principal token|
|`_minimumCollateralPerPrincipalAmount`|`uint256`|- The amount of collateral required per principal amount. Expanded by the principal token decimals and collateral token decimals.|


### _requiredCollateralAmount

Calculates the minimum amount of collateral the loan requires based on principal amount


```solidity
function _requiredCollateralAmount(
    uint256 _principalAmount,
    uint256 _principalTokenDecimals,
    uint256 _collateralTokenDecimals,
    uint256 _minimumCollateralPerPrincipalAmount
) internal view virtual returns (uint256);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_principalAmount`|`uint256`|- The number of decimals of the principal token|
|`_principalTokenDecimals`|`uint256`|- The number of decimals of the principal token|
|`_collateralTokenDecimals`|`uint256`|- The number of decimals of the collateral token|
|`_minimumCollateralPerPrincipalAmount`|`uint256`|- The amount of collateral required per principal amount. Expanded by the principal token decimals and collateral token decimals.|


### _verifyLoanStartTime

Verifies that the loan start time is within the bounds set by the allocation requirements


```solidity
function _verifyLoanStartTime(uint32 _loanStartTime, uint32 _minStartTime, uint32 _maxStartTime) internal virtual;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_loanStartTime`|`uint32`|- The timestamp when the loan was accepted|
|`_minStartTime`|`uint32`|- The minimum time required, after which the loan must have been accepted|
|`_maxStartTime`|`uint32`|- The maximum time required, before which the loan must have been accepted|


### _verifyExpectedTokenAddress

Verifies that the loan principal token address is per the requirements of the allocation


```solidity
function _verifyExpectedTokenAddress(address _loanTokenAddress, address _expectedTokenAddress) internal virtual;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_loanTokenAddress`|`address`|- The contract address of the token|
|`_expectedTokenAddress`|`address`|- The expected contract address per the allocation|


## Events
### CreatedAllocation

```solidity
event CreatedAllocation(uint256 allocationId, address allocator, uint256 marketId);
```

### UpdatedAllocation

```solidity
event UpdatedAllocation(uint256 allocationId);
```

### IncreasedAllocation

```solidity
event IncreasedAllocation(uint256 allocationId, uint256 amount);
```

### DecreasedAllocation

```solidity
event DecreasedAllocation(uint256 allocationId, uint256 amount);
```

### DeletedAllocation

```solidity
event DeletedAllocation(uint256 allocationId);
```

### ClaimedRewards

```solidity
event ClaimedRewards(uint256 allocationId, uint256 bidId, address recipient, uint256 amount);
```

