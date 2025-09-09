# IMorphoLendingAdapter
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/1dbcbcfe9a8bcf9392b2ada63dd8f1827a90783b/src/interfaces/IMorphoLendingAdapter.sol)

**Inherits:**
[IPreLiquidationLendingAdapter](/src/interfaces/IPreLiquidationLendingAdapter.sol/interface.IPreLiquidationLendingAdapter.md)


## Functions
### authorizedCreator

The authorized creator of the MorphoLendingAdapter

*Only the authorized creator can create a new LeverageToken using this adapter on the LeverageManager*


```solidity
function authorizedCreator() external view returns (address _authorizedCreator);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`_authorizedCreator`|`address`|The authorized creator of the MorphoLendingAdapter|


### isUsed

Whether the MorphoLendingAdapter is in use

*If this is true, the MorphoLendingAdapter cannot be used to create a new LeverageToken*


```solidity
function isUsed() external view returns (bool _isUsed);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`_isUsed`|`bool`|Whether the MorphoLendingAdapter is in use|


### leverageManager

The LeverageManager contract


```solidity
function leverageManager() external view returns (ILeverageManager _leverageManager);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`_leverageManager`|`ILeverageManager`|The LeverageManager contract|


### morphoMarketId

The ID of the Morpho market that the MorphoLendingAdapter manages a position in


```solidity
function morphoMarketId() external view returns (Id _morphoMarketId);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`_morphoMarketId`|`Id`|The ID of the Morpho market that the MorphoLendingAdapter manages a position in|


### marketParams

The market parameters of the Morpho lending pool


```solidity
function marketParams()
    external
    view
    returns (address loanToken, address collateralToken, address oracle, address irm, uint256 lltv);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`loanToken`|`address`|The loan token of the Morpho lending pool|
|`collateralToken`|`address`|The collateral token of the Morpho lending pool|
|`oracle`|`address`|The oracle of the Morpho lending pool|
|`irm`|`address`|The IRM of the Morpho lending pool|
|`lltv`|`uint256`|The LLTV of the Morpho lending pool|


### morpho

The Morpho core protocol contract


```solidity
function morpho() external view returns (IMorpho _morpho);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`_morpho`|`IMorpho`|The Morpho core protocol contract|


## Events
### MorphoLendingAdapterInitialized
Event emitted when the MorphoLendingAdapter is initialized


```solidity
event MorphoLendingAdapterInitialized(
    Id indexed morphoMarketId, MarketParams marketParams, address indexed authorizedCreator
);
```

**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`morphoMarketId`|`Id`|The ID of the Morpho market that the MorphoLendingAdapter manages a position in|
|`marketParams`|`MarketParams`|The market parameters of the Morpho market|
|`authorizedCreator`|`address`|The authorized creator of the MorphoLendingAdapter, allowed to create LeverageTokens using this adapter|

### MorphoLendingAdapterUsed
Event emitted when the MorphoLendingAdapter is flagged as used


```solidity
event MorphoLendingAdapterUsed();
```

## Errors
### LendingAdapterAlreadyInUse
Thrown when someone tries to create a LeverageToken with this MorphoLendingAdapter but it is already in use


```solidity
error LendingAdapterAlreadyInUse();
```

