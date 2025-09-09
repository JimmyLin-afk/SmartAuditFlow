# ICollateralRatiosRebalanceAdapter
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/1dbcbcfe9a8bcf9392b2ada63dd8f1827a90783b/src/interfaces/ICollateralRatiosRebalanceAdapter.sol)

Interface for the CollateralRatiosRebalanceAdapter contract


## Functions
### getLeverageManager

Returns the LeverageManager


```solidity
function getLeverageManager() external view returns (ILeverageManager leverageManager);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`leverageManager`|`ILeverageManager`|The LeverageManager|


### getLeverageTokenMinCollateralRatio

Returns the minimum collateral ratio for a LeverageToken


```solidity
function getLeverageTokenMinCollateralRatio() external view returns (uint256 minCollateralRatio);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`minCollateralRatio`|`uint256`|Minimum collateral ratio for the LeverageToken|


### getLeverageTokenTargetCollateralRatio

Returns the target collateral ratio for a LeverageToken


```solidity
function getLeverageTokenTargetCollateralRatio() external view returns (uint256 targetCollateralRatio);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`targetCollateralRatio`|`uint256`|Target collateral ratio for the LeverageToken|


### getLeverageTokenMaxCollateralRatio

Returns the maximum collateral ratio for a LeverageToken


```solidity
function getLeverageTokenMaxCollateralRatio() external view returns (uint256 maxCollateralRatio);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`maxCollateralRatio`|`uint256`|Maximum collateral ratio for the LeverageToken|


### getLeverageTokenInitialCollateralRatio

Returns the initial collateral ratio for a LeverageToken


```solidity
function getLeverageTokenInitialCollateralRatio(ILeverageToken token)
    external
    view
    returns (uint256 initialCollateralRatio);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`initialCollateralRatio`|`uint256`|Initial collateral ratio for the LeverageToken|


### isEligibleForRebalance

Returns true if the LeverageToken is eligible for rebalance


```solidity
function isEligibleForRebalance(ILeverageToken token, LeverageTokenState memory state, address caller)
    external
    view
    returns (bool isEligible);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|The LeverageToken|
|`state`|`LeverageTokenState`|The state of the LeverageToken|
|`caller`|`address`|The caller of the function|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`isEligible`|`bool`|True if the LeverageToken is eligible for rebalance, false otherwise|


### isStateAfterRebalanceValid

Returns true if the LeverageToken state after rebalance is valid


```solidity
function isStateAfterRebalanceValid(ILeverageToken token, LeverageTokenState memory stateBefore)
    external
    view
    returns (bool isValid);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|The LeverageToken|
|`stateBefore`|`LeverageTokenState`|The state of the LeverageToken before rebalance|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`isValid`|`bool`|True if the LeverageToken state after rebalance is valid, false otherwise|


## Events
### CollateralRatiosRebalanceAdapterInitialized
Event emitted when the collateral ratios are set


```solidity
event CollateralRatiosRebalanceAdapterInitialized(
    uint256 minCollateralRatio, uint256 targetCollateralRatio, uint256 maxCollateralRatio
);
```

**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`minCollateralRatio`|`uint256`|Minimum collateral ratio for the LeverageToken|
|`targetCollateralRatio`|`uint256`|Target collateral ratio for the LeverageToken|
|`maxCollateralRatio`|`uint256`|Maximum collateral ratio for the LeverageToken|

## Errors
### InvalidCollateralRatios
Error thrown when min collateral ratio is too high


```solidity
error InvalidCollateralRatios();
```

