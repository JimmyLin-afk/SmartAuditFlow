# IPreLiquidationRebalanceAdapter
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/1dbcbcfe9a8bcf9392b2ada63dd8f1827a90783b/src/interfaces/IPreLiquidationRebalanceAdapter.sol)


## Functions
### getLeverageManager

Returns the LeverageManager contract


```solidity
function getLeverageManager() external view returns (ILeverageManager leverageManager);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`leverageManager`|`ILeverageManager`|The LeverageManager contract|


### getCollateralRatioThreshold

Returns the collateral ratio threshold for pre-liquidation rebalancing

*When the LeverageToken collateral ratio is below this threshold, the LeverageToken can be pre-liquidation
rebalanced*


```solidity
function getCollateralRatioThreshold() external view returns (uint256 collateralRatioThreshold);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`collateralRatioThreshold`|`uint256`|The collateral ratio threshold for pre-liquidation rebalancing|


### getRebalanceReward

Returns the rebalance reward percentage

*The rebalance reward represents the percentage of liquidation cost that will be rewarded to the caller of the
rebalance function. 10000 means 100%*


```solidity
function getRebalanceReward() external view returns (uint256 rebalanceRewardPercentage);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`rebalanceRewardPercentage`|`uint256`|The rebalance reward percentage|


### isStateAfterRebalanceValid

Returns true if the state after rebalance is valid


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
|`stateBefore`|`LeverageTokenState`|The state before rebalance|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`isValid`|`bool`|True if the state after rebalance is valid|


### isEligibleForRebalance

Returns true if the LeverageToken is eligible for pre-liquidation rebalance

*Token is eligible for pre-liquidation rebalance if health factor is below the threshold*


```solidity
function isEligibleForRebalance(ILeverageToken token, LeverageTokenState memory stateBefore, address caller)
    external
    view
    returns (bool isEligible);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|The LeverageToken|
|`stateBefore`|`LeverageTokenState`|The state before rebalance|
|`caller`|`address`|The caller of the rebalance function|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`isEligible`|`bool`|True if the LeverageToken is eligible for pre-liquidation rebalance|


## Events
### PreLiquidationRebalanceAdapterInitialized
Emitted when the PreLiquidationRebalanceAdapter is initialized


```solidity
event PreLiquidationRebalanceAdapterInitialized(uint256 collateralRatioThreshold, uint256 rebalanceReward);
```

**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`collateralRatioThreshold`|`uint256`|The collateral ratio threshold for pre-liquidation rebalancing. If the LeverageToken collateral ratio is below this threshold, the LeverageToken can be pre-liquidation rebalanced|
|`rebalanceReward`|`uint256`|The rebalance reward percentage. The rebalance reward represents the percentage of liquidation penalty that will be rewarded to the caller of the rebalance function. 10_000 means 100%|

