# IRebalanceAdapterBase
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/1dbcbcfe9a8bcf9392b2ada63dd8f1827a90783b/src/interfaces/IRebalanceAdapterBase.sol)

Interface for the base RebalanceAdapter

*This is minimal interface required for the RebalanceAdapter to be used by the LeverageManager*


## Functions
### getLeverageTokenInitialCollateralRatio

Returns the initial collateral ratio for a LeverageToken

*Initial collateral ratio is followed when the LeverageToken has no shares and on mints when debt is 0.*


```solidity
function getLeverageTokenInitialCollateralRatio(ILeverageToken token)
    external
    view
    returns (uint256 initialCollateralRatio);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to get initial collateral ratio for|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`initialCollateralRatio`|`uint256`|Initial collateral ratio for the LeverageToken|


### isEligibleForRebalance

Validates if a LeverageToken is eligible for rebalance


```solidity
function isEligibleForRebalance(ILeverageToken token, LeverageTokenState memory state, address caller)
    external
    view
    returns (bool isEligible);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to check eligibility for|
|`state`|`LeverageTokenState`|State of the LeverageToken|
|`caller`|`address`|Caller of the function|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`isEligible`|`bool`|True if LeverageToken is eligible for rebalance, false otherwise|


### isStateAfterRebalanceValid

Validates if the LeverageToken's state after rebalance is valid


```solidity
function isStateAfterRebalanceValid(ILeverageToken token, LeverageTokenState memory stateBefore)
    external
    view
    returns (bool isValid);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to validate state for|
|`stateBefore`|`LeverageTokenState`|State of the LeverageToken before rebalance|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`isValid`|`bool`|True if state after rebalance is valid, false otherwise|


### postLeverageTokenCreation

Post-LeverageToken creation hook. Used for any validation logic or initialization after a LeverageToken
is created using this adapter

*This function is called in `LeverageManager.createNewLeverageToken` after the new LeverageToken is created*


```solidity
function postLeverageTokenCreation(address creator, address leverageToken) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`creator`|`address`|The address of the creator of the LeverageToken|
|`leverageToken`|`address`|The address of the LeverageToken that was created|


