# ILeverageRouterBase
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/40214436ae3956021858cb95e6ff881f6ede8e11/src/interfaces/periphery/ILeverageRouterBase.sol)


## Functions
### leverageManager

The LeverageManager contract


```solidity
function leverageManager() external view returns (ILeverageManager _leverageManager);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`_leverageManager`|`ILeverageManager`|The LeverageManager contract|


### morpho

The Morpho core protocol contract


```solidity
function morpho() external view returns (IMorpho _morpho);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`_morpho`|`IMorpho`|The Morpho core protocol contract|


## Errors
### Unauthorized
Error thrown when the caller is not authorized to execute a function


```solidity
error Unauthorized();
```

