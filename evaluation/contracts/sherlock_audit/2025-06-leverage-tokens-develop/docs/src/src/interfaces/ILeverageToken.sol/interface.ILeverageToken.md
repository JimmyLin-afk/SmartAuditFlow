# ILeverageToken
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/1dbcbcfe9a8bcf9392b2ada63dd8f1827a90783b/src/interfaces/ILeverageToken.sol)

**Inherits:**
IERC20


## Functions
### mint

Mints new tokens to the specified address

*Only the owner can call this function. Owner should be the LeverageManager contract*


```solidity
function mint(address to, uint256 amount) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`to`|`address`|The address to mint tokens to|
|`amount`|`uint256`|The amount of tokens to mint|


### burn

Burns tokens from the specified address

*Only the owner can call this function. Owner should be the LeverageManager contract*


```solidity
function burn(address from, uint256 amount) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`from`|`address`|The address to burn tokens from|
|`amount`|`uint256`|The amount of tokens to burn|


## Events
### LeverageTokenInitialized
Event emitted when the leverage token is initialized


```solidity
event LeverageTokenInitialized(string name, string symbol);
```

**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`name`|`string`|The name of the LeverageToken|
|`symbol`|`string`|The symbol of the LeverageToken|

