# IFeeManager
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/1dbcbcfe9a8bcf9392b2ada63dd8f1827a90783b/src/interfaces/IFeeManager.sol)


## Functions
### chargeManagementFee

Function that charges any accrued management fees for the LeverageToken by minting shares to the treasury

*If the treasury is not set, the management fee is not charged (shares are not minted to the treasury) but
still accrues*


```solidity
function chargeManagementFee(ILeverageToken token) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to charge management fee for|


### getDefaultManagementFeeAtCreation

Returns the default management fee for new LeverageTokens


```solidity
function getDefaultManagementFeeAtCreation() external view returns (uint256 fee);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`fee`|`uint256`|The default management fee for new LeverageTokens, 100_00 is 100%|


### getLastManagementFeeAccrualTimestamp

Returns the timestamp of the most recent management fee accrual for a LeverageToken


```solidity
function getLastManagementFeeAccrualTimestamp(ILeverageToken leverageToken) external view returns (uint120 timestamp);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`leverageToken`|`ILeverageToken`|The LeverageToken to get the timestamp for|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`timestamp`|`uint120`|The timestamp of the most recent management fee accrual|


### getLeverageTokenActionFee

Returns the LeverageToken fee for a specific action


```solidity
function getLeverageTokenActionFee(ILeverageToken leverageToken, ExternalAction action)
    external
    view
    returns (uint256 fee);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`leverageToken`|`ILeverageToken`|The LeverageToken to get fee for|
|`action`|`ExternalAction`|The action to get fee for|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`fee`|`uint256`|Fee for action, 100_00 is 100%|


### getManagementFee

Returns the management fee for a LeverageToken


```solidity
function getManagementFee(ILeverageToken token) external view returns (uint256 fee);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to get management fee for|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`fee`|`uint256`|Management fee for the LeverageToken, 100_00 is 100%|


### getTreasury

Returns the address of the treasury


```solidity
function getTreasury() external view returns (address treasury);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`treasury`|`address`|The address of the treasury|


### getTreasuryActionFee

Returns the treasury fee for a specific action


```solidity
function getTreasuryActionFee(ExternalAction action) external view returns (uint256 fee);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`action`|`ExternalAction`|Action to get fee for|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`fee`|`uint256`|Fee for action, 100_00 is 100%|


### setDefaultManagementFeeAtCreation

Sets the default management fee for new LeverageTokens

*Only `FEE_MANAGER_ROLE` can call this function*


```solidity
function setDefaultManagementFeeAtCreation(uint256 fee) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`fee`|`uint256`|The default management fee for new LeverageTokens, 100_00 is 100%|


### setManagementFee

Sets the management fee for a LeverageToken

*Only `FEE_MANAGER_ROLE` can call this function*


```solidity
function setManagementFee(ILeverageToken token, uint256 fee) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to set management fee for|
|`fee`|`uint256`|Management fee, 100_00 is 100%|


### setTreasury

Sets the address of the treasury. The treasury receives all treasury and management fees from the
LeverageManager.

*Only `FEE_MANAGER_ROLE` can call this function*


```solidity
function setTreasury(address treasury) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`treasury`|`address`|The address of the treasury|


### setTreasuryActionFee

Sets the treasury fee for a specific action

*Only `FEE_MANAGER_ROLE` can call this function.*


```solidity
function setTreasuryActionFee(ExternalAction action, uint256 fee) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`action`|`ExternalAction`|The action to set fee for|
|`fee`|`uint256`|The fee for action, 100_00 is 100%|


## Events
### DefaultManagementFeeAtCreationSet
Emitted when the default management fee for new LeverageTokens is updated


```solidity
event DefaultManagementFeeAtCreationSet(uint256 fee);
```

**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`fee`|`uint256`|The default management fee for new LeverageTokens, 100_00 is 100%|

### LeverageTokenActionFeeSet
Emitted when a LeverageToken fee is set for a specific action


```solidity
event LeverageTokenActionFeeSet(ILeverageToken indexed leverageToken, ExternalAction indexed action, uint256 fee);
```

**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`leverageToken`|`ILeverageToken`|The LeverageToken that the fee was set for|
|`action`|`ExternalAction`|The action that the fee was set for|
|`fee`|`uint256`|The fee that was set|

### ManagementFeeCharged
Emitted when the management fee is charged for a LeverageToken


```solidity
event ManagementFeeCharged(ILeverageToken indexed leverageToken, uint256 sharesFee);
```

**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`leverageToken`|`ILeverageToken`|The LeverageToken that the management fee was charged for|
|`sharesFee`|`uint256`|The amount of shares that were minted to the treasury|

### ManagementFeeSet
Emitted when the management fee is set


```solidity
event ManagementFeeSet(ILeverageToken indexed token, uint256 fee);
```

**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|The LeverageToken that the management fee was set for|
|`fee`|`uint256`|The fee that was set|

### TreasuryActionFeeSet
Emitted when a treasury fee is set for a specific action


```solidity
event TreasuryActionFeeSet(ExternalAction indexed action, uint256 fee);
```

**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`action`|`ExternalAction`|The action that the fee was set for|
|`fee`|`uint256`|The fee that was set|

### TreasurySet
Emitted when the treasury address is set


```solidity
event TreasurySet(address treasury);
```

**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`treasury`|`address`|The address of the treasury|

## Errors
### FeeTooHigh
Error emitted when `FEE_MANAGER_ROLE` tries to set fee higher than `MAX_FEE`


```solidity
error FeeTooHigh(uint256 fee, uint256 maxFee);
```

**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`fee`|`uint256`|The fee that was set|
|`maxFee`|`uint256`|The maximum fee that can be set|

### ZeroAddressTreasury
Error emitted when trying to set the treasury address to the zero address


```solidity
error ZeroAddressTreasury();
```

