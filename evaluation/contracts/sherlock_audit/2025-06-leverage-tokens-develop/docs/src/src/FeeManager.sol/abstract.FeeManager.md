# FeeManager
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/1dbcbcfe9a8bcf9392b2ada63dd8f1827a90783b/src/FeeManager.sol)

**Inherits:**
[IFeeManager](/src/interfaces/IFeeManager.sol/interface.IFeeManager.md), Initializable, AccessControlUpgradeable

*The FeeManager contract is an abstract upgradeable core contract that is responsible for managing the fees for LeverageTokens.
There are three types of fees:
- Token action fees: Fees charged that accumulate towards the value of the LeverageToken for current LeverageToken
holders, applied on equity for mints and redeems
- Treasury action fees: Fees charged in shares that are transferred to the configured treasury address, applied on
shares minted for mints and shares burned for redeems
- Management fees: Fees charged in shares that are transferred to the configured treasury address. The management fee
accrues linearly over time and is minted to the treasury when the `chargeManagementFee` function is executed
Note: This contract is abstract and meant to be inherited by LeverageManager
The maximum fee that can be set for each action is 100_00 (100%).*


## State Variables
### FEE_MANAGER_ROLE

```solidity
bytes32 public constant FEE_MANAGER_ROLE = keccak256("FEE_MANAGER_ROLE");
```


### MAX_FEE

```solidity
uint256 internal constant MAX_FEE = 100_00;
```


### SECS_PER_YEAR

```solidity
uint256 internal constant SECS_PER_YEAR = 31536000;
```


## Functions
### _getFeeManagerStorage


```solidity
function _getFeeManagerStorage() internal pure returns (FeeManagerStorage storage $);
```

### __FeeManager_init


```solidity
function __FeeManager_init(address defaultAdmin, address treasury) public onlyInitializing;
```

### __FeeManager_init_unchained


```solidity
function __FeeManager_init_unchained(address defaultAdmin, address treasury) internal onlyInitializing;
```

### getDefaultManagementFeeAtCreation

Returns the default management fee for new LeverageTokens


```solidity
function getDefaultManagementFeeAtCreation() public view returns (uint256);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|fee The default management fee for new LeverageTokens, 100_00 is 100%|


### getLastManagementFeeAccrualTimestamp

Returns the timestamp of the most recent management fee accrual for a LeverageToken


```solidity
function getLastManagementFeeAccrualTimestamp(ILeverageToken token) public view returns (uint120);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`||

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint120`|timestamp The timestamp of the most recent management fee accrual|


### getLeverageTokenActionFee

Returns the LeverageToken fee for a specific action


```solidity
function getLeverageTokenActionFee(ILeverageToken token, ExternalAction action) public view returns (uint256 fee);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`||
|`action`|`ExternalAction`|The action to get fee for|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`fee`|`uint256`|Fee for action, 100_00 is 100%|


### getManagementFee

Returns the management fee for a LeverageToken


```solidity
function getManagementFee(ILeverageToken token) public view returns (uint256 fee);
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
function getTreasury() public view returns (address treasury);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`treasury`|`address`|The address of the treasury|


### getTreasuryActionFee

Returns the treasury fee for a specific action


```solidity
function getTreasuryActionFee(ExternalAction action) public view returns (uint256 fee);
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
function setDefaultManagementFeeAtCreation(uint256 fee) external onlyRole(FEE_MANAGER_ROLE);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`fee`|`uint256`|The default management fee for new LeverageTokens, 100_00 is 100%|


### setManagementFee

Sets the management fee for a LeverageToken

*Only `FEE_MANAGER_ROLE` can call this function*


```solidity
function setManagementFee(ILeverageToken token, uint256 fee) external onlyRole(FEE_MANAGER_ROLE);
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
function setTreasury(address treasury) external onlyRole(FEE_MANAGER_ROLE);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`treasury`|`address`|The address of the treasury|


### setTreasuryActionFee

Sets the treasury fee for a specific action

*Only `FEE_MANAGER_ROLE` can call this function.*


```solidity
function setTreasuryActionFee(ExternalAction action, uint256 fee) external onlyRole(FEE_MANAGER_ROLE);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`action`|`ExternalAction`|The action to set fee for|
|`fee`|`uint256`|The fee for action, 100_00 is 100%|


### chargeManagementFee

Function that charges any accrued management fees for the LeverageToken by minting shares to the treasury

*If the treasury is not set, the management fee is not charged (shares are not minted to the treasury) but
still accrues*


```solidity
function chargeManagementFee(ILeverageToken token) public;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to charge management fee for|


### _chargeTreasuryFee

Function that mints shares to the treasury for the treasury action fee, if the treasury is set

*This contract must be authorized to mint shares for the LeverageToken*


```solidity
function _chargeTreasuryFee(ILeverageToken token, uint256 shares) internal;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to mint shares to treasury for|
|`shares`|`uint256`|Shares to mint|


### _computeTokenFee

Computes the token action fee for a given action

*Fees are always rounded up.*


```solidity
function _computeTokenFee(ILeverageToken token, uint256 equity, ExternalAction action)
    internal
    view
    returns (uint256, uint256);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to compute token action fee for|
|`equity`|`uint256`|Amount of equity to compute token action fee for, denominated in collateral asset|
|`action`|`ExternalAction`|Action to compute token action fee for, Mint or Redeem|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|equityForShares Equity to mint / burn shares for the LeverageToken after token action fees, denominated in collateral asset of the LeverageToken|
|`<none>`|`uint256`|tokenFee LeverageToken token action fee amount in equity, denominated in the collateral asset of the LeverageToken|


### _computeTreasuryFee

Computes the treasury action fee for a given action


```solidity
function _computeTreasuryFee(ExternalAction action, uint256 shares) internal view returns (uint256);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`action`|`ExternalAction`|Action to compute treasury action fee for|
|`shares`|`uint256`|Shares to compute treasury action fee for|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|treasuryFee Treasury action fee amount in shares|


### _getFeeAdjustedTotalSupply

Function that returns the total supply of the LeverageToken adjusted for any accrued management fees


```solidity
function _getFeeAdjustedTotalSupply(ILeverageToken token) internal view returns (uint256);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to get fee adjusted total supply for|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|totalSupply Fee adjusted total supply of the LeverageToken|


### _getAccruedManagementFee

Function that calculates how many shares to mint for the accrued management fee at the current timestamp


```solidity
function _getAccruedManagementFee(ILeverageToken token) internal view returns (uint256);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to calculate management fee shares for|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|shares Shares to mint|


### _setLeverageTokenActionFee

Sets the LeverageToken fee for a specific action

*If caller tries to set fee above 100% it reverts with FeeTooHigh error*


```solidity
function _setLeverageTokenActionFee(ILeverageToken token, ExternalAction action, uint256 fee) internal;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to set fee for|
|`action`|`ExternalAction`|Action to set fee for|
|`fee`|`uint256`|Fee for action, 100_00 is 100%|


### _setNewLeverageTokenManagementFee

Sets the management fee for a new LeverageToken and the last management fee accrual timestamp to the
current timestamp


```solidity
function _setNewLeverageTokenManagementFee(ILeverageToken token) internal;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to set management fee for|


### _setTreasury

Sets the treasury address

*Reverts if the treasury address is zero*


```solidity
function _setTreasury(address treasury) internal;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`treasury`|`address`|Treasury address to set|


### _validateFee

Validates that the fee is not higher than 100%


```solidity
function _validateFee(uint256 fee) internal pure;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`fee`|`uint256`|Fee to validate|


## Structs
### FeeManagerStorage
*Struct containing all state for the FeeManager contract*

**Note:**
storage-location: erc7201:seamless.contracts.storage.FeeManager


```solidity
struct FeeManagerStorage {
    address treasury;
    uint256 defaultManagementFeeAtCreation;
    mapping(ILeverageToken token => uint256) managementFee;
    mapping(ILeverageToken token => uint120) lastManagementFeeAccrualTimestamp;
    mapping(ExternalAction action => uint256) treasuryActionFee;
    mapping(ILeverageToken token => mapping(ExternalAction action => uint256)) tokenActionFee;
}
```

