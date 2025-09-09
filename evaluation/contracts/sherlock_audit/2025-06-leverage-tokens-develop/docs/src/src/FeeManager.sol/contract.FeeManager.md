# FeeManager
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/ca7af3bd8afb6a515c334e2f448f621a379dc94e/src/FeeManager.sol)

**Inherits:**
[IFeeManager](/src/interfaces/IFeeManager.sol/interface.IFeeManager.md), Initializable, AccessControlUpgradeable

*The FeeManager contract is an upgradeable core contract that is responsible for managing the fees for LeverageTokens.
There are three types of fees:
- Token action fees: Fees charged that accumulate towards the value of the LeverageToken for current LeverageToken
holders, applied on equity for mints and redeems
- Treasury action fees: Fees charged in shares that are transferred to the configured treasury address, applied on
shares minted for mints and shares burned for redeems
- Management fees: Fees charged in shares that are transferred to the configured treasury address. The management fee
accrues linearly over time and is minted to the treasury when the `chargeManagementFee` function is executed
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
function __FeeManager_init(address defaultAdmin) public onlyInitializing;
```

### __FeeManager_init_unchained


```solidity
function __FeeManager_init_unchained(address defaultAdmin) internal onlyInitializing;
```

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

Returns the management fee for the LeverageManager


```solidity
function getManagementFee() public view returns (uint256 fee);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`fee`|`uint256`|Management fee for the LeverageManager, 100_00 is 100%|


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


### setManagementFee

Sets the management fee

*Only `FEE_MANAGER_ROLE` can call this function*


```solidity
function setManagementFee(uint128 fee) external onlyRole(FEE_MANAGER_ROLE);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`fee`|`uint128`|Management fee, 100_00 is 100%|


### setTreasury

Sets the address of the treasury. The treasury receives all treasury fees from the LeverageManager. If the
treasury is set to the zero address, the treasury fees are reset to 0 as well

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

*If the treasury is not set, this function does not mint any shares and does not revert*

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

*If the treasury is not set, this function returns 0, regardless of what the treasury action fee is set to*


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
    uint256 managementFee;
    mapping(ILeverageToken token => uint120) lastManagementFeeAccrualTimestamp;
    mapping(ExternalAction action => uint256) treasuryActionFee;
    mapping(ILeverageToken token => mapping(ExternalAction action => uint256)) tokenActionFee;
}
```

