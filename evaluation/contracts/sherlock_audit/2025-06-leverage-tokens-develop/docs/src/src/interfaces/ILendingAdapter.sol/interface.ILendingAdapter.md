# ILendingAdapter
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/1dbcbcfe9a8bcf9392b2ada63dd8f1827a90783b/src/interfaces/ILendingAdapter.sol)


## Functions
### getCollateralAsset

Returns the address of the collateral asset


```solidity
function getCollateralAsset() external view returns (IERC20 collateralAsset);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`collateralAsset`|`IERC20`|Address of the collateral asset|


### getDebtAsset

Returns the address of the debt asset


```solidity
function getDebtAsset() external view returns (IERC20 debtAsset);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`debtAsset`|`IERC20`|Address of the debt asset|


### convertCollateralToDebtAsset

Converts an amount of collateral asset to a debt asset amount based on the lending pool oracle


```solidity
function convertCollateralToDebtAsset(uint256 collateral) external view returns (uint256 debt);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`collateral`|`uint256`|Collateral amount|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`debt`|`uint256`|Amount of debt asset|


### convertDebtToCollateralAsset

Converts an amount of debt asset to a collateral asset amount based on the lending pool oracle


```solidity
function convertDebtToCollateralAsset(uint256 debt) external view returns (uint256 collateral);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`debt`|`uint256`|Debt amount|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`collateral`|`uint256`|Amount of collateral asset|


### getCollateral

Returns total collateral of the position held by the lending adapter


```solidity
function getCollateral() external view returns (uint256 collateral);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`collateral`|`uint256`|Total collateral of the position held by the lending adapter|


### getCollateralInDebtAsset

Returns the total collateral of the position held by the lending adapter denominated in the debt asset


```solidity
function getCollateralInDebtAsset() external view returns (uint256 collateral);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`collateral`|`uint256`|Total collateral of the position held by the lending adapter denominated in the debt asset|


### getDebt

Returns the total debt of the position held by the lending adapter


```solidity
function getDebt() external view returns (uint256 debt);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`debt`|`uint256`|Total debt of the position held by the lending adapter|


### getEquityInCollateralAsset

Returns the total equity of the position held by the lending adapter denominated in the collateral asset


```solidity
function getEquityInCollateralAsset() external view returns (uint256 equity);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`equity`|`uint256`|Equity of the position held by the lending adapter|


### getEquityInDebtAsset

Returns the total equity of the position held by the lending adapter denominated in the debt asset

*Equity is calculated as collateral - debt*


```solidity
function getEquityInDebtAsset() external view returns (uint256 equity);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`equity`|`uint256`|Equity of the position held by the lending adapter|


### addCollateral

Supplies collateral assets to the lending pool


```solidity
function addCollateral(uint256 amount) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`amount`|`uint256`|Amount of assets to supply|


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


### removeCollateral

Withdraws collateral assets from the lending pool


```solidity
function removeCollateral(uint256 amount) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`amount`|`uint256`|Amount of collateral assets to withdraw|


### borrow

Borrows debt assets from the lending pool


```solidity
function borrow(uint256 amount) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`amount`|`uint256`|Amount of debt assets to borrow|


### repay

Repays debt to the lending pool


```solidity
function repay(uint256 amount) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`amount`|`uint256`|Amount of debt assets to repay|


## Errors
### Unauthorized
Error thrown when the caller is unauthorized to call a function


```solidity
error Unauthorized();
```

