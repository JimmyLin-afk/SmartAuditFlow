# LeverageRouterDepositBase
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/002c85336929e7b2f8b2193e3cb727fe9cf4b9e6/src/periphery/LeverageRouterDepositBase.sol)

**Inherits:**
[LeverageRouterBase](/src/periphery/LeverageRouterBase.sol/abstract.LeverageRouterBase.md)

*The LeverageRouterDepositBase contract is an abstract periphery contract that facilitates the use of Morpho flash loans
to deposit equity into LeverageTokens.*


## Functions
### constructor

Creates a new LeverageRouterDeposit


```solidity
constructor(ILeverageManager _leverageManager, IMorpho _morpho) LeverageRouterBase(_leverageManager, _morpho);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_leverageManager`|`ILeverageManager`|The LeverageManager contract|
|`_morpho`|`IMorpho`|The Morpho core protocol contract|


### _depositAndRepayMorphoFlashLoan

Executes the deposit of equity into a LeverageToken and the logic to obtain collateral assets from debt assets
to repay the flash loan from Morpho


```solidity
function _depositAndRepayMorphoFlashLoan(DepositParams memory params, uint256 collateralLoanAmount) internal virtual;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`params`|`DepositParams`|Params for the deposit of equity into a LeverageToken|
|`collateralLoanAmount`|`uint256`|Amount of collateral asset flash loaned|


### _deposit

Performs the logic to deposit equity into a LeverageToken, including the transfer of collateral from the sender,
the approval of the collateral asset, and the deposit into the LeverageToken


```solidity
function _deposit(DepositParams memory params, IERC20 collateralAsset, uint256 collateralLoanAmount)
    internal
    virtual
    returns (ActionData memory);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`params`|`DepositParams`|Params for the deposit of equity into a LeverageToken|
|`collateralAsset`|`IERC20`||
|`collateralLoanAmount`|`uint256`|The amount of collateral asset flash loaned for the deposit|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`ActionData`|actionData The ActionData for the deposit|


### _getCollateralFromDebt

Performs logic to obtain collateral assets from some amount of debt asset


```solidity
function _getCollateralFromDebt(
    IERC20 debtAsset,
    uint256 debtAmount,
    uint256 minCollateralAmount,
    bytes memory additionalData
) internal virtual returns (uint256);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`debtAsset`|`IERC20`|The debt asset|
|`debtAmount`|`uint256`|The amount of debt to convert to collateral|
|`minCollateralAmount`|`uint256`|The minimum amount of collateral to obtain from the debt|
|`additionalData`|`bytes`|Any additional data to pass to the logic|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|The amount of collateral assets obtained|


## Structs
### DepositParams
Deposit related parameters to pass to the Morpho flash loan callback handler for deposits


```solidity
struct DepositParams {
    ILeverageToken token;
    uint256 equityInCollateralAsset;
    uint256 minShares;
    address sender;
    bytes additionalData;
}
```

