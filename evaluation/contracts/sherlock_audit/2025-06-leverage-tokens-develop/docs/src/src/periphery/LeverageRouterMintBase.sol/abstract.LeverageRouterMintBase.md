# LeverageRouterMintBase
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/40214436ae3956021858cb95e6ff881f6ede8e11/src/periphery/LeverageRouterMintBase.sol)

**Inherits:**
[LeverageRouterBase](/src/periphery/LeverageRouterBase.sol/abstract.LeverageRouterBase.md)

*The LeverageRouterMintBase contract is an abstract periphery contract that facilitates the use of Morpho flash loans
to mint LeverageTokens (shares).*


## Functions
### constructor

Creates a new LeverageRouterMint


```solidity
constructor(ILeverageManager _leverageManager, IMorpho _morpho) LeverageRouterBase(_leverageManager, _morpho);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_leverageManager`|`ILeverageManager`|The LeverageManager contract|
|`_morpho`|`IMorpho`|The Morpho core protocol contract|


### _mintAndRepayMorphoFlashLoan

Executes the mint for a LeverageToken and the logic to obtain collateral assets from debt assets
to repay the flash loan from Morpho


```solidity
function _mintAndRepayMorphoFlashLoan(MintParams memory params, uint256 collateralLoanAmount) internal virtual;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`params`|`MintParams`|Params for the mint of a LeverageToken|
|`collateralLoanAmount`|`uint256`|Amount of collateral asset flash loaned|


### _mint

Performs the logic to mint shares of a LeverageToken by adding equity, including the transfer of collateral from the sender,
the approval of the collateral asset, and the mint into the LeverageToken


```solidity
function _mint(MintParams memory params, IERC20 collateralAsset, uint256 collateralLoanAmount)
    internal
    virtual
    returns (ActionData memory);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`params`|`MintParams`|Params for the mint of LeverageToken shares|
|`collateralAsset`|`IERC20`||
|`collateralLoanAmount`|`uint256`|The amount of collateral asset flash loaned for the mint|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`ActionData`|actionData The ActionData for the mint|


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
### MintParams
Mint related parameters to pass to the Morpho flash loan callback handler for mints


```solidity
struct MintParams {
    ILeverageToken token;
    uint256 equityInCollateralAsset;
    uint256 minShares;
    address sender;
    bytes additionalData;
}
```

