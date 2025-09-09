# ILeverageRouter
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/1dbcbcfe9a8bcf9392b2ada63dd8f1827a90783b/src/interfaces/periphery/ILeverageRouter.sol)


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


### swapper

The swap adapter contract used to facilitate swaps


```solidity
function swapper() external view returns (ISwapAdapter _swapper);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`_swapper`|`ISwapAdapter`|The swap adapter contract|


### mint

Mint shares of a LeverageToken by adding equity

*Flash loans the collateral required to add the equity to the LeverageToken, receives debt, then swaps the debt to the
LeverageToken's collateral asset. The swapped assets and the sender's supplied collateral are used to repay the flash loan*

*The sender should approve the LeverageRouter to spend an amount of collateral assets greater than the equity being added
to facilitate the mint in the case that the mint requires additional collateral to cover swap slippage when swapping
debt to collateral to repay the flash loan. The approved amount should equal at least `equityInCollateralAsset + maxSwapCostInCollateralAsset`.
To see the preview of the mint, `LeverageRouter.leverageManager().previewMint(...)` can be used.*


```solidity
function mint(
    ILeverageToken token,
    uint256 equityInCollateralAsset,
    uint256 minShares,
    uint256 maxSwapCostInCollateralAsset,
    ISwapAdapter.SwapContext memory swapContext
) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to mint shares of|
|`equityInCollateralAsset`|`uint256`|The amount of equity to mint LeverageToken shares for. Denominated in the collateral asset of the LeverageToken|
|`minShares`|`uint256`|Minimum shares (LeverageTokens) to receive from the mint|
|`maxSwapCostInCollateralAsset`|`uint256`|The maximum amount of collateral from the sender to use to help repay the flash loan due to the swap of debt to collateral being unfavorable|
|`swapContext`|`ISwapAdapter.SwapContext`|Swap context to use for the swap (which DEX to use, the route, tick spacing, etc.)|


### redeem

Redeems equity of a LeverageToken by repaying debt and burning shares


```solidity
function redeem(
    ILeverageToken token,
    uint256 equityInCollateralAsset,
    uint256 maxShares,
    uint256 maxSwapCostInCollateralAsset,
    ISwapAdapter.SwapContext memory swapContext
) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to redeem|
|`equityInCollateralAsset`|`uint256`|The amount of equity to receive by redeeming LeverageToken. Denominated in the collateral asset of the LeverageToken|
|`maxShares`|`uint256`|Maximum shares (LeverageTokens) to redeem|
|`maxSwapCostInCollateralAsset`|`uint256`|The maximum amount of equity to pay for the redeem of the LeverageToken to use to help repay the debt flash loan due to the swap of debt to collateral being unfavorable|
|`swapContext`|`ISwapAdapter.SwapContext`|Swap context to use for the swap (which DEX to use, the route, tick spacing, etc.)|


## Errors
### MaxSwapCostExceeded
Error thrown when the cost of a swap exceeds the maximum allowed cost


```solidity
error MaxSwapCostExceeded(uint256 actualCost, uint256 maxCost);
```

**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`actualCost`|`uint256`|The actual cost of the swap|
|`maxCost`|`uint256`|The maximum allowed cost of the swap|

### Unauthorized
Error thrown when the caller is not authorized to execute a function


```solidity
error Unauthorized();
```

