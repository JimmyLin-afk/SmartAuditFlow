# IEtherFiLeverageRouter
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/40214436ae3956021858cb95e6ff881f6ede8e11/src/interfaces/periphery/IEtherFiLeverageRouter.sol)

**Inherits:**
[ILeverageRouterBase](/src/interfaces/periphery/ILeverageRouterBase.sol/interface.ILeverageRouterBase.md)


## Functions
### etherFiL2ModeSyncPool

The EtherFi L2 Mode Sync Pool contract


```solidity
function etherFiL2ModeSyncPool() external view returns (IEtherFiL2ModeSyncPool _etherFiL2ModeSyncPool);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`_etherFiL2ModeSyncPool`|`IEtherFiL2ModeSyncPool`|The EtherFi L2 Mode Sync Pool contract|


### mint

Mints LeverageTokens (shares) that use weETH as collateral and WETH as debt

*Transfers `equityInCollateralAsset` of weETH to the LeverageRouter, flash loans the additional weETH collateral
required to add the equity to the LeverageToken, receives WETH debt, then unwraps the WETH debt to ETH and deposits
the ETH into the EtherFi L2 Mode Sync Pool to obtain weETH. The received weETH is used to repay the flash loan*


```solidity
function mint(ILeverageToken token, uint256 equityInCollateralAsset, uint256 minShares) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to mint|
|`equityInCollateralAsset`|`uint256`|The amount of weETH equity to add to the LeverageToken and mint shares for.|
|`minShares`|`uint256`|Minimum shares (LeverageTokens) to receive from the mint|


