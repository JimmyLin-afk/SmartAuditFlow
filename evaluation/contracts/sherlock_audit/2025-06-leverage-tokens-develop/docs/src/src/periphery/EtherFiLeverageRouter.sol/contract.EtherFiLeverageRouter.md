# EtherFiLeverageRouter
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/40214436ae3956021858cb95e6ff881f6ede8e11/src/periphery/EtherFiLeverageRouter.sol)

**Inherits:**
[LeverageRouterMintBase](/src/periphery/LeverageRouterMintBase.sol/abstract.LeverageRouterMintBase.md), [IEtherFiLeverageRouter](/src/interfaces/periphery/IEtherFiLeverageRouter.sol/interface.IEtherFiLeverageRouter.md)

*The EtherFiLeverageRouter contract is an immutable periphery contract that facilitates the use of Morpho flash loans
to deposit equity into LeverageTokens that use weETH as collateral and WETH as debt.
The high-level deposit flow is as follows:
1. The user calls `deposit` with the amount of weETH equity to deposit, and the minimum amount of shares (LeverageTokens)
to receive.
2. The EtherFiLeverageRouter will flash loan the additional required weETH from Morpho.
3. The EtherFiLeverageRouter will use the flash loaned weETH and the weETH equity from the sender for the deposit into
the LeverageToken, receiving LeverageTokens and WETH debt in return.
4. The EtherFiLeverageRouter will unwrap the WETH debt to ETH and deposit the ETH into the EtherFi L2 Mode Sync Pool
to obtain weETH.
5. The weETH received from the EtherFi L2 Mode Sync Pool is used to repay the flash loan to Morpho.
6. The EtherFiLeverageRouter will transfer the LeverageTokens and any remaining weETH to the sender.*

*Note: This router is intended to be used for LeverageTokens that use weETH as collateral and WETH as debt and will
otherwise revert.*


## State Variables
### ETH_ADDRESS
The ETH address per the EtherFi L2 Mode Sync Pool contract


```solidity
address internal constant ETH_ADDRESS = 0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE;
```


### etherFiL2ModeSyncPool
The EtherFi L2 Mode Sync Pool contract


```solidity
IEtherFiL2ModeSyncPool public immutable etherFiL2ModeSyncPool;
```


## Functions
### constructor

Creates a new EtherFiLeverageRouter


```solidity
constructor(ILeverageManager _leverageManager, IMorpho _morpho, IEtherFiL2ModeSyncPool _etherFiL2ModeSyncPool)
    LeverageRouterMintBase(_leverageManager, _morpho);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_leverageManager`|`ILeverageManager`|The LeverageManager contract|
|`_morpho`|`IMorpho`|The Morpho core protocol contract|
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


### onMorphoFlashLoan

Morpho flash loan callback function


```solidity
function onMorphoFlashLoan(uint256 loanAmount, bytes calldata data) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`loanAmount`|`uint256`|Amount of asset flash loaned|
|`data`|`bytes`|Encoded data passed to `morpho.flashLoan`|


### _getCollateralFromDebt

Performs logic to obtain weETH collateral from WETH debt


```solidity
function _getCollateralFromDebt(IERC20 weth, uint256 wethAmount, uint256 weethLoanAmount, bytes memory)
    internal
    override
    returns (uint256);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`weth`|`IERC20`|The WETH contract|
|`wethAmount`|`uint256`|The amount of WETH debt to convert to weETH collateral|
|`weethLoanAmount`|`uint256`|The amount of weETH flash loaned|
|`<none>`|`bytes`||

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|The amount of weETH collateral obtained|


