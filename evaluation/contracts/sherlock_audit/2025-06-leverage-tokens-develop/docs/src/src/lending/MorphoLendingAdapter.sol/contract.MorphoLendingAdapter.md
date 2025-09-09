# MorphoLendingAdapter
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/1dbcbcfe9a8bcf9392b2ada63dd8f1827a90783b/src/lending/MorphoLendingAdapter.sol)

**Inherits:**
[IMorphoLendingAdapter](/src/interfaces/IMorphoLendingAdapter.sol/interface.IMorphoLendingAdapter.md), Initializable

*The MorphoLendingAdapter is an adapter to interface with Morpho markets. LeverageToken creators can configure their LeverageToken
to use a MorphoLendingAdapter to use Morpho as the lending protocol for their LeverageToken.
The MorphoLendingAdapter uses the underlying oracle of the Morpho market to convert between the collateral and debt asset. It also
uses Morpho's libraries to calculate the collateral and debt held by the adapter, including any accrued interest.
Note: `getDebt` uses `MorphoBalancesLib.expectedBorrowAssets` which calculates the total debt of the adapter based on the Morpho
market's borrow shares owned by the adapter. This logic rounds up, so it is possible that `getDebt` returns a value that is
greater than the actual debt owed to the Morpho market.*


## State Variables
### WAD

```solidity
uint256 internal constant WAD = 1e18;
```


### leverageManager
The LeverageManager contract


```solidity
ILeverageManager public immutable leverageManager;
```


### morpho
The Morpho core protocol contract


```solidity
IMorpho public immutable morpho;
```


### morphoMarketId
The ID of the Morpho market that the MorphoLendingAdapter manages a position in


```solidity
Id public morphoMarketId;
```


### marketParams
The market parameters of the Morpho lending pool


```solidity
MarketParams public marketParams;
```


### authorizedCreator
The authorized creator of the MorphoLendingAdapter

*Only the authorized creator can create a new LeverageToken using this adapter on the LeverageManager*


```solidity
address public authorizedCreator;
```


### isUsed
Whether the MorphoLendingAdapter is in use

*If this is true, the MorphoLendingAdapter cannot be used to create a new LeverageToken*


```solidity
bool public isUsed;
```


## Functions
### onlyLeverageManager

*Reverts if the caller is not the stored LeverageManager address*


```solidity
modifier onlyLeverageManager();
```

### constructor

Creates a new MorphoLendingAdapter


```solidity
constructor(ILeverageManager _leverageManager, IMorpho _morpho);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_leverageManager`|`ILeverageManager`|The LeverageManager contract|
|`_morpho`|`IMorpho`|The Morpho core protocol contract|


### initialize

Initializes the MorphoLendingAdapter


```solidity
function initialize(Id _morphoMarketId, address _authorizedCreator) external initializer;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_morphoMarketId`|`Id`|The Morpho market ID|
|`_authorizedCreator`|`address`|The authorized creator of this MorphoLendingAdapter. The authorized creator can create a new LeverageToken using this adapter on the LeverageManager|


### postLeverageTokenCreation

Post-LeverageToken creation hook. Used for any validation logic or initialization after a LeverageToken
is created using this adapter

*This function is called in `LeverageManager.createNewLeverageToken` after the new LeverageToken is created*


```solidity
function postLeverageTokenCreation(address creator, address) external onlyLeverageManager;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`creator`|`address`|The address of the creator of the LeverageToken|
|`<none>`|`address`||


### getCollateralAsset

Returns the address of the collateral asset


```solidity
function getCollateralAsset() external view returns (IERC20);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`IERC20`|collateralAsset Address of the collateral asset|


### getDebtAsset

Returns the address of the debt asset


```solidity
function getDebtAsset() external view returns (IERC20);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`IERC20`|debtAsset Address of the debt asset|


### convertCollateralToDebtAsset

Converts an amount of collateral asset to a debt asset amount based on the lending pool oracle


```solidity
function convertCollateralToDebtAsset(uint256 collateral) public view returns (uint256);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`collateral`|`uint256`|Collateral amount|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|debt Amount of debt asset|


### convertDebtToCollateralAsset

Converts an amount of debt asset to a collateral asset amount based on the lending pool oracle


```solidity
function convertDebtToCollateralAsset(uint256 debt) public view returns (uint256);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`debt`|`uint256`|Debt amount|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|collateral Amount of collateral asset|


### getCollateral

Returns total collateral of the position held by the lending adapter


```solidity
function getCollateral() public view returns (uint256);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|collateral Total collateral of the position held by the lending adapter|


### getCollateralInDebtAsset

Returns the total collateral of the position held by the lending adapter denominated in the debt asset


```solidity
function getCollateralInDebtAsset() public view returns (uint256);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|collateral Total collateral of the position held by the lending adapter denominated in the debt asset|


### getDebt

Returns the total debt of the position held by the lending adapter


```solidity
function getDebt() public view returns (uint256);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|debt Total debt of the position held by the lending adapter|


### getEquityInCollateralAsset

Returns the total equity of the position held by the lending adapter denominated in the collateral asset


```solidity
function getEquityInCollateralAsset() external view returns (uint256);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|equity Equity of the position held by the lending adapter|


### getEquityInDebtAsset

Returns the total equity of the position held by the lending adapter denominated in the debt asset

*Equity is calculated as collateral - debt*


```solidity
function getEquityInDebtAsset() external view returns (uint256);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|equity Equity of the position held by the lending adapter|


### getLiquidationPenalty

Returns the liquidation penalty of the position held by the lending adapter

*1e18 means that the liquidation penalty is 100%*


```solidity
function getLiquidationPenalty() external view returns (uint256);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|liquidationPenalty Liquidation penalty of the position held by the lending adapter, scaled by 1e18|


### addCollateral

Supplies collateral assets to the lending pool


```solidity
function addCollateral(uint256 amount) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`amount`|`uint256`|Amount of assets to supply|


### removeCollateral

Withdraws collateral assets from the lending pool


```solidity
function removeCollateral(uint256 amount) external onlyLeverageManager;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`amount`|`uint256`|Amount of collateral assets to withdraw|


### borrow

Borrows debt assets from the lending pool


```solidity
function borrow(uint256 amount) external onlyLeverageManager;
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


