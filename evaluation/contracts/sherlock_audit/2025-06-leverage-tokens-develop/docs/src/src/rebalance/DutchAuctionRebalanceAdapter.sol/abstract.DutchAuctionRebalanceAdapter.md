# DutchAuctionRebalanceAdapter
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/1dbcbcfe9a8bcf9392b2ada63dd8f1827a90783b/src/rebalance/DutchAuctionRebalanceAdapter.sol)

**Inherits:**
[IDutchAuctionRebalanceAdapter](/src/interfaces/IDutchAuctionRebalanceAdapter.sol/interface.IDutchAuctionRebalanceAdapter.md), Initializable

*The DutchAuctionRebalanceAdapter is a periphery abstract contract that implements the IDutchAuctionRebalanceAdapter interface.
It is used to create Dutch auctions to determine the price of a rebalance action for a LeverageToken.
The DutchAuctionRebalanceAdapter is initialized for a LeverageToken with an auction duration, initial price multiplier,
and min price multiplier.
When the LeverageToken is eligible for rebalance, an auction can be created. The auction will run for the duration specified
by the auction duration and follow an exponential decay curve (following the exponential approximation (1-x)^4)) starting at
the initial price multiplier, decreasing towards the min price multiplier.
When a rebalancer sees a favorable price, they can call `take` to rebalance the LeverageToken. The `take` function will either
decrease or increase the collateral ratio of the LeverageToken, depending on the current collateral ratio of the LeverageToken.
If the LeverageToken is over-collateralized, the rebalancer will borrow debt and add collateral. If the LeverageToken is
under-collateralized, the rebalancer will repay debt and remove collateral.
Note: If the auction is no longer valid, `take` will revert*


## State Variables
### PRICE_MULTIPLIER_PRECISION

```solidity
uint256 public constant PRICE_MULTIPLIER_PRECISION = 1e18;
```


## Functions
### _getDutchAuctionRebalanceAdapterStorage


```solidity
function _getDutchAuctionRebalanceAdapterStorage()
    internal
    pure
    returns (DutchAuctionRebalanceAdapterStorage storage $);
```

### __DutchAuctionRebalanceAdapter_init


```solidity
function __DutchAuctionRebalanceAdapter_init(
    uint120 _auctionDuration,
    uint256 _initialPriceMultiplier,
    uint256 _minPriceMultiplier
) internal onlyInitializing;
```

### __DutchAuctionRebalanceAdapter_init_unchained


```solidity
function __DutchAuctionRebalanceAdapter_init_unchained(
    uint120 _auctionDuration,
    uint256 _initialPriceMultiplier,
    uint256 _minPriceMultiplier
) internal onlyInitializing;
```

### _setLeverageToken

Sets the LeverageToken for the DutchAuctionRebalanceAdapter


```solidity
function _setLeverageToken(ILeverageToken leverageToken) internal;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`leverageToken`|`ILeverageToken`|The LeverageToken to set|


### getLeverageManager

Returns the LeverageManager


```solidity
function getLeverageManager() public view virtual returns (ILeverageManager);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`ILeverageManager`|leverageManager The LeverageManager|


### getLeverageToken

Returns the LeverageToken


```solidity
function getLeverageToken() public view returns (ILeverageToken);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`ILeverageToken`|leverageToken The LeverageToken|


### getAuction

Returns the current ongoing auction, if one exists

*If there is no ongoing auction, this function will return a un-initialized Auction struct*


```solidity
function getAuction() public view returns (Auction memory auction);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`auction`|`Auction`|The current ongoing auction, if one exists|


### getAuctionDuration

Returns the maximum duration of all auctions in seconds


```solidity
function getAuctionDuration() public view returns (uint120);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint120`|auctionDuration The maximum duration of all auctions in seconds|


### getInitialPriceMultiplier

Returns the initial price multiplier for all auctions


```solidity
function getInitialPriceMultiplier() public view returns (uint256);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|initialPriceMultiplier The initial price multiplier for all auctions|


### getMinPriceMultiplier

Returns the minimum price multiplier for all auctions


```solidity
function getMinPriceMultiplier() public view returns (uint256);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|minPriceMultiplier The minimum price multiplier for all auctions|


### getLeverageTokenTargetCollateralRatio

Returns target collateral ratio for the LeverageToken


```solidity
function getLeverageTokenTargetCollateralRatio() public view virtual returns (uint256 targetCollateralRatio);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`targetCollateralRatio`|`uint256`|Target collateral ratio|


### getLeverageTokenRebalanceStatus

Returns the LeverageToken's rebalance status


```solidity
function getLeverageTokenRebalanceStatus() public view returns (bool isEligible, bool isOverCollateralized);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`isEligible`|`bool`|_isEligibleForRebalance True if the LeverageToken is eligible for rebalance, false otherwise|
|`isOverCollateralized`|`bool`|True if the LeverageToken is over-collateralized, false otherwise|


### isAuctionValid

Returns whether the current auction is valid


```solidity
function isAuctionValid() public view returns (bool isValid);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`isValid`|`bool`|Whether the current auction is valid|


### getCurrentAuctionMultiplier

Returns the current auction multiplier

*This module uses exponential approximation (1-x)^4 to calculate the current auction multiplier*


```solidity
function getCurrentAuctionMultiplier() public view returns (uint256);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|multiplier The current auction multiplier|


### getAmountIn

Returns the amount of tokens to provide for a given amount of tokens to receive for the current auction

*If there is no valid auction in the current block, this function will still return a value based on the auction
saved in storage (whether that be the most recent auction or an un-initialized auction)*


```solidity
function getAmountIn(uint256 amountOut) public view returns (uint256);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`amountOut`|`uint256`|The amount of tokens to receive|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|amountIn The amount of tokens to provide|


### createAuction

Creates a new auction for the LeverageToken that needs rebalancing


```solidity
function createAuction() external;
```

### endAuction

Ends the current auction


```solidity
function endAuction() public;
```

### take

Takes part in the current auction at the current price

*To preview the amount of tokens to provide, the `getAmountIn` function can be used*


```solidity
function take(uint256 amountOut) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`amountOut`|`uint256`|The amount of tokens to receive|


### _executeRebalanceDown

Executes the rebalance down operation, meaning decreasing collateral ratio

*This function prepares rebalance parameters, takes collateral token from sender, executes rebalance and returns debt token to sender*


```solidity
function _executeRebalanceDown(uint256 collateralAmount, uint256 debtAmount) internal;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`collateralAmount`|`uint256`|Amount of collateral to add|
|`debtAmount`|`uint256`|Amount of debt to borrow|


### _executeRebalanceUp

Executes the rebalance up operation, meaning increasing collateral ratio

*This function prepares rebalance parameters, takes debt token from sender, executes rebalance and returns collateral token to sender*


```solidity
function _executeRebalanceUp(uint256 collateralAmount, uint256 debtAmount) internal;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`collateralAmount`|`uint256`|Amount of collateral to remove|
|`debtAmount`|`uint256`|Amount of debt to repay|


### isEligibleForRebalance

Returns true if the LeverageToken is eligible for rebalance


```solidity
function isEligibleForRebalance(ILeverageToken, LeverageTokenState memory, address caller)
    public
    view
    virtual
    returns (bool);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`ILeverageToken`||
|`<none>`|`LeverageTokenState`||
|`caller`|`address`|The caller of the function|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`bool`|isEligible True if the LeverageToken is eligible for rebalance, false otherwise|


### isStateAfterRebalanceValid

Returns true if the LeverageToken state after rebalance is valid


```solidity
function isStateAfterRebalanceValid(ILeverageToken, LeverageTokenState memory) public view virtual returns (bool);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`ILeverageToken`||
|`<none>`|`LeverageTokenState`||

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`bool`|isValid True if the LeverageToken state after rebalance is valid, false otherwise|


## Structs
### DutchAuctionRebalanceAdapterStorage
*Struct containing all state for the DutchAuctionRebalanceAdapter contract*

**Note:**
storage-location: erc7201:seamless.contracts.storage.DutchAuctionRebalanceAdapter


```solidity
struct DutchAuctionRebalanceAdapterStorage {
    ILeverageToken leverageToken;
    Auction auction;
    uint120 auctionDuration;
    uint256 initialPriceMultiplier;
    uint256 minPriceMultiplier;
}
```

