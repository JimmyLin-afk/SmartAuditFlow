# IDutchAuctionRebalanceAdapter
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/1dbcbcfe9a8bcf9392b2ada63dd8f1827a90783b/src/interfaces/IDutchAuctionRebalanceAdapter.sol)


## Functions
### getLeverageManager

Returns the LeverageManager


```solidity
function getLeverageManager() external view returns (ILeverageManager leverageManager);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`leverageManager`|`ILeverageManager`|The LeverageManager|


### getLeverageToken

Returns the LeverageToken


```solidity
function getLeverageToken() external view returns (ILeverageToken leverageToken);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`leverageToken`|`ILeverageToken`|The LeverageToken|


### getAuction

Returns the current ongoing auction, if one exists

*If there is no ongoing auction, this function will return a un-initialized Auction struct*


```solidity
function getAuction() external view returns (Auction memory auction);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`auction`|`Auction`|The current ongoing auction, if one exists|


### getAuctionDuration

Returns the maximum duration of all auctions in seconds


```solidity
function getAuctionDuration() external view returns (uint120 auctionDuration);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`auctionDuration`|`uint120`|The maximum duration of all auctions in seconds|


### getInitialPriceMultiplier

Returns the initial price multiplier for all auctions


```solidity
function getInitialPriceMultiplier() external view returns (uint256 initialPriceMultiplier);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`initialPriceMultiplier`|`uint256`|The initial price multiplier for all auctions|


### getMinPriceMultiplier

Returns the minimum price multiplier for all auctions


```solidity
function getMinPriceMultiplier() external view returns (uint256 minPriceMultiplier);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`minPriceMultiplier`|`uint256`|The minimum price multiplier for all auctions|


### getLeverageTokenTargetCollateralRatio

Returns target collateral ratio for the LeverageToken


```solidity
function getLeverageTokenTargetCollateralRatio() external view returns (uint256 targetCollateralRatio);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`targetCollateralRatio`|`uint256`|Target collateral ratio|


### getLeverageTokenRebalanceStatus

Returns the LeverageToken's rebalance status


```solidity
function getLeverageTokenRebalanceStatus()
    external
    view
    returns (bool _isEligibleForRebalance, bool isOverCollateralized);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`_isEligibleForRebalance`|`bool`|True if the LeverageToken is eligible for rebalance, false otherwise|
|`isOverCollateralized`|`bool`|True if the LeverageToken is over-collateralized, false otherwise|


### getCurrentAuctionMultiplier

Returns the current auction multiplier

*This module uses exponential approximation (1-x)^4 to calculate the current auction multiplier*


```solidity
function getCurrentAuctionMultiplier() external view returns (uint256 multiplier);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`multiplier`|`uint256`|The current auction multiplier|


### isEligibleForRebalance

Returns true if the LeverageToken is eligible for rebalance


```solidity
function isEligibleForRebalance(ILeverageToken token, LeverageTokenState memory state, address caller)
    external
    view
    returns (bool isEligible);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|The LeverageToken|
|`state`|`LeverageTokenState`|The state of the LeverageToken|
|`caller`|`address`|The caller of the function|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`isEligible`|`bool`|True if the LeverageToken is eligible for rebalance, false otherwise|


### isStateAfterRebalanceValid

Returns true if the LeverageToken state after rebalance is valid


```solidity
function isStateAfterRebalanceValid(ILeverageToken token, LeverageTokenState memory stateBefore)
    external
    view
    returns (bool isValid);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|The LeverageToken|
|`stateBefore`|`LeverageTokenState`|The state of the LeverageToken before rebalance|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`isValid`|`bool`|True if the LeverageToken state after rebalance is valid, false otherwise|


### isAuctionValid

Returns whether the current auction is valid


```solidity
function isAuctionValid() external view returns (bool isValid);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`isValid`|`bool`|Whether the current auction is valid|


### getAmountIn

Returns the amount of tokens to provide for a given amount of tokens to receive for the current auction

*If there is no valid auction in the current block, this function will still return a value based on the auction
saved in storage (whether that be the most recent auction or an un-initialized auction)*


```solidity
function getAmountIn(uint256 amountOut) external view returns (uint256 amountIn);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`amountOut`|`uint256`|The amount of tokens to receive|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`amountIn`|`uint256`|The amount of tokens to provide|


### createAuction

Creates a new auction for the LeverageToken that needs rebalancing


```solidity
function createAuction() external;
```

### endAuction

Ends the current auction


```solidity
function endAuction() external;
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


## Events
### DutchAuctionRebalanceAdapterInitialized
Event emitted when the Dutch auction rebalancer is initialized


```solidity
event DutchAuctionRebalanceAdapterInitialized(
    uint256 auctionDuration, uint256 initialPriceMultiplier, uint256 minPriceMultiplier
);
```

**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`auctionDuration`|`uint256`|The duration of auctions|
|`initialPriceMultiplier`|`uint256`|The initial price multiplier for auctions|
|`minPriceMultiplier`|`uint256`|The minimum price multiplier for auctions|

### LeverageTokenSet
Event emitted when the LeverageToken is set


```solidity
event LeverageTokenSet(ILeverageToken leverageToken);
```

**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`leverageToken`|`ILeverageToken`|The LeverageToken|

### AuctionCreated
Event emitted when a new auction is created


```solidity
event AuctionCreated(Auction auction);
```

**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`auction`|`Auction`|The auction|

### Take
Event emitted when an auction is taken


```solidity
event Take(address indexed taker, uint256 amountIn, uint256 amountOut);
```

**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`taker`|`address`|The taker of the auction|
|`amountIn`|`uint256`|The amount of tokens provided|
|`amountOut`|`uint256`|The amount of tokens received|

### AuctionEnded
Event emitted when an auction ends


```solidity
event AuctionEnded();
```

## Errors
### LeverageTokenAlreadySet
Error thrown when the LeverageToken is already set


```solidity
error LeverageTokenAlreadySet();
```

### AuctionNotValid
Error thrown when an auction is not valid


```solidity
error AuctionNotValid();
```

### AuctionStillValid
Error thrown when an auction is still valid


```solidity
error AuctionStillValid();
```

### LeverageTokenNotEligibleForRebalance
Error thrown when the LeverageToken is not eligible for rebalance


```solidity
error LeverageTokenNotEligibleForRebalance();
```

### InvalidAuctionDuration
Error thrown when attempting to set an auction duration of zero


```solidity
error InvalidAuctionDuration();
```

### MinPriceMultiplierTooHigh
Error thrown when the minimum price multiplier is higher than the initial price multiplier


```solidity
error MinPriceMultiplierTooHigh();
```

