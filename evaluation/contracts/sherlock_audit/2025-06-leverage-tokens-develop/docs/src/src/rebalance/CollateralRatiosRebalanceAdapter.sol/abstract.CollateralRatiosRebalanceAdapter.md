# CollateralRatiosRebalanceAdapter
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/1dbcbcfe9a8bcf9392b2ada63dd8f1827a90783b/src/rebalance/CollateralRatiosRebalanceAdapter.sol)

**Inherits:**
[ICollateralRatiosRebalanceAdapter](/src/interfaces/ICollateralRatiosRebalanceAdapter.sol/interface.ICollateralRatiosRebalanceAdapter.md), Initializable

*The CollateralRatiosRebalanceAdapter is an abstract contract that implements the ICollateralRatiosRebalanceAdapter interface.
The CollateralRatiosRebalanceAdapter is initialized for a LeverageToken with a minimum collateral ratio, target collateral ratio, and maximum collateral ratio.
The `isEligibleForRebalance` function will return true if the current collateral ratio of the LeverageToken is below the configured
minimum collateral ratio or above the configured maximum collateral ratio, allowing for a rebalance action to be performed on the LeverageToken.
The `isStateAfterRebalanceValid` function will return true if the collateral ratio is better than before:
- The collateral ratio is closer to the target collateral ratio than before
- If the collateral ratio was below the target collateral ratio, the collateral ratio is still below the target collateral ratio or equal to it
- If the collateral ratio was above the target collateral ratio, the collateral ratio is still above the target collateral ratio or equal to it*


## Functions
### _getCollateralRatiosRebalanceAdapterStorage


```solidity
function _getCollateralRatiosRebalanceAdapterStorage()
    internal
    pure
    returns (CollateralRatiosRebalanceAdapterStorage storage $);
```

### __CollateralRatiosRebalanceAdapter_init


```solidity
function __CollateralRatiosRebalanceAdapter_init(
    uint256 minCollateralRatio,
    uint256 targetCollateralRatio,
    uint256 maxCollateralRatio
) internal onlyInitializing;
```

### __CollateralRatiosRebalanceAdapter_init_unchained


```solidity
function __CollateralRatiosRebalanceAdapter_init_unchained(
    uint256 minCollateralRatio,
    uint256 targetCollateralRatio,
    uint256 maxCollateralRatio
) internal onlyInitializing;
```

### getLeverageManager

Returns the LeverageManager


```solidity
function getLeverageManager() public view virtual returns (ILeverageManager);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`ILeverageManager`|leverageManager The LeverageManager|


### getLeverageTokenMinCollateralRatio

Returns the minimum collateral ratio for a LeverageToken


```solidity
function getLeverageTokenMinCollateralRatio() public view returns (uint256);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|minCollateralRatio Minimum collateral ratio for the LeverageToken|


### getLeverageTokenTargetCollateralRatio

Returns the target collateral ratio for a LeverageToken


```solidity
function getLeverageTokenTargetCollateralRatio() public view virtual returns (uint256);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|targetCollateralRatio Target collateral ratio for the LeverageToken|


### getLeverageTokenMaxCollateralRatio

Returns the maximum collateral ratio for a LeverageToken


```solidity
function getLeverageTokenMaxCollateralRatio() public view returns (uint256);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|maxCollateralRatio Maximum collateral ratio for the LeverageToken|


### getLeverageTokenInitialCollateralRatio

Returns the initial collateral ratio for a LeverageToken


```solidity
function getLeverageTokenInitialCollateralRatio(ILeverageToken) public view virtual returns (uint256);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|initialCollateralRatio Initial collateral ratio for the LeverageToken|


### isEligibleForRebalance

Returns true if the LeverageToken is eligible for rebalance


```solidity
function isEligibleForRebalance(ILeverageToken token, LeverageTokenState memory state, address)
    public
    view
    virtual
    returns (bool isEligible);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|The LeverageToken|
|`state`|`LeverageTokenState`|The state of the LeverageToken|
|`<none>`|`address`||

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`isEligible`|`bool`|True if the LeverageToken is eligible for rebalance, false otherwise|


### isStateAfterRebalanceValid

Returns true if the LeverageToken state after rebalance is valid


```solidity
function isStateAfterRebalanceValid(ILeverageToken token, LeverageTokenState memory stateBefore)
    public
    view
    virtual
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


## Structs
### CollateralRatiosRebalanceAdapterStorage
*Struct containing all state for the CollateralRatiosRebalanceAdapter contract*

**Note:**
storage-location: erc7201:seamless.contracts.storage.CollateralRatiosRebalanceAdapter


```solidity
struct CollateralRatiosRebalanceAdapterStorage {
    uint256 minCollateralRatio;
    uint256 targetCollateralRatio;
    uint256 maxCollateralRatio;
}
```

