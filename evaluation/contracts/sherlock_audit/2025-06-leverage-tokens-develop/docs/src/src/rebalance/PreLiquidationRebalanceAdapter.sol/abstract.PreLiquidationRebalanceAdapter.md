# PreLiquidationRebalanceAdapter
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/1dbcbcfe9a8bcf9392b2ada63dd8f1827a90783b/src/rebalance/PreLiquidationRebalanceAdapter.sol)

**Inherits:**
Initializable, [IPreLiquidationRebalanceAdapter](/src/interfaces/IPreLiquidationRebalanceAdapter.sol/interface.IPreLiquidationRebalanceAdapter.md)

*The PreLiquidationRebalanceAdapter is an abstract contract that implements the IPreLiquidationRebalanceAdapter interface.
It is intended to be used to create pre-liquidation rebalance mechanisms for LeverageTokens.
The PreLiquidationRebalanceAdapter is initialized for a LeverageToken with a collateral ratio threshold and a rebalance reward.
The `isEligibleForRebalance` function will return true if the current collateral ratio of the LeverageToken is below the configured
collateral ratio threshold, allowing for a rebalance action to be performed on LeverageToken on the LeverageManager.
The PreLiquidationRebalanceAdapter is also initialized with a rebalance reward, which is a flat percentage that is applied to the
liquidation penalty of the underlying lending pool used by the LeverageToken. The result is the amount of equity that the rebalancer
can earn for rebalancing the LeverageToken. It is expected that the rebalance reward is set to a value that is less than the liquidation penalty,
but high enough such that rebalancing is attractive to rebalancers.*


## State Variables
### WAD

```solidity
uint256 internal constant WAD = 1e18;
```


### REWARD_BASE
Reward base, 100_00 means that the reward is 100%


```solidity
uint256 public constant REWARD_BASE = 1e4;
```


## Functions
### _getPreLiquidationRebalanceAdapterStorage


```solidity
function _getPreLiquidationRebalanceAdapterStorage()
    internal
    pure
    returns (PreLiquidationRebalanceAdapterStorage storage $);
```

### __PreLiquidationRebalanceAdapter_init


```solidity
function __PreLiquidationRebalanceAdapter_init(uint256 collateralRatioThreshold, uint256 rebalanceReward)
    internal
    onlyInitializing;
```

### __PreLiquidationRebalanceAdapter_init_unchained


```solidity
function __PreLiquidationRebalanceAdapter_init_unchained(uint256 collateralRatioThreshold, uint256 rebalanceReward)
    internal
    onlyInitializing;
```

### getLeverageManager

Returns the LeverageManager contract


```solidity
function getLeverageManager() public view virtual returns (ILeverageManager);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`ILeverageManager`|leverageManager The LeverageManager contract|


### getCollateralRatioThreshold

Returns the collateral ratio threshold for pre-liquidation rebalancing

*When the LeverageToken collateral ratio is below this threshold, the LeverageToken can be pre-liquidation
rebalanced*


```solidity
function getCollateralRatioThreshold() public view returns (uint256);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|collateralRatioThreshold The collateral ratio threshold for pre-liquidation rebalancing|


### getRebalanceReward

Returns the rebalance reward percentage

*The rebalance reward represents the percentage of liquidation cost that will be rewarded to the caller of the
rebalance function. 10000 means 100%*


```solidity
function getRebalanceReward() public view returns (uint256);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|rebalanceRewardPercentage The rebalance reward percentage|


### isStateAfterRebalanceValid

Returns true if the state after rebalance is valid


```solidity
function isStateAfterRebalanceValid(ILeverageToken token, LeverageTokenState memory stateBefore)
    public
    view
    virtual
    returns (bool);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|The LeverageToken|
|`stateBefore`|`LeverageTokenState`|The state before rebalance|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`bool`|isValid True if the state after rebalance is valid|


### isEligibleForRebalance

Returns true if the LeverageToken is eligible for pre-liquidation rebalance

*Token is eligible for pre-liquidation rebalance if health factor is below the threshold*


```solidity
function isEligibleForRebalance(ILeverageToken, LeverageTokenState memory state, address)
    public
    view
    virtual
    returns (bool);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`ILeverageToken`||
|`state`|`LeverageTokenState`||
|`<none>`|`address`||

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`bool`|isEligible True if the LeverageToken is eligible for pre-liquidation rebalance|


## Structs
### PreLiquidationRebalanceAdapterStorage
*Struct containing all state for the PreLiquidationRebalanceAdapter contract*

**Note:**
storage-location: erc7201:seamless.contracts.storage.PreLiquidationRebalanceAdapter


```solidity
struct PreLiquidationRebalanceAdapterStorage {
    uint256 collateralRatioThreshold;
    uint256 rebalanceReward;
}
```

