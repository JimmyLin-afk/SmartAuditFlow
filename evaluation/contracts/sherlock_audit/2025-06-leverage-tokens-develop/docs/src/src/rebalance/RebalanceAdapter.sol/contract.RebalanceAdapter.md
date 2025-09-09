# RebalanceAdapter
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/1dbcbcfe9a8bcf9392b2ada63dd8f1827a90783b/src/rebalance/RebalanceAdapter.sol)

**Inherits:**
[IRebalanceAdapter](/src/interfaces/IRebalanceAdapter.sol/interface.IRebalanceAdapter.md), UUPSUpgradeable, OwnableUpgradeable, [CollateralRatiosRebalanceAdapter](/src/rebalance/CollateralRatiosRebalanceAdapter.sol/abstract.CollateralRatiosRebalanceAdapter.md), [DutchAuctionRebalanceAdapter](/src/rebalance/DutchAuctionRebalanceAdapter.sol/abstract.DutchAuctionRebalanceAdapter.md), [PreLiquidationRebalanceAdapter](/src/rebalance/PreLiquidationRebalanceAdapter.sol/abstract.PreLiquidationRebalanceAdapter.md)

*The RebalanceAdapter contract is an upgradeable periphery contract that implements the IRebalanceAdapter interface.
LeverageTokens configured on the LeverageManager must specify a RebalanceAdapter, which defines hooks for determining
when a LeverageToken can be rebalanced and if a rebalance action is valid, and how rebalancers should be rewarded.
This RebalanceAdapter utilizes the DutchAuctionRebalanceAdapter, MinMaxCollateralRatioRebalanceAdapter, and
PreLiquidationRebalanceAdapter abstract contracts.
- The DutchAuctionRebalanceAdapter creates Dutch auctions to determine the price of a rebalance action
- The MinMaxCollateralRatioRebalanceAdapter ensures that the collateral ratio of a LeverageToken must be outside
of a specified range before a rebalance action can be performed.
- The PreLiquidationRebalanceAdapter allows for fast-tracking rebalance operations for LeverageTokens that are below
a specified collateral ratio threshold. The intention is that this acts as a pre-liquidation rebalance mechanism
in cases that the dutch auction price is too slow to react to a dramatic drop in collateral ratio.*


## Functions
### _getRebalanceAdapterStorage


```solidity
function _getRebalanceAdapterStorage() internal pure returns (RebalanceAdapterStorage storage $);
```

### _authorizeUpgrade


```solidity
function _authorizeUpgrade(address newImplementation) internal override onlyOwner;
```

### initialize


```solidity
function initialize(RebalanceAdapterInitParams memory params) external initializer;
```

### postLeverageTokenCreation

Post-LeverageToken creation hook. Used for any validation logic or initialization after a LeverageToken
is created using this adapter

*This function is called in `LeverageManager.createNewLeverageToken` after the new LeverageToken is created*


```solidity
function postLeverageTokenCreation(address creator, address leverageToken) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`creator`|`address`|The address of the creator of the LeverageToken|
|`leverageToken`|`address`|The address of the LeverageToken that was created|


### getAuthorizedCreator

Returns the authorized creator of the RebalanceAdapter


```solidity
function getAuthorizedCreator() public view returns (address);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`address`|authorizedCreator The authorized creator of the RebalanceAdapter|


### getLeverageManager

Returns the LeverageManager of the RebalanceAdapter


```solidity
function getLeverageManager()
    public
    view
    override(
        IRebalanceAdapter, DutchAuctionRebalanceAdapter, CollateralRatiosRebalanceAdapter, PreLiquidationRebalanceAdapter
    )
    returns (ILeverageManager);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`ILeverageManager`|leverageManager The LeverageManager of the RebalanceAdapter|


### getLeverageTokenInitialCollateralRatio

Returns the initial collateral ratio for a LeverageToken

*Initial collateral ratio is followed when the LeverageToken has no shares and on mints when debt is 0.*


```solidity
function getLeverageTokenInitialCollateralRatio(ILeverageToken token)
    public
    view
    override(IRebalanceAdapterBase, CollateralRatiosRebalanceAdapter)
    returns (uint256);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to get initial collateral ratio for|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|initialCollateralRatio Initial collateral ratio for the LeverageToken|


### getLeverageTokenTargetCollateralRatio

Returns target collateral ratio for the LeverageToken


```solidity
function getLeverageTokenTargetCollateralRatio()
    public
    view
    override(DutchAuctionRebalanceAdapter, CollateralRatiosRebalanceAdapter)
    returns (uint256 targetCollateralRatio);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`targetCollateralRatio`|`uint256`|Target collateral ratio|


### isEligibleForRebalance

Validates if a LeverageToken is eligible for rebalance


```solidity
function isEligibleForRebalance(ILeverageToken token, LeverageTokenState memory state, address caller)
    public
    view
    override(
        IRebalanceAdapterBase,
        DutchAuctionRebalanceAdapter,
        CollateralRatiosRebalanceAdapter,
        PreLiquidationRebalanceAdapter
    )
    returns (bool);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to check eligibility for|
|`state`|`LeverageTokenState`|State of the LeverageToken|
|`caller`|`address`|Caller of the function|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`bool`|isEligible True if LeverageToken is eligible for rebalance, false otherwise|


### isStateAfterRebalanceValid

Validates if the LeverageToken's state after rebalance is valid


```solidity
function isStateAfterRebalanceValid(ILeverageToken token, LeverageTokenState memory stateBefore)
    public
    view
    override(
        IRebalanceAdapterBase,
        DutchAuctionRebalanceAdapter,
        CollateralRatiosRebalanceAdapter,
        PreLiquidationRebalanceAdapter
    )
    returns (bool);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`token`|`ILeverageToken`|LeverageToken to validate state for|
|`stateBefore`|`LeverageTokenState`|State of the LeverageToken before rebalance|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`bool`|isValid True if state after rebalance is valid, false otherwise|


## Structs
### RebalanceAdapterStorage
*Struct containing all state for the RebalanceAdapter contract*

**Note:**
storage-location: erc7201:seamless.contracts.storage.RebalanceAdapter


```solidity
struct RebalanceAdapterStorage {
    address authorizedCreator;
    ILeverageManager leverageManager;
}
```

### RebalanceAdapterInitParams

```solidity
struct RebalanceAdapterInitParams {
    address owner;
    address authorizedCreator;
    ILeverageManager leverageManager;
    uint256 minCollateralRatio;
    uint256 targetCollateralRatio;
    uint256 maxCollateralRatio;
    uint120 auctionDuration;
    uint256 initialPriceMultiplier;
    uint256 minPriceMultiplier;
    uint256 preLiquidationCollateralRatioThreshold;
    uint256 rebalanceReward;
}
```

