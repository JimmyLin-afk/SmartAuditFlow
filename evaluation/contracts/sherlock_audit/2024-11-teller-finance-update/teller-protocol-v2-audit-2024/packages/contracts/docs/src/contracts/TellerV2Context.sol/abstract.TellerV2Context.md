# TellerV2Context
[Git Source](https://github.com/teller-protocol/teller-protocol-v2/blob/f4bf5a00ae7113b0344876c13db9b3dd705154f6/contracts/TellerV2Context.sol)

**Inherits:**
[ERC2771ContextUpgradeable](/contracts/ERC2771ContextUpgradeable.sol/abstract.ERC2771ContextUpgradeable.md), [TellerV2Storage](/contracts/TellerV2Storage.sol/abstract.TellerV2Storage.md)

*This contract should not use any storage*


## Functions
### constructor


```solidity
constructor(address trustedForwarder) ERC2771ContextUpgradeable(trustedForwarder);
```

### isTrustedMarketForwarder

Checks if an address is a trusted forwarder contract for a given market.


```solidity
function isTrustedMarketForwarder(uint256 _marketId, address _trustedMarketForwarder) public view returns (bool);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|An ID for a lending market.|
|`_trustedMarketForwarder`|`address`|An address to check if is a trusted forwarder in the given market.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`bool`|A boolean indicating the forwarder address is trusted in a market.|


### hasApprovedMarketForwarder

Checks if an account has approved a forwarder for a market.


```solidity
function hasApprovedMarketForwarder(uint256 _marketId, address _forwarder, address _account)
    public
    view
    returns (bool);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|An ID for a lending market.|
|`_forwarder`|`address`|A forwarder contract address.|
|`_account`|`address`|The address to verify set an approval.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`bool`|A boolean indicating if an approval was set.|


### setTrustedMarketForwarder

Sets a trusted forwarder for a lending market.

The caller must owner the market given. See {MarketRegistry}


```solidity
function setTrustedMarketForwarder(uint256 _marketId, address _forwarder) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|An ID for a lending market.|
|`_forwarder`|`address`|A forwarder contract address.|


### approveMarketForwarder

Approves a forwarder contract to use their address as a sender for a specific market.

The forwarder given must be trusted by the market given.


```solidity
function approveMarketForwarder(uint256 _marketId, address _forwarder) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|An ID for a lending market.|
|`_forwarder`|`address`|A forwarder contract address.|


### _msgSenderForMarket

Retrieves the function caller address by checking the appended calldata if the _actual_ caller is a trusted forwarder.


```solidity
function _msgSenderForMarket(uint256 _marketId) internal view virtual returns (address);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|An ID for a lending market.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`address`|sender The address to use as the function caller.|


### _msgDataForMarket

Retrieves the actual function calldata from a trusted forwarder call.


```solidity
function _msgDataForMarket(uint256 _marketId) internal view virtual returns (bytes calldata);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|An ID for a lending market to verify if the caller is a trusted forwarder.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`bytes`|calldata The modified bytes array of the function calldata without the appended sender's address.|


## Events
### TrustedMarketForwarderSet

```solidity
event TrustedMarketForwarderSet(uint256 indexed marketId, address forwarder, address sender);
```

### MarketForwarderApproved

```solidity
event MarketForwarderApproved(uint256 indexed marketId, address indexed forwarder, address sender);
```

