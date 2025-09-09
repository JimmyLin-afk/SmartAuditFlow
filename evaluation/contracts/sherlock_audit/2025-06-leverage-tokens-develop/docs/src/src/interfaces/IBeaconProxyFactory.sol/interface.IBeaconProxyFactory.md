# IBeaconProxyFactory
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/1dbcbcfe9a8bcf9392b2ada63dd8f1827a90783b/src/interfaces/IBeaconProxyFactory.sol)


## Functions
### computeProxyAddress

Computes the address of a BeaconProxy before deployment


```solidity
function computeProxyAddress(address sender, bytes memory data, bytes32 baseSalt)
    external
    view
    returns (address proxy);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`sender`|`address`|The address that will deploy the BeaconProxy using the factory|
|`data`|`bytes`|The initialization data passed to the BeaconProxy|
|`baseSalt`|`bytes32`|The base salt used for deterministic deployment|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`proxy`|`address`|The predicted address of the BeaconProxy|


### numProxies

Returns the number of BeaconProxys deployed by the factory


```solidity
function numProxies() external view returns (uint256 _numProxies);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`_numProxies`|`uint256`|The number of BeaconProxys deployed by the factory|


### createProxy

Creates a new beacon proxy


```solidity
function createProxy(bytes memory data, bytes32 baseSalt) external returns (address proxy);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`data`|`bytes`|The initialization data passed to the proxy|
|`baseSalt`|`bytes32`|The base salt used for deterministic deployment|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`proxy`|`address`|The address of the new BeaconProxy|


## Events
### BeaconProxyCreated
Emitted when a new BeaconProxy is created


```solidity
event BeaconProxyCreated(address indexed proxy, bytes data, bytes32 baseSalt);
```

**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`proxy`|`address`|The address of the new BeaconProxy|
|`data`|`bytes`|The data used to initialize the BeaconProxy|
|`baseSalt`|`bytes32`|The base salt used for deterministic deployment|

## Errors
### InvalidAddress
Error thrown when an invalid address is provided


```solidity
error InvalidAddress();
```

