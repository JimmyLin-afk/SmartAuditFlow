# BeaconProxyFactory
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/1dbcbcfe9a8bcf9392b2ada63dd8f1827a90783b/src/BeaconProxyFactory.sol)

**Inherits:**
[IBeaconProxyFactory](/src/interfaces/IBeaconProxyFactory.sol/interface.IBeaconProxyFactory.md), UpgradeableBeacon

*Implementation of a factory that allows for deterministic deployment of BeaconProxys from an UpgradeableBeacon
using the Create2 opcode. The salt used for the Create2 deployment is the hash of the sender and the base salt.*


## State Variables
### numProxies
Returns the number of BeaconProxys deployed by the factory


```solidity
uint256 public numProxies;
```


## Functions
### constructor

Creates a new BeaconProxyFactory


```solidity
constructor(address _implementation, address _owner) UpgradeableBeacon(_implementation, _owner);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_implementation`|`address`|The implementation contract for the beacon that will be used by BeaconProxys created by this factory|
|`_owner`|`address`|The owner of this factory, allowed to update the beacon implementation|


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


### _getDeploySalt

*Returns the deploy salt for the BeaconProxy, which is the hash of the sender and the base salt*


```solidity
function _getDeploySalt(address sender, bytes32 baseSalt) internal pure returns (bytes32 salt);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`sender`|`address`|The address that will deploy the beacon proxy using the factory|
|`baseSalt`|`bytes32`|The base salt used for deterministic deployment|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`salt`|`bytes32`|The deploy salt for the BeaconProxy|


### _getCreationCode

*Returns the creation code for the BeaconProxy*


```solidity
function _getCreationCode(bytes memory data) internal view returns (bytes memory bytecode);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`data`|`bytes`|The initialization data for the BeaconProxy|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`bytecode`|`bytes`|The creation code for the BeaconProxy|


