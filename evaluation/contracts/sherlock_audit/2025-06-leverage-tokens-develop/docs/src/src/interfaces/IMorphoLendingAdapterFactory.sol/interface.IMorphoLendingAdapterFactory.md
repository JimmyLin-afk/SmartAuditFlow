# IMorphoLendingAdapterFactory
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/1dbcbcfe9a8bcf9392b2ada63dd8f1827a90783b/src/interfaces/IMorphoLendingAdapterFactory.sol)


## Functions
### computeAddress

Given the `sender` and `baseSalt` compute and return the address that MorphoLendingAdapter will be deployed to
using the `IMorphoLendingAdapterFactory.deployAdapter` function.

*MorphoLendingAdapter addresses are uniquely determined by their salt because the deployer is always the factory,
and the use of minimal proxies means they all have identical bytecode and therefore an identical bytecode hash.*

*The `baseSalt` is the user-provided salt, not the final salt after hashing with the sender's address.*


```solidity
function computeAddress(address sender, bytes32 baseSalt) external view returns (address);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`sender`|`address`|The address of the sender of the `IMorphoLendingAdapterFactory.deployAdapter` call.|
|`baseSalt`|`bytes32`|The user-provided salt.|


### lendingAdapterLogic

Returns the address of the MorphoLendingAdapter logic contract used to deploy minimal proxies.


```solidity
function lendingAdapterLogic() external view returns (IMorphoLendingAdapter);
```

### deployAdapter

Deploys a new MorphoLendingAdapter contract with the specified configuration.

*MorphoLendingAdapters deployed by this factory are minimal proxies.*


```solidity
function deployAdapter(Id morphoMarketId, address authorizedCreator, bytes32 baseSalt)
    external
    returns (IMorphoLendingAdapter lendingAdapter);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`morphoMarketId`|`Id`|The Morpho market ID|
|`authorizedCreator`|`address`|The authorized creator of the deployed MorphoLendingAdapter. The authorized creator can create a new LeverageToken using this adapter on the LeverageManager|
|`baseSalt`|`bytes32`|Used to compute the resulting address of the MorphoLendingAdapter.|


## Events
### MorphoLendingAdapterDeployed
Emitted when a new MorphoLendingAdapter is deployed.


```solidity
event MorphoLendingAdapterDeployed(IMorphoLendingAdapter lendingAdapter);
```

**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`lendingAdapter`|`IMorphoLendingAdapter`|The deployed MorphoLendingAdapter|

