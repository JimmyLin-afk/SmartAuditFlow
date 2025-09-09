# MorphoLendingAdapterFactory
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/1dbcbcfe9a8bcf9392b2ada63dd8f1827a90783b/src/lending/MorphoLendingAdapterFactory.sol)

**Inherits:**
[IMorphoLendingAdapterFactory](/src/interfaces/IMorphoLendingAdapterFactory.sol/interface.IMorphoLendingAdapterFactory.md)

*The MorphoLendingAdapterFactory is a factory contract for deploying ERC-1167 minimal proxies of the
MorphoLendingAdapter contract using OpenZeppelin's Clones library.*


## State Variables
### lendingAdapterLogic
Returns the address of the MorphoLendingAdapter logic contract used to deploy minimal proxies.


```solidity
IMorphoLendingAdapter public immutable lendingAdapterLogic;
```


## Functions
### constructor


```solidity
constructor(IMorphoLendingAdapter _lendingAdapterLogic);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_lendingAdapterLogic`|`IMorphoLendingAdapter`|Logic contract for deploying new MorphoLendingAdapters.|


### computeAddress

Given the `sender` and `baseSalt` compute and return the address that MorphoLendingAdapter will be deployed to
using the `IMorphoLendingAdapterFactory.deployAdapter` function.

*MorphoLendingAdapter addresses are uniquely determined by their salt because the deployer is always the factory,
and the use of minimal proxies means they all have identical bytecode and therefore an identical bytecode hash.*


```solidity
function computeAddress(address sender, bytes32 baseSalt) external view returns (address);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`sender`|`address`|The address of the sender of the `IMorphoLendingAdapterFactory.deployAdapter` call.|
|`baseSalt`|`bytes32`|The user-provided salt.|


### deployAdapter

Deploys a new MorphoLendingAdapter contract with the specified configuration.

*MorphoLendingAdapters deployed by this factory are minimal proxies.*


```solidity
function deployAdapter(Id morphoMarketId, address authorizedCreator, bytes32 baseSalt)
    public
    returns (IMorphoLendingAdapter);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`morphoMarketId`|`Id`|The Morpho market ID|
|`authorizedCreator`|`address`|The authorized creator of the deployed MorphoLendingAdapter. The authorized creator can create a new LeverageToken using this adapter on the LeverageManager|
|`baseSalt`|`bytes32`|Used to compute the resulting address of the MorphoLendingAdapter.|


### salt

Given the `sender` and `baseSalt`, return the salt that will be used for deployment.


```solidity
function salt(address sender, bytes32 baseSalt) internal pure returns (bytes32);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`sender`|`address`|The address of the sender of the `deployAdapter` call.|
|`baseSalt`|`bytes32`|The user-provided base salt.|


