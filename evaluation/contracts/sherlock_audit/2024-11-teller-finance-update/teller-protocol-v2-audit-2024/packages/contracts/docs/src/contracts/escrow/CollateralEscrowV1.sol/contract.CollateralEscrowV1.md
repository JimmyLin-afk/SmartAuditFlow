# CollateralEscrowV1
[Git Source](https://github.com/teller-protocol/teller-protocol-v2/blob/f4bf5a00ae7113b0344876c13db9b3dd705154f6/contracts/escrow/CollateralEscrowV1.sol)

**Inherits:**
OwnableUpgradeable, ICollateralEscrowV1


## State Variables
### bidId

```solidity
uint256 public bidId;
```


### collateralBalances

```solidity
mapping(address => Collateral) public collateralBalances;
```


## Functions
### initialize

Initializes an escrow.

The id of the associated bid.


```solidity
function initialize(uint256 _bidId) public initializer;
```

### getBid

Returns the id of the associated bid.


```solidity
function getBid() external view returns (uint256);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint256`|The id of the associated bid.|


### depositAsset

Deposits a collateral asset into the escrow.


```solidity
function depositAsset(CollateralType _collateralType, address _collateralAddress, uint256 _amount, uint256 _tokenId)
    external
    payable
    virtual
    onlyOwner;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_collateralType`|`CollateralType`|The type of collateral asset to deposit (ERC721, ERC1155).|
|`_collateralAddress`|`address`|The address of the collateral token.|
|`_amount`|`uint256`|The amount to deposit.|
|`_tokenId`|`uint256`||


### withdraw

Withdraws a collateral asset from the escrow.


```solidity
function withdraw(address _collateralAddress, uint256 _amount, address _recipient) external virtual onlyOwner;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_collateralAddress`|`address`|The address of the collateral contract.|
|`_amount`|`uint256`|The amount to withdraw.|
|`_recipient`|`address`|The address to send the assets to.|


### _depositCollateral

Internal function for transferring collateral assets into this contract.


```solidity
function _depositCollateral(
    CollateralType _collateralType,
    address _collateralAddress,
    uint256 _amount,
    uint256 _tokenId
) internal;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_collateralType`|`CollateralType`||
|`_collateralAddress`|`address`|The address of the collateral contract.|
|`_amount`|`uint256`|The amount to deposit.|
|`_tokenId`|`uint256`|The token id of the collateral asset.|


### _withdrawCollateral

Internal function for transferring collateral assets out of this contract.


```solidity
function _withdrawCollateral(
    Collateral memory _collateral,
    address _collateralAddress,
    uint256 _amount,
    address _recipient
) internal;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_collateral`|`Collateral`|The collateral asset to withdraw.|
|`_collateralAddress`|`address`|The address of the collateral contract.|
|`_amount`|`uint256`|The amount to withdraw.|
|`_recipient`|`address`|The address to send the assets to.|


### onERC721Received


```solidity
function onERC721Received(address, address, uint256, bytes calldata) external pure returns (bytes4);
```

### onERC1155Received


```solidity
function onERC1155Received(address, address, uint256 id, uint256 value, bytes calldata) external returns (bytes4);
```

### onERC1155BatchReceived


```solidity
function onERC1155BatchReceived(address, address, uint256[] calldata _ids, uint256[] calldata _values, bytes calldata)
    external
    returns (bytes4);
```

## Events
### CollateralDeposited

```solidity
event CollateralDeposited(address _collateralAddress, uint256 _amount);
```

### CollateralWithdrawn

```solidity
event CollateralWithdrawn(address _collateralAddress, uint256 _amount, address _recipient);
```

