# CollateralManager
[Git Source](https://github.com/teller-protocol/teller-protocol-v2/blob/f4bf5a00ae7113b0344876c13db9b3dd705154f6/contracts/CollateralManager.sol)

**Inherits:**
OwnableUpgradeable, ICollateralManager


## State Variables
### tellerV2

```solidity
ITellerV2 public tellerV2;
```


### collateralEscrowBeacon

```solidity
address private collateralEscrowBeacon;
```


### _escrows

```solidity
mapping(uint256 => address) public _escrows;
```


### _bidCollaterals

```solidity
mapping(uint256 => CollateralInfo) internal _bidCollaterals;
```


## Functions
### onlyTellerV2


```solidity
modifier onlyTellerV2();
```

### initialize

Initializes the collateral manager.


```solidity
function initialize(address _collateralEscrowBeacon, address _tellerV2) external initializer;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_collateralEscrowBeacon`|`address`|The address of the escrow implementation.|
|`_tellerV2`|`address`|The address of the protocol.|


### setCollateralEscrowBeacon

Sets the address of the Beacon contract used for the collateral escrow contracts.


```solidity
function setCollateralEscrowBeacon(address _collateralEscrowBeacon) external reinitializer(2);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_collateralEscrowBeacon`|`address`|The address of the Beacon contract.|


### isBidCollateralBacked

Checks to see if a bid is backed by collateral.


```solidity
function isBidCollateralBacked(uint256 _bidId) public virtual returns (bool);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the bid to check.|


### commitCollateral

Checks the validity of a borrower's multiple collateral balances and commits it to a bid.


```solidity
function commitCollateral(uint256 _bidId, Collateral[] calldata _collateralInfo) public returns (bool validation_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the associated bid.|
|`_collateralInfo`|`Collateral[]`|Additional information about the collateral assets.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`validation_`|`bool`|Boolean indicating if the collateral balances were validated.|


### commitCollateral

Checks the validity of a borrower's collateral balance and commits it to a bid.


```solidity
function commitCollateral(uint256 _bidId, Collateral calldata _collateralInfo) public returns (bool validation_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the associated bid.|
|`_collateralInfo`|`Collateral`|Additional information about the collateral asset.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`validation_`|`bool`|Boolean indicating if the collateral balance was validated.|


### revalidateCollateral

Re-checks the validity of a borrower's collateral balance committed to a bid.


```solidity
function revalidateCollateral(uint256 _bidId) external returns (bool validation_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the associated bid.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`validation_`|`bool`|Boolean indicating if the collateral balance was validated.|


### checkBalances

Checks the validity of a borrower's multiple collateral balances.


```solidity
function checkBalances(address _borrowerAddress, Collateral[] calldata _collateralInfo)
    public
    returns (bool validated_, bool[] memory checks_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_borrowerAddress`|`address`|The address of the borrower holding the collateral.|
|`_collateralInfo`|`Collateral[]`|Additional information about the collateral assets.|


### deployAndDeposit

Deploys a new collateral escrow and deposits collateral.


```solidity
function deployAndDeposit(uint256 _bidId) external onlyTellerV2;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The associated bidId of the collateral escrow.|


### getEscrow

Gets the address of a deployed escrow.

_bidId The bidId to return the escrow for.


```solidity
function getEscrow(uint256 _bidId) external view returns (address);
```
**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`address`|The address of the escrow.|


### getCollateralInfo

Gets the collateral info for a given bid id.


```solidity
function getCollateralInfo(uint256 _bidId) public view returns (Collateral[] memory infos_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The bidId to return the collateral info for.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`infos_`|`Collateral[]`|The stored collateral info.|


### getCollateralAmount

Gets the collateral asset amount for a given bid id on the TellerV2 contract.


```solidity
function getCollateralAmount(uint256 _bidId, address _collateralAddress) public view returns (uint256 amount_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The ID of a bid on TellerV2.|
|`_collateralAddress`|`address`|An address used as collateral.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`amount_`|`uint256`|The amount of collateral of type _collateralAddress.|


### withdraw

Withdraws deposited collateral from the created escrow of a bid that has been successfully repaid.


```solidity
function withdraw(uint256 _bidId) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the bid to withdraw collateral for.|


### liquidateCollateral

Sends the deposited collateral to a liquidator of a bid.

Can only be called by the protocol.


```solidity
function liquidateCollateral(uint256 _bidId, address _liquidatorAddress) external onlyTellerV2;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the liquidated bid.|
|`_liquidatorAddress`|`address`|The address of the liquidator to send the collateral to.|


### _deployEscrow

Deploys a new collateral escrow.


```solidity
function _deployEscrow(uint256 _bidId) internal virtual returns (address proxyAddress_, address borrower_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The associated bidId of the collateral escrow.|


### _deposit


```solidity
function _deposit(uint256 _bidId, Collateral memory collateralInfo) internal virtual;
```

### _withdraw

Withdraws collateral to a given receiver's address.


```solidity
function _withdraw(uint256 _bidId, address _receiver) internal virtual;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the bid to withdraw collateral for.|
|`_receiver`|`address`|The address to withdraw the collateral to.|


### _commitCollateral

Checks the validity of a borrower's collateral balance and commits it to a bid.


```solidity
function _commitCollateral(uint256 _bidId, Collateral memory _collateralInfo) internal virtual;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the associated bid.|
|`_collateralInfo`|`Collateral`|Additional information about the collateral asset.|


### _checkBalances

Checks the validity of a borrower's multiple collateral balances.


```solidity
function _checkBalances(address _borrowerAddress, Collateral[] memory _collateralInfo, bool _shortCircut)
    internal
    virtual
    returns (bool validated_, bool[] memory checks_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_borrowerAddress`|`address`|The address of the borrower holding the collateral.|
|`_collateralInfo`|`Collateral[]`|Additional information about the collateral assets.|
|`_shortCircut`|`bool`| if true, will return immediately until an invalid balance|


### _checkBalance

Checks the validity of a borrower's single collateral balance.


```solidity
function _checkBalance(address _borrowerAddress, Collateral memory _collateralInfo) internal virtual returns (bool);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_borrowerAddress`|`address`|The address of the borrower holding the collateral.|
|`_collateralInfo`|`Collateral`|Additional information about the collateral asset.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`bool`|validation_ Boolean indicating if the collateral balances were validated.|


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
### CollateralEscrowDeployed

```solidity
event CollateralEscrowDeployed(uint256 _bidId, address _collateralEscrow);
```

### CollateralCommitted

```solidity
event CollateralCommitted(
    uint256 _bidId, CollateralType _type, address _collateralAddress, uint256 _amount, uint256 _tokenId
);
```

### CollateralClaimed

```solidity
event CollateralClaimed(uint256 _bidId);
```

### CollateralDeposited

```solidity
event CollateralDeposited(
    uint256 _bidId, CollateralType _type, address _collateralAddress, uint256 _amount, uint256 _tokenId
);
```

### CollateralWithdrawn

```solidity
event CollateralWithdrawn(
    uint256 _bidId,
    CollateralType _type,
    address _collateralAddress,
    uint256 _amount,
    uint256 _tokenId,
    address _recipient
);
```

## Structs
### CollateralInfo
Since collateralInfo is mapped (address assetAddress => Collateral) that means
that only a single tokenId per nft per loan can be collateralized.
Ex. Two bored apes cannot be used as collateral for a single loan.


```solidity
struct CollateralInfo {
    EnumerableSetUpgradeable.AddressSet collateralAddresses;
    mapping(address => Collateral) collateralInfo;
}
```

