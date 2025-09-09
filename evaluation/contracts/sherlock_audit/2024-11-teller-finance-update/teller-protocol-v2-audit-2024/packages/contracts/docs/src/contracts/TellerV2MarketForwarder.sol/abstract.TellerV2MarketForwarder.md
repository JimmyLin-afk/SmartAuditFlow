# TellerV2MarketForwarder
[Git Source](https://github.com/teller-protocol/teller-protocol-v2/blob/f4bf5a00ae7113b0344876c13db9b3dd705154f6/contracts/TellerV2MarketForwarder.sol)

**Inherits:**
Initializable, ContextUpgradeable

*Simple helper contract to forward an encoded function call to the TellerV2 contract. See {TellerV2Context}*


## State Variables
### _tellerV2

```solidity
address public immutable _tellerV2;
```


### _marketRegistry

```solidity
address public immutable _marketRegistry;
```


### __gap
*This empty reserved space is put in place to allow future versions to add new
variables without shifting down storage in the inheritance chain.
See https://docs.openzeppelin.com/contracts/4.x/upgradeable#storage_gaps*


```solidity
uint256[50] private __gap;
```


## Functions
### constructor


```solidity
constructor(address _protocolAddress, address _marketRegistryAddress);
```

### getTellerV2


```solidity
function getTellerV2() public view returns (address);
```

### getMarketRegistry


```solidity
function getMarketRegistry() public view returns (address);
```

### getTellerV2MarketOwner


```solidity
function getTellerV2MarketOwner(uint256 marketId) public returns (address);
```

### _forwardCall

*Performs function call to the TellerV2 contract by appending an address to the calldata.*


```solidity
function _forwardCall(bytes memory _data, address _msgSender) internal returns (bytes memory);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_data`|`bytes`|The encoded function calldata on TellerV2.|
|`_msgSender`|`address`|The address that should be treated as the underlying function caller.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`bytes`|The encoded response from the called function. Requirements: - The {_msgSender} address must set an approval on TellerV2 for this forwarder contract __before__ making this call.|


### _submitBid

Creates a new loan using the TellerV2 lending protocol.


```solidity
function _submitBid(CreateLoanArgs memory _createLoanArgs, address _borrower)
    internal
    virtual
    returns (uint256 bidId);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_createLoanArgs`|`CreateLoanArgs`|Details describing the loan agreement.]|
|`_borrower`|`address`|The borrower address for the new loan.|


### _submitBidWithCollateral

Creates a new loan using the TellerV2 lending protocol.


```solidity
function _submitBidWithCollateral(
    CreateLoanArgs memory _createLoanArgs,
    Collateral[] memory _collateralInfo,
    address _borrower
) internal virtual returns (uint256 bidId);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_createLoanArgs`|`CreateLoanArgs`|Details describing the loan agreement.]|
|`_collateralInfo`|`Collateral[]`||
|`_borrower`|`address`|The borrower address for the new loan.|


### _acceptBid

Accepts a new loan using the TellerV2 lending protocol.


```solidity
function _acceptBid(uint256 _bidId, address _lender) internal virtual returns (bool);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the new loan.|
|`_lender`|`address`|The address of the lender who will provide funds for the new loan.|


## Structs
### CreateLoanArgs

```solidity
struct CreateLoanArgs {
    uint256 marketId;
    address lendingToken;
    uint256 principal;
    uint32 duration;
    uint16 interestRate;
    string metadataURI;
    address recipient;
}
```

