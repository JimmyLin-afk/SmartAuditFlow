# LenderCommitmentForwarder
[Git Source](https://github.com/teller-protocol/teller-protocol-v2/blob/f4bf5a00ae7113b0344876c13db9b3dd705154f6/contracts/LenderCommitmentForwarder.sol)

**Inherits:**
[TellerV2MarketForwarder](/contracts/TellerV2MarketForwarder.sol/abstract.TellerV2MarketForwarder.md)


## State Variables
### commitments

```solidity
mapping(uint256 => Commitment) public commitments;
```


### commitmentCount

```solidity
uint256 commitmentCount;
```


### commitmentBorrowersList

```solidity
mapping(uint256 => EnumerableSetUpgradeable.AddressSet) internal commitmentBorrowersList;
```


## Functions
### commitmentLender

Modifiers *


```solidity
modifier commitmentLender(uint256 _commitmentId);
```

### validateCommitment


```solidity
function validateCommitment(Commitment storage _commitment) internal;
```

### constructor

External Functions *


```solidity
constructor(address _protocolAddress, address _marketRegistry)
    TellerV2MarketForwarder(_protocolAddress, _marketRegistry);
```

### createCommitment

Creates a loan commitment from a lender for a market.


```solidity
function createCommitment(Commitment calldata _commitment, address[] calldata _borrowerAddressList)
    public
    returns (uint256 commitmentId_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_commitment`|`Commitment`|The new commitment data expressed as a struct|
|`_borrowerAddressList`|`address[]`|The array of borrowers that are allowed to accept loans using this commitment|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`commitmentId_`|`uint256`|returns the commitmentId for the created commitment|


### updateCommitment

Updates the commitment of a lender to a market.


```solidity
function updateCommitment(uint256 _commitmentId, Commitment calldata _commitment)
    public
    commitmentLender(_commitmentId);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_commitmentId`|`uint256`|The Id of the commitment to update.|
|`_commitment`|`Commitment`|The new commitment data expressed as a struct|


### updateCommitmentBorrowers

Updates the borrowers allowed to accept a commitment


```solidity
function updateCommitmentBorrowers(uint256 _commitmentId, address[] calldata _borrowerAddressList)
    public
    commitmentLender(_commitmentId);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_commitmentId`|`uint256`|The Id of the commitment to update.|
|`_borrowerAddressList`|`address[]`|The array of borrowers that are allowed to accept loans using this commitment|


### _addBorrowersToCommitmentAllowlist

Adds a borrower to the allowlist for a commmitment.


```solidity
function _addBorrowersToCommitmentAllowlist(uint256 _commitmentId, address[] calldata _borrowerArray) internal;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_commitmentId`|`uint256`|The id of the commitment that will allow the new borrower|
|`_borrowerArray`|`address[]`|the address array of the borrowers that will be allowed to accept loans using the commitment|


### deleteCommitment

Removes the commitment of a lender to a market.


```solidity
function deleteCommitment(uint256 _commitmentId) public commitmentLender(_commitmentId);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_commitmentId`|`uint256`|The id of the commitment to delete.|


### _decrementCommitment

Reduces the commitment amount for a lender to a market.


```solidity
function _decrementCommitment(uint256 _commitmentId, uint256 _tokenAmountDelta) internal;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_commitmentId`|`uint256`|The id of the commitment to modify.|
|`_tokenAmountDelta`|`uint256`|The amount of change in the maxPrincipal.|


### acceptCommitment

Accept the commitment to submitBid and acceptBid using the funds

*LoanDuration must be longer than the market payment cycle*


```solidity
function acceptCommitment(
    uint256 _commitmentId,
    uint256 _principalAmount,
    uint256 _collateralAmount,
    uint256 _collateralTokenId,
    address _collateralTokenAddress,
    uint16 _interestRate,
    uint32 _loanDuration
) external returns (uint256 bidId);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_commitmentId`|`uint256`|The id of the commitment being accepted.|
|`_principalAmount`|`uint256`|The amount of currency to borrow for the loan.|
|`_collateralAmount`|`uint256`|The amount of collateral to use for the loan.|
|`_collateralTokenId`|`uint256`|The tokenId of collateral to use for the loan if ERC721 or ERC1155.|
|`_collateralTokenAddress`|`address`|The contract address to use for the loan collateral tokens.|
|`_interestRate`|`uint16`|The interest rate APY to use for the loan in basis points.|
|`_loanDuration`|`uint32`|The overall duration for the loan.  Must be longer than market payment cycle duration.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`bidId`|`uint256`|The ID of the loan that was created on TellerV2|


### getRequiredCollateral

Calculate the amount of collateral required to borrow a loan with _principalAmount of principal


```solidity
function getRequiredCollateral(
    uint256 _principalAmount,
    uint256 _maxPrincipalPerCollateralAmount,
    CommitmentCollateralType _collateralTokenType,
    address _collateralTokenAddress,
    address _principalTokenAddress
) public view virtual returns (uint256);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_principalAmount`|`uint256`|The amount of currency to borrow for the loan.|
|`_maxPrincipalPerCollateralAmount`|`uint256`|The ratio for the amount of principal that can be borrowed for each amount of collateral. This is expanded additionally by the principal decimals.|
|`_collateralTokenType`|`CommitmentCollateralType`|The type of collateral for the loan either ERC20, ERC721, ERC1155, or None.|
|`_collateralTokenAddress`|`address`|The contract address for the collateral for the loan.|
|`_principalTokenAddress`|`address`|The contract address for the principal for the loan.|


### getCommitmentBorrowers

Return the array of borrowers that are allowlisted for a commitment


```solidity
function getCommitmentBorrowers(uint256 _commitmentId) external view returns (address[] memory borrowers_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_commitmentId`|`uint256`|The commitment id for the commitment to query.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`borrowers_`|`address[]`|An array of addresses restricted to accept the commitment. Empty array means unrestricted.|


### _submitBidFromCommitment

Internal function to submit a bid to the lending protocol using a commitment


```solidity
function _submitBidFromCommitment(
    address _borrower,
    uint256 _marketId,
    address _principalTokenAddress,
    uint256 _principalAmount,
    address _collateralTokenAddress,
    uint256 _collateralAmount,
    uint256 _collateralTokenId,
    CommitmentCollateralType _collateralTokenType,
    uint32 _loanDuration,
    uint16 _interestRate
) internal returns (uint256 bidId);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_borrower`|`address`|The address of the borrower for the loan.|
|`_marketId`|`uint256`|The id for the market of the loan in the lending protocol.|
|`_principalTokenAddress`|`address`|The contract address for the principal token.|
|`_principalAmount`|`uint256`|The amount of principal to borrow for the loan.|
|`_collateralTokenAddress`|`address`|The contract address for the collateral token.|
|`_collateralAmount`|`uint256`|The amount of collateral to use for the loan.|
|`_collateralTokenId`|`uint256`|The tokenId for the collateral (if it is ERC721 or ERC1155).|
|`_collateralTokenType`|`CommitmentCollateralType`|The type of collateral token (ERC20,ERC721,ERC1177,None).|
|`_loanDuration`|`uint32`|The duration of the loan in seconds delta.  Must be longer than loan payment cycle for the market.|
|`_interestRate`|`uint16`|The amount of interest APY for the loan expressed in basis points.|


### _getEscrowCollateralType

Return the collateral type based on the commitmentcollateral type.  Collateral type is used in the base lending protocol.


```solidity
function _getEscrowCollateralType(CommitmentCollateralType _type) internal pure returns (CollateralType);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_type`|`CommitmentCollateralType`|The type of collateral to be used for the loan.|


## Events
### CreatedCommitment
This event is emitted when a lender's commitment is created.


```solidity
event CreatedCommitment(
    uint256 indexed commitmentId, address lender, uint256 marketId, address lendingToken, uint256 tokenAmount
);
```

### UpdatedCommitment
This event is emitted when a lender's commitment is updated.


```solidity
event UpdatedCommitment(
    uint256 indexed commitmentId, address lender, uint256 marketId, address lendingToken, uint256 tokenAmount
);
```

### UpdatedCommitmentBorrowers
This event is emitted when the allowed borrowers for a commitment is updated.


```solidity
event UpdatedCommitmentBorrowers(uint256 indexed commitmentId);
```

### DeletedCommitment
This event is emitted when a lender's commitment has been deleted.


```solidity
event DeletedCommitment(uint256 indexed commitmentId);
```

### ExercisedCommitment
This event is emitted when a lender's commitment is exercised for a loan.


```solidity
event ExercisedCommitment(uint256 indexed commitmentId, address borrower, uint256 tokenAmount, uint256 bidId);
```

## Errors
### InsufficientCommitmentAllocation

```solidity
error InsufficientCommitmentAllocation(uint256 allocated, uint256 requested);
```

### InsufficientBorrowerCollateral

```solidity
error InsufficientBorrowerCollateral(uint256 required, uint256 actual);
```

## Structs
### Commitment
Details about a lender's capital commitment.


```solidity
struct Commitment {
    uint256 maxPrincipal;
    uint32 expiration;
    uint32 maxDuration;
    uint16 minInterestRate;
    address collateralTokenAddress;
    uint256 collateralTokenId;
    uint256 maxPrincipalPerCollateralAmount;
    CommitmentCollateralType collateralTokenType;
    address lender;
    uint256 marketId;
    address principalTokenAddress;
}
```

## Enums
### CommitmentCollateralType

```solidity
enum CommitmentCollateralType {
    NONE,
    ERC20,
    ERC721,
    ERC1155,
    ERC721_ANY_ID,
    ERC1155_ANY_ID
}
```

