# MarketRegistry
[Git Source](https://github.com/teller-protocol/teller-protocol-v2/blob/f4bf5a00ae7113b0344876c13db9b3dd705154f6/contracts/MarketRegistry.sol)

**Inherits:**
IMarketRegistry, Initializable, Context, TellerASResolver


## State Variables
### CURRENT_CODE_VERSION
Constant Variables *


```solidity
uint256 public constant CURRENT_CODE_VERSION = 8;
```


### lenderAttestationSchemaId

```solidity
bytes32 public lenderAttestationSchemaId;
```


### markets

```solidity
mapping(uint256 => Marketplace) internal markets;
```


### __uriToId

```solidity
mapping(bytes32 => uint256) internal __uriToId;
```


### marketCount

```solidity
uint256 public marketCount;
```


### _attestingSchemaId

```solidity
bytes32 private _attestingSchemaId;
```


### borrowerAttestationSchemaId

```solidity
bytes32 public borrowerAttestationSchemaId;
```


### version

```solidity
uint256 public version;
```


### marketIsClosed

```solidity
mapping(uint256 => bool) private marketIsClosed;
```


### tellerAS

```solidity
TellerAS public tellerAS;
```


## Functions
### ownsMarket


```solidity
modifier ownsMarket(uint256 _marketId);
```

### withAttestingSchema


```solidity
modifier withAttestingSchema(bytes32 schemaId);
```

### initialize


```solidity
function initialize(TellerAS _tellerAS) external initializer;
```

### createMarket

Creates a new market.


```solidity
function createMarket(
    address _initialOwner,
    uint32 _paymentCycleDuration,
    uint32 _paymentDefaultDuration,
    uint32 _bidExpirationTime,
    uint16 _feePercent,
    bool _requireLenderAttestation,
    bool _requireBorrowerAttestation,
    PaymentType _paymentType,
    PaymentCycleType _paymentCycleType,
    string calldata _uri
) external returns (uint256 marketId_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_initialOwner`|`address`|Address who will initially own the market.|
|`_paymentCycleDuration`|`uint32`|Length of time in seconds before a bid's next payment is required to be made.|
|`_paymentDefaultDuration`|`uint32`|Length of time in seconds before a loan is considered in default for non-payment.|
|`_bidExpirationTime`|`uint32`|Length of time in seconds before pending bids expire.|
|`_feePercent`|`uint16`||
|`_requireLenderAttestation`|`bool`|Boolean that indicates if lenders require attestation to join market.|
|`_requireBorrowerAttestation`|`bool`|Boolean that indicates if borrowers require attestation to join market.|
|`_paymentType`|`PaymentType`|The payment type for loans in the market.|
|`_paymentCycleType`|`PaymentCycleType`|The payment cycle type for loans in the market - Seconds or Monthly|
|`_uri`|`string`|URI string to get metadata details about the market.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`marketId_`|`uint256`|The market ID of the newly created market.|


### createMarket

Creates a new market.

*Uses the default EMI payment type.*


```solidity
function createMarket(
    address _initialOwner,
    uint32 _paymentCycleDuration,
    uint32 _paymentDefaultDuration,
    uint32 _bidExpirationTime,
    uint16 _feePercent,
    bool _requireLenderAttestation,
    bool _requireBorrowerAttestation,
    string calldata _uri
) external returns (uint256 marketId_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_initialOwner`|`address`|Address who will initially own the market.|
|`_paymentCycleDuration`|`uint32`|Length of time in seconds before a bid's next payment is required to be made.|
|`_paymentDefaultDuration`|`uint32`|Length of time in seconds before a loan is considered in default for non-payment.|
|`_bidExpirationTime`|`uint32`|Length of time in seconds before pending bids expire.|
|`_feePercent`|`uint16`||
|`_requireLenderAttestation`|`bool`|Boolean that indicates if lenders require attestation to join market.|
|`_requireBorrowerAttestation`|`bool`|Boolean that indicates if borrowers require attestation to join market.|
|`_uri`|`string`|URI string to get metadata details about the market.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`marketId_`|`uint256`|The market ID of the newly created market.|


### _createMarket

Creates a new market.


```solidity
function _createMarket(
    address _initialOwner,
    uint32 _paymentCycleDuration,
    uint32 _paymentDefaultDuration,
    uint32 _bidExpirationTime,
    uint16 _feePercent,
    bool _requireLenderAttestation,
    bool _requireBorrowerAttestation,
    PaymentType _paymentType,
    PaymentCycleType _paymentCycleType,
    string calldata _uri
) internal returns (uint256 marketId_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_initialOwner`|`address`|Address who will initially own the market.|
|`_paymentCycleDuration`|`uint32`|Length of time in seconds before a bid's next payment is required to be made.|
|`_paymentDefaultDuration`|`uint32`|Length of time in seconds before a loan is considered in default for non-payment.|
|`_bidExpirationTime`|`uint32`|Length of time in seconds before pending bids expire.|
|`_feePercent`|`uint16`||
|`_requireLenderAttestation`|`bool`|Boolean that indicates if lenders require attestation to join market.|
|`_requireBorrowerAttestation`|`bool`|Boolean that indicates if borrowers require attestation to join market.|
|`_paymentType`|`PaymentType`|The payment type for loans in the market.|
|`_paymentCycleType`|`PaymentCycleType`|The payment cycle type for loans in the market - Seconds or Monthly|
|`_uri`|`string`|URI string to get metadata details about the market.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`marketId_`|`uint256`|The market ID of the newly created market.|


### closeMarket

Closes a market so new bids cannot be added.


```solidity
function closeMarket(uint256 _marketId) public ownsMarket(_marketId);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The market ID for the market to close.|


### isMarketClosed

Returns the status of a market being open or closed for new bids.


```solidity
function isMarketClosed(uint256 _marketId) public view override returns (bool);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The market ID for the market to check.|


### attestLender

Adds a lender to a market.

*See {_attestStakeholder}.*


```solidity
function attestLender(uint256 _marketId, address _lenderAddress, uint256 _expirationTime) external;
```

### attestLender

Adds a lender to a market via delegated attestation.

*See {_attestStakeholderViaDelegation}.*


```solidity
function attestLender(
    uint256 _marketId,
    address _lenderAddress,
    uint256 _expirationTime,
    uint8 _v,
    bytes32 _r,
    bytes32 _s
) external;
```

### revokeLender

Removes a lender from an market.

*See {_revokeStakeholder}.*


```solidity
function revokeLender(uint256 _marketId, address _lenderAddress) external;
```

### revokeLender

Removes a borrower from a market via delegated revocation.

*See {_revokeStakeholderViaDelegation}.*


```solidity
function revokeLender(uint256 _marketId, address _lenderAddress, uint8 _v, bytes32 _r, bytes32 _s) external;
```

### lenderExitMarket

Allows a lender to voluntarily leave a market.


```solidity
function lenderExitMarket(uint256 _marketId) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The market ID to leave.|


### attestBorrower

Adds a borrower to a market.

*See {_attestStakeholder}.*


```solidity
function attestBorrower(uint256 _marketId, address _borrowerAddress, uint256 _expirationTime) external;
```

### attestBorrower

Adds a borrower to a market via delegated attestation.

*See {_attestStakeholderViaDelegation}.*


```solidity
function attestBorrower(
    uint256 _marketId,
    address _borrowerAddress,
    uint256 _expirationTime,
    uint8 _v,
    bytes32 _r,
    bytes32 _s
) external;
```

### revokeBorrower

Removes a borrower from an market.

*See {_revokeStakeholder}.*


```solidity
function revokeBorrower(uint256 _marketId, address _borrowerAddress) external;
```

### revokeBorrower

Removes a borrower from a market via delegated revocation.

*See {_revokeStakeholderViaDelegation}.*


```solidity
function revokeBorrower(uint256 _marketId, address _borrowerAddress, uint8 _v, bytes32 _r, bytes32 _s) external;
```

### borrowerExitMarket

Allows a borrower to voluntarily leave a market.


```solidity
function borrowerExitMarket(uint256 _marketId) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The market ID to leave.|


### resolve

Verifies an attestation is valid.

*This function must only be called by the `attestLender` function above.*


```solidity
function resolve(address recipient, bytes calldata schema, bytes calldata data, uint256, address attestor)
    external
    payable
    override
    returns (bool);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`recipient`|`address`|Lender's address who is being attested.|
|`schema`|`bytes`|The schema used for the attestation.|
|`data`|`bytes`|Data the must include the market ID and lender's address|
|`<none>`|`uint256`||
|`attestor`|`address`|Market owner's address who signed the attestation.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`bool`|Boolean indicating the attestation was successful.|


### transferMarketOwnership

Transfers ownership of a marketplace.


```solidity
function transferMarketOwnership(uint256 _marketId, address _newOwner) public ownsMarket(_marketId);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The ID of a market.|
|`_newOwner`|`address`|Address of the new market owner. Requirements: - The caller must be the current owner.|


### updateMarketSettings

Updates multiple market settings for a given market.


```solidity
function updateMarketSettings(
    uint256 _marketId,
    uint32 _paymentCycleDuration,
    PaymentType _newPaymentType,
    PaymentCycleType _paymentCycleType,
    uint32 _paymentDefaultDuration,
    uint32 _bidExpirationTime,
    uint16 _feePercent,
    bool _borrowerAttestationRequired,
    bool _lenderAttestationRequired,
    string calldata _metadataURI
) public ownsMarket(_marketId);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The ID of a market.|
|`_paymentCycleDuration`|`uint32`|Delinquency duration for new loans|
|`_newPaymentType`|`PaymentType`|The payment type for the market.|
|`_paymentCycleType`|`PaymentCycleType`|The payment cycle type for loans in the market - Seconds or Monthly|
|`_paymentDefaultDuration`|`uint32`|Default duration for new loans|
|`_bidExpirationTime`|`uint32`|Duration of time before a bid is considered out of date|
|`_feePercent`|`uint16`||
|`_borrowerAttestationRequired`|`bool`||
|`_lenderAttestationRequired`|`bool`||
|`_metadataURI`|`string`|A URI that points to a market's metadata. Requirements: - The caller must be the current owner.|


### setMarketFeeRecipient

Sets the fee recipient address for a market.


```solidity
function setMarketFeeRecipient(uint256 _marketId, address _recipient) public ownsMarket(_marketId);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The ID of a market.|
|`_recipient`|`address`|Address of the new fee recipient. Requirements: - The caller must be the current owner.|


### setMarketURI

Sets the metadata URI for a market.


```solidity
function setMarketURI(uint256 _marketId, string calldata _uri) public ownsMarket(_marketId);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The ID of a market.|
|`_uri`|`string`|A URI that points to a market's metadata. Requirements: - The caller must be the current owner.|


### setPaymentCycle

Sets the duration of new loans for this market before they turn delinquent.

Changing this value does not change the terms of existing loans for this market.


```solidity
function setPaymentCycle(uint256 _marketId, PaymentCycleType _paymentCycleType, uint32 _duration)
    public
    ownsMarket(_marketId);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The ID of a market.|
|`_paymentCycleType`|`PaymentCycleType`|Cycle type (seconds or monthly)|
|`_duration`|`uint32`|Delinquency duration for new loans|


### setPaymentDefaultDuration

Sets the duration of new loans for this market before they turn defaulted.

Changing this value does not change the terms of existing loans for this market.


```solidity
function setPaymentDefaultDuration(uint256 _marketId, uint32 _duration) public ownsMarket(_marketId);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The ID of a market.|
|`_duration`|`uint32`|Default duration for new loans|


### setBidExpirationTime


```solidity
function setBidExpirationTime(uint256 _marketId, uint32 _duration) public ownsMarket(_marketId);
```

### setMarketFeePercent

Sets the fee for the market.


```solidity
function setMarketFeePercent(uint256 _marketId, uint16 _newPercent) public ownsMarket(_marketId);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The ID of a market.|
|`_newPercent`|`uint16`|The percentage fee in basis points. Requirements: - The caller must be the current owner.|


### setMarketPaymentType

Set the payment type for the market.


```solidity
function setMarketPaymentType(uint256 _marketId, PaymentType _newPaymentType) public ownsMarket(_marketId);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The ID of the market.|
|`_newPaymentType`|`PaymentType`|The payment type for the market.|


### setLenderAttestationRequired

Enable/disables market whitelist for lenders.


```solidity
function setLenderAttestationRequired(uint256 _marketId, bool _required) public ownsMarket(_marketId);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The ID of a market.|
|`_required`|`bool`|Boolean indicating if the market requires whitelist. Requirements: - The caller must be the current owner.|


### setBorrowerAttestationRequired

Enable/disables market whitelist for borrowers.


```solidity
function setBorrowerAttestationRequired(uint256 _marketId, bool _required) public ownsMarket(_marketId);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The ID of a market.|
|`_required`|`bool`|Boolean indicating if the market requires whitelist. Requirements: - The caller must be the current owner.|


### getMarketData

Gets the data associated with a market.


```solidity
function getMarketData(uint256 _marketId)
    public
    view
    returns (
        address owner,
        uint32 paymentCycleDuration,
        uint32 paymentDefaultDuration,
        uint32 loanExpirationTime,
        string memory metadataURI,
        uint16 marketplaceFeePercent,
        bool lenderAttestationRequired
    );
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The ID of a market.|


### getMarketAttestationRequirements

Gets the attestation requirements for a given market.


```solidity
function getMarketAttestationRequirements(uint256 _marketId)
    public
    view
    returns (bool lenderAttestationRequired, bool borrowerAttestationRequired);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The ID of the market.|


### getMarketOwner

Gets the address of a market's owner.


```solidity
function getMarketOwner(uint256 _marketId) public view virtual override returns (address);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The ID of a market.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`address`|The address of a market's owner.|


### _getMarketOwner

Gets the address of a market's owner.


```solidity
function _getMarketOwner(uint256 _marketId) internal view virtual returns (address);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The ID of a market.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`address`|The address of a market's owner.|


### getMarketFeeRecipient

Gets the fee recipient of a market.


```solidity
function getMarketFeeRecipient(uint256 _marketId) public view override returns (address);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The ID of a market.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`address`|The address of a market's fee recipient.|


### getMarketURI

Gets the metadata URI of a market.


```solidity
function getMarketURI(uint256 _marketId) public view override returns (string memory);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The ID of a market.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`string`|URI of a market's metadata.|


### getPaymentCycle

Gets the loan delinquent duration of a market.


```solidity
function getPaymentCycle(uint256 _marketId) public view override returns (uint32, PaymentCycleType);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The ID of a market.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint32`|Duration of a loan until it is delinquent.|
|`<none>`|`PaymentCycleType`|The type of payment cycle for loans in the market.|


### getPaymentDefaultDuration

Gets the loan default duration of a market.


```solidity
function getPaymentDefaultDuration(uint256 _marketId) public view override returns (uint32);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The ID of a market.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`uint32`|Duration of a loan repayment interval until it is default.|


### getPaymentType

Get the payment type of a market.


```solidity
function getPaymentType(uint256 _marketId) public view override returns (PaymentType);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|the ID of the market.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`PaymentType`|The type of payment for loans in the market.|


### getBidExpirationTime


```solidity
function getBidExpirationTime(uint256 marketId) public view override returns (uint32);
```

### getMarketplaceFee

Gets the marketplace fee in basis points


```solidity
function getMarketplaceFee(uint256 _marketId) public view override returns (uint16 fee);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The ID of a market.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`fee`|`uint16`|in basis points|


### isVerifiedLender

Checks if a lender has been attested and added to a market.


```solidity
function isVerifiedLender(uint256 _marketId, address _lenderAddress)
    public
    view
    override
    returns (bool isVerified_, bytes32 uuid_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The ID of a market.|
|`_lenderAddress`|`address`|Address to check.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`isVerified_`|`bool`|Boolean indicating if a lender has been added to a market.|
|`uuid_`|`bytes32`|Bytes32 representing the UUID of the lender.|


### isVerifiedBorrower

Checks if a borrower has been attested and added to a market.


```solidity
function isVerifiedBorrower(uint256 _marketId, address _borrowerAddress)
    public
    view
    override
    returns (bool isVerified_, bytes32 uuid_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The ID of a market.|
|`_borrowerAddress`|`address`|Address of the borrower to check.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`isVerified_`|`bool`|Boolean indicating if a borrower has been added to a market.|
|`uuid_`|`bytes32`|Bytes32 representing the UUID of the borrower.|


### getAllVerifiedLendersForMarket

Gets addresses of all attested lenders.


```solidity
function getAllVerifiedLendersForMarket(uint256 _marketId, uint256 _page, uint256 _perPage)
    public
    view
    returns (address[] memory);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The ID of a market.|
|`_page`|`uint256`|Page index to start from.|
|`_perPage`|`uint256`|Number of items in a page to return.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`address[]`|Array of addresses that have been added to a market.|


### getAllVerifiedBorrowersForMarket

Gets addresses of all attested borrowers.


```solidity
function getAllVerifiedBorrowersForMarket(uint256 _marketId, uint256 _page, uint256 _perPage)
    public
    view
    returns (address[] memory);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The ID of the market.|
|`_page`|`uint256`|Page index to start from.|
|`_perPage`|`uint256`|Number of items in a page to return.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`address[]`|Array of addresses that have been added to a market.|


### _setMarketSettings

Sets multiple market settings for a given market.


```solidity
function _setMarketSettings(
    uint256 _marketId,
    uint32 _paymentCycleDuration,
    PaymentType _newPaymentType,
    PaymentCycleType _paymentCycleType,
    uint32 _paymentDefaultDuration,
    uint32 _bidExpirationTime,
    uint16 _feePercent,
    bool _borrowerAttestationRequired,
    bool _lenderAttestationRequired,
    string calldata _metadataURI
) internal;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The ID of a market.|
|`_paymentCycleDuration`|`uint32`|Delinquency duration for new loans|
|`_newPaymentType`|`PaymentType`|The payment type for the market.|
|`_paymentCycleType`|`PaymentCycleType`|The payment cycle type for loans in the market - Seconds or Monthly|
|`_paymentDefaultDuration`|`uint32`|Default duration for new loans|
|`_bidExpirationTime`|`uint32`|Duration of time before a bid is considered out of date|
|`_feePercent`|`uint16`||
|`_borrowerAttestationRequired`|`bool`||
|`_lenderAttestationRequired`|`bool`||
|`_metadataURI`|`string`|A URI that points to a market's metadata.|


### _getStakeholdersForMarket

Gets addresses of all attested relevant stakeholders.


```solidity
function _getStakeholdersForMarket(EnumerableSet.AddressSet storage _set, uint256 _page, uint256 _perPage)
    internal
    view
    returns (address[] memory stakeholders_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_set`|`AddressSet.EnumerableSet`|The stored set of stakeholders to index from.|
|`_page`|`uint256`|Page index to start from.|
|`_perPage`|`uint256`|Number of items in a page to return.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`stakeholders_`|`address[]`|Array of addresses that have been added to a market.|


### _attestStakeholder

Adds a stakeholder (lender or borrower) to a market.


```solidity
function _attestStakeholder(uint256 _marketId, address _stakeholderAddress, uint256 _expirationTime, bool _isLender)
    internal
    virtual
    withAttestingSchema(_isLender ? lenderAttestationSchemaId : borrowerAttestationSchemaId);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The market ID to add a borrower to.|
|`_stakeholderAddress`|`address`|The address of the stakeholder to add to the market.|
|`_expirationTime`|`uint256`|The expiration time of the attestation.|
|`_isLender`|`bool`|Boolean indicating if the stakeholder is a lender. Otherwise it is a borrower.|


### _attestStakeholderViaDelegation

Adds a stakeholder (lender or borrower) to a market via delegated attestation.

*The signature must match that of the market owner.*


```solidity
function _attestStakeholderViaDelegation(
    uint256 _marketId,
    address _stakeholderAddress,
    uint256 _expirationTime,
    bool _isLender,
    uint8 _v,
    bytes32 _r,
    bytes32 _s
) internal virtual withAttestingSchema(_isLender ? lenderAttestationSchemaId : borrowerAttestationSchemaId);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The market ID to add a lender to.|
|`_stakeholderAddress`|`address`|The address of the lender to add to the market.|
|`_expirationTime`|`uint256`|The expiration time of the attestation.|
|`_isLender`|`bool`|Boolean indicating if the stakeholder is a lender. Otherwise it is a borrower.|
|`_v`|`uint8`|Signature value|
|`_r`|`bytes32`|Signature value|
|`_s`|`bytes32`|Signature value|


### _attestStakeholderVerification

Adds a stakeholder (borrower/lender) to a market.


```solidity
function _attestStakeholderVerification(uint256 _marketId, address _stakeholderAddress, bytes32 _uuid, bool _isLender)
    internal
    virtual;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The market ID to add a stakeholder to.|
|`_stakeholderAddress`|`address`|The address of the stakeholder to add to the market.|
|`_uuid`|`bytes32`|The UUID of the attestation created.|
|`_isLender`|`bool`|Boolean indicating if the stakeholder is a lender. Otherwise it is a borrower.|


### _revokeStakeholder

Removes a stakeholder from an market.

*The caller must be the market owner.*


```solidity
function _revokeStakeholder(uint256 _marketId, address _stakeholderAddress, bool _isLender) internal virtual;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The market ID to remove the borrower from.|
|`_stakeholderAddress`|`address`|The address of the borrower to remove from the market.|
|`_isLender`|`bool`|Boolean indicating if the stakeholder is a lender. Otherwise it is a borrower.|


### _revokeStakeholderViaDelegation

Removes a stakeholder from an market via delegated revocation.


```solidity
function _revokeStakeholderViaDelegation(
    uint256 _marketId,
    address _stakeholderAddress,
    bool _isLender,
    uint8 _v,
    bytes32 _r,
    bytes32 _s
) internal;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The market ID to remove the borrower from.|
|`_stakeholderAddress`|`address`|The address of the borrower to remove from the market.|
|`_isLender`|`bool`|Boolean indicating if the stakeholder is a lender. Otherwise it is a borrower.|
|`_v`|`uint8`|Signature value|
|`_r`|`bytes32`|Signature value|
|`_s`|`bytes32`|Signature value|


### _revokeStakeholderVerification

Removes a stakeholder (borrower/lender) from a market.


```solidity
function _revokeStakeholderVerification(uint256 _marketId, address _stakeholderAddress, bool _isLender)
    internal
    virtual
    returns (bytes32 uuid_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_marketId`|`uint256`|The market ID to remove the lender from.|
|`_stakeholderAddress`|`address`|The address of the stakeholder to remove from the market.|
|`_isLender`|`bool`|Boolean indicating if the stakeholder is a lender. Otherwise it is a borrower.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`uuid_`|`bytes32`|The ID of the previously verified attestation.|


### _isVerified

Checks if a stakeholder has been attested and added to a market.


```solidity
function _isVerified(
    address _stakeholderAddress,
    bool _attestationRequired,
    mapping(address => bytes32) storage _stakeholderAttestationIds,
    EnumerableSet.AddressSet storage _verifiedStakeholderForMarket
) internal view virtual returns (bool isVerified_, bytes32 uuid_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_stakeholderAddress`|`address`|Address of the stakeholder to check.|
|`_attestationRequired`|`bool`|Stored boolean indicating if attestation is required for the stakeholder class.|
|`_stakeholderAttestationIds`|`mapping(address => bytes32)`|Mapping of attested Ids for the stakeholder class.|
|`_verifiedStakeholderForMarket`|`AddressSet.EnumerableSet`||


## Events
### MarketCreated

```solidity
event MarketCreated(address indexed owner, uint256 marketId);
```

### SetMarketURI

```solidity
event SetMarketURI(uint256 marketId, string uri);
```

### SetPaymentCycleDuration

```solidity
event SetPaymentCycleDuration(uint256 marketId, uint32 duration);
```

### SetPaymentCycle

```solidity
event SetPaymentCycle(uint256 marketId, PaymentCycleType paymentCycleType, uint32 value);
```

### SetPaymentDefaultDuration

```solidity
event SetPaymentDefaultDuration(uint256 marketId, uint32 duration);
```

### SetBidExpirationTime

```solidity
event SetBidExpirationTime(uint256 marketId, uint32 duration);
```

### SetMarketFee

```solidity
event SetMarketFee(uint256 marketId, uint16 feePct);
```

### LenderAttestation

```solidity
event LenderAttestation(uint256 marketId, address lender);
```

### BorrowerAttestation

```solidity
event BorrowerAttestation(uint256 marketId, address borrower);
```

### LenderRevocation

```solidity
event LenderRevocation(uint256 marketId, address lender);
```

### BorrowerRevocation

```solidity
event BorrowerRevocation(uint256 marketId, address borrower);
```

### MarketClosed

```solidity
event MarketClosed(uint256 marketId);
```

### LenderExitMarket

```solidity
event LenderExitMarket(uint256 marketId, address lender);
```

### BorrowerExitMarket

```solidity
event BorrowerExitMarket(uint256 marketId, address borrower);
```

### SetMarketOwner

```solidity
event SetMarketOwner(uint256 marketId, address newOwner);
```

### SetMarketFeeRecipient

```solidity
event SetMarketFeeRecipient(uint256 marketId, address newRecipient);
```

### SetMarketLenderAttestation

```solidity
event SetMarketLenderAttestation(uint256 marketId, bool required);
```

### SetMarketBorrowerAttestation

```solidity
event SetMarketBorrowerAttestation(uint256 marketId, bool required);
```

### SetMarketPaymentType

```solidity
event SetMarketPaymentType(uint256 marketId, PaymentType paymentType);
```

## Structs
### Marketplace

```solidity
struct Marketplace {
    address owner;
    string metadataURI;
    uint16 marketplaceFeePercent;
    bool lenderAttestationRequired;
    EnumerableSet.AddressSet verifiedLendersForMarket;
    mapping(address => bytes32) lenderAttestationIds;
    uint32 paymentCycleDuration;
    uint32 paymentDefaultDuration;
    uint32 bidExpirationTime;
    bool borrowerAttestationRequired;
    EnumerableSet.AddressSet verifiedBorrowersForMarket;
    mapping(address => bytes32) borrowerAttestationIds;
    address feeRecipient;
    PaymentType paymentType;
    PaymentCycleType paymentCycleType;
}
```

