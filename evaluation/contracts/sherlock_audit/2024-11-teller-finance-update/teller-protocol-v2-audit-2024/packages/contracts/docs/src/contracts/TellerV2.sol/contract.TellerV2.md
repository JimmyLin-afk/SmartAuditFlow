# TellerV2
[Git Source](https://github.com/teller-protocol/teller-protocol-v2/blob/f4bf5a00ae7113b0344876c13db9b3dd705154f6/contracts/TellerV2.sol)

**Inherits:**
ITellerV2, OwnableUpgradeable, [ProtocolFee](/contracts/ProtocolFee.sol/contract.ProtocolFee.md), PausableUpgradeable, [TellerV2Storage](/contracts/TellerV2Storage.sol/abstract.TellerV2Storage.md), [TellerV2Context](/contracts/TellerV2Context.sol/abstract.TellerV2Context.md)


## State Variables
### CURRENT_CODE_VERSION
Constant Variables *


```solidity
uint8 public constant CURRENT_CODE_VERSION = 9;
```


### LIQUIDATION_DELAY

```solidity
uint32 public constant LIQUIDATION_DELAY = 86400;
```


## Functions
### pendingBid

Modifiers

This modifier is used to check if the state of a bid is pending, before running an action.


```solidity
modifier pendingBid(uint256 _bidId, string memory _action);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the bid to check the state for.|
|`_action`|`string`|The desired action to run on the bid.|


### acceptedLoan

This modifier is used to check if the state of a loan has been accepted, before running an action.


```solidity
modifier acceptedLoan(uint256 _bidId, string memory _action);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the bid to check the state for.|
|`_action`|`string`|The desired action to run on the bid.|


### constructor

Constructor *


```solidity
constructor(address trustedForwarder) TellerV2Context(trustedForwarder);
```

### initialize

External Functions *

Initializes the proxy.


```solidity
function initialize(
    uint16 _protocolFee,
    address _marketRegistry,
    address _reputationManager,
    address _lenderCommitmentForwarder,
    address _collateralManager,
    address _lenderManager
) external initializer;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_protocolFee`|`uint16`|The fee collected by the protocol for loan processing.|
|`_marketRegistry`|`address`|The address of the market registry contract for the protocol.|
|`_reputationManager`|`address`|The address of the reputation manager contract.|
|`_lenderCommitmentForwarder`|`address`|The address of the lender commitment forwarder contract.|
|`_collateralManager`|`address`|The address of the collateral manager contracts.|
|`_lenderManager`|`address`|The address of the lender manager contract for loans on the protocol.|


### setLenderManager


```solidity
function setLenderManager(address _lenderManager) external reinitializer(8) onlyOwner;
```

### _setLenderManager


```solidity
function _setLenderManager(address _lenderManager) internal onlyInitializing;
```

### getMetadataURI

Gets the metadataURI for a bidId.


```solidity
function getMetadataURI(uint256 _bidId) public view returns (string memory metadataURI_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the bid to return the metadataURI for|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`metadataURI_`|`string`|The metadataURI for the bid, as a string.|


### setReputationManager

Lets the DAO/owner of the protocol to set a new reputation manager contract.


```solidity
function setReputationManager(address _reputationManager) public onlyOwner;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_reputationManager`|`address`|The new contract address.|


### submitBid

Function for a borrower to create a bid for a loan without Collateral.


```solidity
function submitBid(
    address _lendingToken,
    uint256 _marketplaceId,
    uint256 _principal,
    uint32 _duration,
    uint16 _APR,
    string calldata _metadataURI,
    address _receiver
) public override whenNotPaused returns (uint256 bidId_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_lendingToken`|`address`|The lending token asset requested to be borrowed.|
|`_marketplaceId`|`uint256`|The unique id of the marketplace for the bid.|
|`_principal`|`uint256`|The principal amount of the loan bid.|
|`_duration`|`uint32`|The recurrent length of time before which a payment is due.|
|`_APR`|`uint16`|The proposed interest rate for the loan bid.|
|`_metadataURI`|`string`|The URI for additional borrower loan information as part of loan bid.|
|`_receiver`|`address`|The address where the loan amount will be sent to.|


### submitBid

Function for a borrower to create a bid for a loan with Collateral.


```solidity
function submitBid(
    address _lendingToken,
    uint256 _marketplaceId,
    uint256 _principal,
    uint32 _duration,
    uint16 _APR,
    string calldata _metadataURI,
    address _receiver,
    Collateral[] calldata _collateralInfo
) public override whenNotPaused returns (uint256 bidId_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_lendingToken`|`address`|The lending token asset requested to be borrowed.|
|`_marketplaceId`|`uint256`|The unique id of the marketplace for the bid.|
|`_principal`|`uint256`|The principal amount of the loan bid.|
|`_duration`|`uint32`|The recurrent length of time before which a payment is due.|
|`_APR`|`uint16`|The proposed interest rate for the loan bid.|
|`_metadataURI`|`string`|The URI for additional borrower loan information as part of loan bid.|
|`_receiver`|`address`|The address where the loan amount will be sent to.|
|`_collateralInfo`|`Collateral[]`|Additional information about the collateral asset.|


### _submitBid


```solidity
function _submitBid(
    address _lendingToken,
    uint256 _marketplaceId,
    uint256 _principal,
    uint32 _duration,
    uint16 _APR,
    string calldata _metadataURI,
    address _receiver
) internal virtual returns (uint256 bidId_);
```

### cancelBid

Function for a borrower to cancel their pending bid.


```solidity
function cancelBid(uint256 _bidId) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the bid to cancel.|


### marketOwnerCancelBid

Function for a market owner to cancel a bid in the market.


```solidity
function marketOwnerCancelBid(uint256 _bidId) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the bid to cancel.|


### _cancelBid

Function for users to cancel a bid.


```solidity
function _cancelBid(uint256 _bidId) internal virtual pendingBid(_bidId, "cancelBid");
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the bid to be cancelled.|


### lenderAcceptBid

Function for a lender to accept a proposed loan bid.


```solidity
function lenderAcceptBid(uint256 _bidId)
    external
    override
    pendingBid(_bidId, "lenderAcceptBid")
    whenNotPaused
    returns (uint256 amountToProtocol, uint256 amountToMarketplace, uint256 amountToBorrower);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the loan bid to accept.|


### claimLoanNFT


```solidity
function claimLoanNFT(uint256 _bidId) external acceptedLoan(_bidId, "claimLoanNFT") whenNotPaused;
```

### repayLoanMinimum

Function for users to make the minimum amount due for an active loan.


```solidity
function repayLoanMinimum(uint256 _bidId) external acceptedLoan(_bidId, "repayLoan");
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the loan to make the payment towards.|


### repayLoanFull

Function for users to repay an active loan in full.


```solidity
function repayLoanFull(uint256 _bidId) external acceptedLoan(_bidId, "repayLoan");
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the loan to make the payment towards.|


### repayLoan

Function for users to make a payment towards an active loan.


```solidity
function repayLoan(uint256 _bidId, uint256 _amount) external acceptedLoan(_bidId, "repayLoan");
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the loan to make the payment towards.|
|`_amount`|`uint256`|The amount of the payment.|


### pauseProtocol

Lets the DAO/owner of the protocol implement an emergency stop mechanism.


```solidity
function pauseProtocol() public virtual onlyOwner whenNotPaused;
```

### unpauseProtocol

Lets the DAO/owner of the protocol undo a previously implemented emergency stop.


```solidity
function unpauseProtocol() public virtual onlyOwner whenPaused;
```

### liquidateLoanFull

Function for users to liquidate a defaulted loan.


```solidity
function liquidateLoanFull(uint256 _bidId) external acceptedLoan(_bidId, "liquidateLoan");
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the loan to make the payment towards.|


### _repayLoan

Internal function to make a loan payment.


```solidity
function _repayLoan(uint256 _bidId, Payment memory _payment, uint256 _owedAmount, bool _shouldWithdrawCollateral)
    internal
    virtual;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the loan to make the payment towards.|
|`_payment`|`Payment`|The Payment struct with payments amounts towards principal and interest respectively.|
|`_owedAmount`|`uint256`|The total amount owed on the loan.|
|`_shouldWithdrawCollateral`|`bool`||


### calculateAmountOwed

Calculates the total amount owed for a bid.


```solidity
function calculateAmountOwed(uint256 _bidId) public view returns (Payment memory owed);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the loan bid to calculate the owed amount for.|


### calculateAmountOwed

Calculates the total amount owed for a loan bid at a specific timestamp.


```solidity
function calculateAmountOwed(uint256 _bidId, uint256 _timestamp) public view returns (Payment memory owed);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the loan bid to calculate the owed amount for.|
|`_timestamp`|`uint256`|The timestamp at which to calculate the loan owed amount at.|


### calculateAmountDue

Calculates the minimum payment amount due for a loan.


```solidity
function calculateAmountDue(uint256 _bidId) public view returns (Payment memory due);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the loan bid to get the payment amount for.|


### calculateAmountDue

Calculates the minimum payment amount due for a loan at a specific timestamp.


```solidity
function calculateAmountDue(uint256 _bidId, uint256 _timestamp) public view returns (Payment memory due);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the loan bid to get the payment amount for.|
|`_timestamp`|`uint256`|The timestamp at which to get the due payment at.|


### calculateNextDueDate

Returns the next due date for a loan payment.


```solidity
function calculateNextDueDate(uint256 _bidId) public view returns (uint32 dueDate_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the loan bid.|


### isPaymentLate

Checks to see if a borrower is delinquent.


```solidity
function isPaymentLate(uint256 _bidId) public view override returns (bool);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the loan bid to check for.|


### isLoanDefaulted

Checks to see if a borrower is delinquent.


```solidity
function isLoanDefaulted(uint256 _bidId) public view override returns (bool);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the loan bid to check for.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`bool`|bool True if the loan is defaulted.|


### isLoanLiquidateable

Checks to see if a loan was delinquent for longer than liquidation delay.


```solidity
function isLoanLiquidateable(uint256 _bidId) public view override returns (bool);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the loan bid to check for.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`bool`|bool True if the loan is liquidateable.|


### _canLiquidateLoan

Checks to see if a borrower is delinquent.


```solidity
function _canLiquidateLoan(uint256 _bidId, uint32 _liquidationDelay) internal view returns (bool);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the loan bid to check for.|
|`_liquidationDelay`|`uint32`|Amount of additional seconds after a loan defaulted to allow a liquidation.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`<none>`|`bool`|bool True if the loan is liquidateable.|


### getBidState


```solidity
function getBidState(uint256 _bidId) external view override returns (BidState);
```

### getBorrowerActiveLoanIds


```solidity
function getBorrowerActiveLoanIds(address _borrower) external view override returns (uint256[] memory);
```

### getBorrowerLoanIds


```solidity
function getBorrowerLoanIds(address _borrower) external view returns (uint256[] memory);
```

### isLoanExpired

Checks to see if a pending loan has expired so it is no longer able to be accepted.


```solidity
function isLoanExpired(uint256 _bidId) public view returns (bool);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the loan bid to check for.|


### lastRepaidTimestamp

Returns the last repaid timestamp for a loan.


```solidity
function lastRepaidTimestamp(uint256 _bidId) public view returns (uint32);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the loan bid to get the timestamp for.|


### getLoanBorrower

Returns the borrower address for a given bid.


```solidity
function getLoanBorrower(uint256 _bidId) public view returns (address borrower_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the bid/loan to get the borrower for.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`borrower_`|`address`|The address of the borrower associated with the bid.|


### getLoanLender

Returns the lender address for a given bid. If the stored lender address is the `LenderManager` NFT address, return the `ownerOf` for the bid ID.


```solidity
function getLoanLender(uint256 _bidId) public view returns (address lender_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the bid/loan to get the lender for.|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`lender_`|`address`|The address of the lender associated with the bid.|


### getLoanLendingToken


```solidity
function getLoanLendingToken(uint256 _bidId) external view returns (address token_);
```

### getLoanMarketId


```solidity
function getLoanMarketId(uint256 _bidId) external view returns (uint256 _marketId);
```

### getLoanSummary


```solidity
function getLoanSummary(uint256 _bidId)
    external
    view
    returns (
        address borrower,
        address lender,
        uint256 marketId,
        address principalTokenAddress,
        uint256 principalAmount,
        uint32 acceptedTimestamp,
        BidState bidState
    );
```

### _msgSender

OpenZeppelin Override Functions *


```solidity
function _msgSender()
    internal
    view
    virtual
    override(ERC2771ContextUpgradeable, ContextUpgradeable)
    returns (address sender);
```

### _msgData


```solidity
function _msgData()
    internal
    view
    virtual
    override(ERC2771ContextUpgradeable, ContextUpgradeable)
    returns (bytes calldata);
```

## Events
### SubmittedBid
Events

This event is emitted when a new bid is submitted.


```solidity
event SubmittedBid(uint256 indexed bidId, address indexed borrower, address receiver, bytes32 indexed metadataURI);
```

### AcceptedBid
This event is emitted when a bid has been accepted by a lender.


```solidity
event AcceptedBid(uint256 indexed bidId, address indexed lender);
```

### CancelledBid
This event is emitted when a previously submitted bid has been cancelled.


```solidity
event CancelledBid(uint256 indexed bidId);
```

### MarketOwnerCancelledBid
This event is emitted when market owner has cancelled a pending bid in their market.


```solidity
event MarketOwnerCancelledBid(uint256 indexed bidId);
```

### LoanRepayment
This event is emitted when a payment is made towards an active loan.


```solidity
event LoanRepayment(uint256 indexed bidId);
```

### LoanRepaid
This event is emitted when a loan has been fully repaid.


```solidity
event LoanRepaid(uint256 indexed bidId);
```

### LoanLiquidated
This event is emitted when a loan has been fully repaid.


```solidity
event LoanLiquidated(uint256 indexed bidId, address indexed liquidator);
```

### FeePaid
This event is emitted when a fee has been paid related to a bid.


```solidity
event FeePaid(uint256 indexed bidId, string indexed feeType, uint256 indexed amount);
```

