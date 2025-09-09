# TellerV2Autopay
[Git Source](https://github.com/teller-protocol/teller-protocol-v2/blob/f4bf5a00ae7113b0344876c13db9b3dd705154f6/contracts/TellerV2Autopay.sol)

**Inherits:**
OwnableUpgradeable, ITellerV2Autopay

*Helper contract to autopay loans*


## State Variables
### tellerV2

```solidity
ITellerV2 public immutable tellerV2;
```


### loanAutoPayEnabled

```solidity
mapping(uint256 => bool) public loanAutoPayEnabled;
```


### _autopayFee

```solidity
uint16 private _autopayFee;
```


## Functions
### constructor


```solidity
constructor(address _protocolAddress);
```

### initialize

Initialized the proxy.


```solidity
function initialize(uint16 _fee, address _owner) external initializer;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_fee`|`uint16`|The fee collected for automatic payment processing.|
|`_owner`|`address`|The address of the ownership to be transferred to.|


### setAutopayFee

Let the owner of the contract set a new autopay fee.


```solidity
function setAutopayFee(uint16 _newFee) public virtual onlyOwner;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_newFee`|`uint16`|The new autopay fee to set.|


### _setAutopayFee


```solidity
function _setAutopayFee(uint16 _newFee) internal;
```

### getAutopayFee

Returns the current autopay fee.


```solidity
function getAutopayFee() public view virtual returns (uint16);
```

### setAutoPayEnabled

Function for a borrower to enable or disable autopayments


```solidity
function setAutoPayEnabled(uint256 _bidId, bool _autoPayEnabled) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the bid to cancel.|
|`_autoPayEnabled`|`bool`|boolean for allowing autopay on a loan|


### autoPayLoanMinimum

Function for a minimum autopayment to be performed on a loan


```solidity
function autoPayLoanMinimum(uint256 _bidId) external;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the bid to repay.|


### getEstimatedMinimumPayment


```solidity
function getEstimatedMinimumPayment(uint256 _bidId) public virtual returns (uint256 _amount);
```

## Events
### AutoPaidLoanMinimum
This event is emitted when a loan is autopaid.


```solidity
event AutoPaidLoanMinimum(uint256 indexed bidId, address indexed msgsender);
```

### AutoPayEnabled
This event is emitted when loan autopayments are enabled or disabled.


```solidity
event AutoPayEnabled(uint256 indexed bidId, bool enabled);
```

### AutopayFeeSet
This event is emitted when the autopay fee has been updated.


```solidity
event AutopayFeeSet(uint16 newFee, uint16 oldFee);
```

