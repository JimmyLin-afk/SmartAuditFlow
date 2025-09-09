# ReputationManager
[Git Source](https://github.com/teller-protocol/teller-protocol-v2/blob/f4bf5a00ae7113b0344876c13db9b3dd705154f6/contracts/ReputationManager.sol)

**Inherits:**
IReputationManager, Initializable


## State Variables
### CONTROLLER

```solidity
bytes32 public constant CONTROLLER = keccak256("CONTROLLER");
```


### tellerV2

```solidity
ITellerV2 public tellerV2;
```


### _delinquencies

```solidity
mapping(address => EnumerableSet.UintSet) private _delinquencies;
```


### _defaults

```solidity
mapping(address => EnumerableSet.UintSet) private _defaults;
```


### _currentDelinquencies

```solidity
mapping(address => EnumerableSet.UintSet) private _currentDelinquencies;
```


### _currentDefaults

```solidity
mapping(address => EnumerableSet.UintSet) private _currentDefaults;
```


## Functions
### initialize

Initializes the proxy.


```solidity
function initialize(address _tellerV2) external initializer;
```

### getDelinquentLoanIds


```solidity
function getDelinquentLoanIds(address _account) public override returns (uint256[] memory);
```

### getDefaultedLoanIds


```solidity
function getDefaultedLoanIds(address _account) public override returns (uint256[] memory);
```

### getCurrentDelinquentLoanIds


```solidity
function getCurrentDelinquentLoanIds(address _account) public override returns (uint256[] memory);
```

### getCurrentDefaultLoanIds


```solidity
function getCurrentDefaultLoanIds(address _account) public override returns (uint256[] memory);
```

### updateAccountReputation


```solidity
function updateAccountReputation(address _account) public override;
```

### updateAccountReputation


```solidity
function updateAccountReputation(address _account, uint256 _bidId) public override returns (RepMark);
```

### _applyReputation


```solidity
function _applyReputation(address _account, uint256 _bidId) internal returns (RepMark mark_);
```

### _addMark


```solidity
function _addMark(address _account, uint256 _bidId, RepMark _mark) internal;
```

### _removeMark


```solidity
function _removeMark(address _account, uint256 _bidId, RepMark _mark) internal;
```

## Events
### MarkAdded

```solidity
event MarkAdded(address indexed account, RepMark indexed repMark, uint256 bidId);
```

### MarkRemoved

```solidity
event MarkRemoved(address indexed account, RepMark indexed repMark, uint256 bidId);
```

