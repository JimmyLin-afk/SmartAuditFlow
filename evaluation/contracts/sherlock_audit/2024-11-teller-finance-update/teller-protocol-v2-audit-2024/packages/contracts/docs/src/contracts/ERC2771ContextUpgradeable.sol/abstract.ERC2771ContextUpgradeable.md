# ERC2771ContextUpgradeable
[Git Source](https://github.com/teller-protocol/teller-protocol-v2/blob/f4bf5a00ae7113b0344876c13db9b3dd705154f6/contracts/ERC2771ContextUpgradeable.sol)

**Inherits:**
Initializable, ContextUpgradeable

*Context variant with ERC2771 support.*

*This is modified from the OZ library to remove the gap of storage variables at the end.*


## State Variables
### _trustedForwarder

```solidity
address private immutable _trustedForwarder;
```


## Functions
### constructor


```solidity
constructor(address trustedForwarder);
```

### isTrustedForwarder


```solidity
function isTrustedForwarder(address forwarder) public view virtual returns (bool);
```

### _msgSender


```solidity
function _msgSender() internal view virtual override returns (address sender);
```

### _msgData


```solidity
function _msgData() internal view virtual override returns (bytes calldata);
```

