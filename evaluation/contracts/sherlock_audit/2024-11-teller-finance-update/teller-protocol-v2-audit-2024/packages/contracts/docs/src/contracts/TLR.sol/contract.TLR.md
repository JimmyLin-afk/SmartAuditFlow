# TLR
[Git Source](https://github.com/teller-protocol/teller-protocol-v2/blob/f4bf5a00ae7113b0344876c13db9b3dd705154f6/contracts/TLR.sol)

**Inherits:**
ERC20Votes, Ownable


## State Variables
### MAX_SUPPLY

```solidity
uint224 private immutable MAX_SUPPLY;
```


## Functions
### constructor

*Sets the value of the `cap`. This value is immutable, it can only be
set once during construction.*


```solidity
constructor(uint224 _supplyCap, address tokenOwner) ERC20("Teller", "TLR") ERC20Permit("Teller");
```

### _maxSupply

*Max supply has been overridden to cap the token supply upon initialization of the contract*

*See OpenZeppelin's implementation of ERC20Votes _mint() function*


```solidity
function _maxSupply() internal view override returns (uint224);
```

### mint

*Creates `amount` tokens and assigns them to `account`
Emits a {Transfer} event with `from` set to the zero address.
Requirements:
- `account` cannot be the zero address.*


```solidity
function mint(address account, uint256 amount) external onlyOwner;
```

### burn

*Destroys `amount` tokens from `account`, reducing the
total supply.
Emits a {Transfer} event with `to` set to the zero address.
Requirements:
- `account` cannot be the zero address.
- `account` must have at least `amount` tokens.*


```solidity
function burn(address account, uint256 amount) external onlyOwner;
```

