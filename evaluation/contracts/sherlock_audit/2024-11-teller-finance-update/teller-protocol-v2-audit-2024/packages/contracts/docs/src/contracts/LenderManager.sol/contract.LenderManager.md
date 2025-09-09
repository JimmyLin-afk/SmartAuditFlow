# LenderManager
[Git Source](https://github.com/teller-protocol/teller-protocol-v2/blob/f4bf5a00ae7113b0344876c13db9b3dd705154f6/contracts/LenderManager.sol)

**Inherits:**
Initializable, OwnableUpgradeable, ERC721Upgradeable, ILenderManager


## State Variables
### marketRegistry

```solidity
IMarketRegistry public immutable marketRegistry;
```


## Functions
### constructor


```solidity
constructor(IMarketRegistry _marketRegistry);
```

### initialize


```solidity
function initialize() external initializer;
```

### __LenderManager_init


```solidity
function __LenderManager_init() internal onlyInitializing;
```

### registerLoan

Registers a new active lender for a loan, minting the nft


```solidity
function registerLoan(uint256 _bidId, address _newLender) public override onlyOwner;
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id for the loan to set.|
|`_newLender`|`address`|The address of the new active lender.|


### _getLoanMarketId

Returns the address of the lender that owns a given loan/bid.


```solidity
function _getLoanMarketId(uint256 _bidId) internal view returns (uint256);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_bidId`|`uint256`|The id of the bid of which to return the market id|


### _hasMarketVerification

Returns the verification status of a lender for a market.


```solidity
function _hasMarketVerification(address _lender, uint256 _bidId) internal view virtual returns (bool isVerified_);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`_lender`|`address`|The address of the lender which should be verified by the market|
|`_bidId`|`uint256`|The id of the bid of which to return the market id|


### _beforeTokenTransfer

ERC721 Functions *


```solidity
function _beforeTokenTransfer(address, address to, uint256 tokenId, uint256) internal override;
```

### _baseURI


```solidity
function _baseURI() internal view override returns (string memory);
```

