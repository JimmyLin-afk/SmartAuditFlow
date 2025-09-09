# TellerV2Storage_G0
[Git Source](https://github.com/teller-protocol/teller-protocol-v2/blob/f4bf5a00ae7113b0344876c13db9b3dd705154f6/contracts/TellerV2Storage.sol)


## State Variables
### bidId
Storage Variables


```solidity
uint256 public bidId = 0;
```


### bids

```solidity
mapping(uint256 => Bid) public bids;
```


### borrowerBids

```solidity
mapping(address => uint256[]) public borrowerBids;
```


### __lenderVolumeFilled

```solidity
mapping(address => uint256) public __lenderVolumeFilled;
```


### __totalVolumeFilled

```solidity
uint256 public __totalVolumeFilled;
```


### __lendingTokensSet

```solidity
EnumerableSet.AddressSet internal __lendingTokensSet;
```


### marketRegistry

```solidity
IMarketRegistry public marketRegistry;
```


### reputationManager

```solidity
IReputationManager public reputationManager;
```


### _borrowerBidsActive

```solidity
mapping(address => EnumerableSet.UintSet) internal _borrowerBidsActive;
```


### bidDefaultDuration

```solidity
mapping(uint256 => uint32) public bidDefaultDuration;
```


### bidExpirationTime

```solidity
mapping(uint256 => uint32) public bidExpirationTime;
```


### lenderVolumeFilled

```solidity
mapping(address => mapping(address => uint256)) public lenderVolumeFilled;
```


### totalVolumeFilled

```solidity
mapping(address => uint256) public totalVolumeFilled;
```


### version

```solidity
uint256 public version;
```


### uris

```solidity
mapping(uint256 => string) public uris;
```


