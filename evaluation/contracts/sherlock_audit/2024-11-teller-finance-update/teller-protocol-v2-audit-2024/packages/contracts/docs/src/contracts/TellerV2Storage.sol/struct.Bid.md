# Bid
[Git Source](https://github.com/teller-protocol/teller-protocol-v2/blob/f4bf5a00ae7113b0344876c13db9b3dd705154f6/contracts/TellerV2Storage.sol)

Details about a loan request.


```solidity
struct Bid {
    address borrower;
    address receiver;
    address lender;
    uint256 marketplaceId;
    bytes32 _metadataURI;
    LoanDetails loanDetails;
    Terms terms;
    BidState state;
    PaymentType paymentType;
}
```

