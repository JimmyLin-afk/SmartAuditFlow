# PaymentNotMinimum
[Git Source](https://github.com/teller-protocol/teller-protocol-v2/blob/f4bf5a00ae7113b0344876c13db9b3dd705154f6/contracts/TellerV2.sol)

This error is reverted when repayment amount is less than the required minimum


```solidity
error PaymentNotMinimum(uint256 bidId, uint256 payment, uint256 minimumOwed);
```

