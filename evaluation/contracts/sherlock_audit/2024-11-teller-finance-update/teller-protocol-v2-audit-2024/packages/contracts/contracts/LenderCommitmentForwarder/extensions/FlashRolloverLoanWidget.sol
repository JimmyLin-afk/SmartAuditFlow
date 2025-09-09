// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;
 
import "./FlashRolloverLoan_G6.sol";

contract FlashRolloverLoanWidget is FlashRolloverLoan_G6 {
    constructor(
        address _tellerV2,
        address _poolAddressesProvider
    )
        FlashRolloverLoan_G6(
            _tellerV2,
            _poolAddressesProvider
        )
    {}
}
