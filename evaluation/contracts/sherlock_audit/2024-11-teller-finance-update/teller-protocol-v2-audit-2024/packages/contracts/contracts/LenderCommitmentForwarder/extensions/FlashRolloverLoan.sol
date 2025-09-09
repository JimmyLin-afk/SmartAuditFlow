// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;
 
import "./FlashRolloverLoan_G5.sol";

contract FlashRolloverLoan is FlashRolloverLoan_G5 {
    constructor(
        address _tellerV2,
        address _poolAddressesProvider
    )
        FlashRolloverLoan_G5(
            _tellerV2,
            _poolAddressesProvider
        )
    {}
}
