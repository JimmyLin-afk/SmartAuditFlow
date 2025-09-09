pragma solidity ^0.8.0;
// SPDX-License-Identifier: MIT

import "./Testable.sol";
import "../contracts/ProtocolFee.sol";

contract ProtocolFee_Override is Initializable, ProtocolFee {
    constructor() {}

    function initialize(uint16 initialFee) external initializer {
        __ProtocolFee_init(initialFee);
    }

    function protocolFeeInit(uint16 initialFee) public initializer {
        __ProtocolFee_init(initialFee);
    }
}
