// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IProtocolFee {
    function protocolFee() external view returns (uint16);
}
