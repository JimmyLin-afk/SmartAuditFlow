// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface ITellerV2Context {
    function setTrustedMarketForwarder(uint256 _marketId, address _forwarder)
        external;

    function approveMarketForwarder(uint256 _marketId, address _forwarder)
        external;
}
