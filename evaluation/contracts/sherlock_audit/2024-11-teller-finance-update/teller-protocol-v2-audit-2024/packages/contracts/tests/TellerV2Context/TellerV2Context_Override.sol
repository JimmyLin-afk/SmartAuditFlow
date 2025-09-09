// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/utils/structs/EnumerableSet.sol";

import { TellerV2Context } from "../../contracts/TellerV2Context.sol";
import { IMarketRegistry } from "../../contracts/interfaces/IMarketRegistry.sol";

contract TellerV2Context_Override is TellerV2Context {
    using EnumerableSet for EnumerableSet.AddressSet;

    constructor(address _marketRegistry, address _lenderCommitmentForwarder)
        TellerV2Context(address(0))
    {
        marketRegistry = IMarketRegistry(_marketRegistry);
        lenderCommitmentForwarder = _lenderCommitmentForwarder;
    }

    function mock_setTrustedMarketForwarder(
        uint256 _marketId,
        address _forwarder
    ) public {
        _trustedMarketForwarders[_marketId] = _forwarder;
    }

    function mock_setApprovedMarketForwarder(
        uint256 _marketId,
        address _forwarder,
        address _account,
        bool _approved
    ) public {
        _approvedForwarderSenders[_forwarder].add(_account);
    }

    function external__msgSenderForMarket(uint256 _marketId)
        public
        view
        returns (address)
    {
        return _msgSenderForMarket(_marketId);
    }
}
