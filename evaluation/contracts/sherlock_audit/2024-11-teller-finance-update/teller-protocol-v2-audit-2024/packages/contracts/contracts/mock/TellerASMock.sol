// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "../EAS/TellerAS.sol";
import "../EAS/TellerASEIP712Verifier.sol";
import "../EAS/TellerASRegistry.sol";

import "../interfaces/IASRegistry.sol";
import "../interfaces/IEASEIP712Verifier.sol";

contract TellerASMock is TellerAS {
    constructor()
        TellerAS(
            IASRegistry(new TellerASRegistry()),
            IEASEIP712Verifier(new TellerASEIP712Verifier())
        )
    {}

    function isAttestationActive(bytes32 uuid)
        public
        view
        override
        returns (bool)
    {
        return true;
    }
}
