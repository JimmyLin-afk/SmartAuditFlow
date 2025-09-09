// SPDX-License-Identifier: MIT
pragma solidity 0.8.26;

contract BeaconDepositMock {
    mapping(bytes => address) public pubkeyToOperator;

    function setOperator(bytes memory pubkey, address operator) public {
        pubkeyToOperator[pubkey] = operator;
    }

    function getOperator(bytes memory pubkey) public view returns (address) {
        return pubkeyToOperator[pubkey];
    }
}