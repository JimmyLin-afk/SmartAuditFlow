// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import {IWasabiPerps} from "../IWasabiPerps.sol";
import {Hash} from "../Hash.sol";

contract MockHasher {
    using Hash for IWasabiPerps.Position;
    using Hash for IWasabiPerps.OpenPositionRequest;
    using Hash for IWasabiPerps.ClosePositionRequest;

    function hashFunctionCallDataList(IWasabiPerps.FunctionCallData[] memory functionCallDataList) public pure returns (bytes32) {
        return Hash.hashFunctionCallDataList(functionCallDataList);
    }

    function hashPosition(IWasabiPerps.Position memory position) public pure returns (bytes32) {
        return position.hash();
    }

    function hashOpenPositionRequest(IWasabiPerps.OpenPositionRequest memory request) public pure returns (bytes32) {
        return request.hash();
    }

    function hashClosePositionRequest(IWasabiPerps.ClosePositionRequest memory request) public pure returns (bytes32) {
        return request.hash();
    }
}