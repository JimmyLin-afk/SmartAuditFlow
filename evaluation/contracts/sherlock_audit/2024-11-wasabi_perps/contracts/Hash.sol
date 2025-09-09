// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import {IWasabiPerps} from "./IWasabiPerps.sol";

library Hash {
    bytes32 private constant _FUNCTION_CALL_DATA_HASH =
        keccak256("FunctionCallData(address to,uint256 value,bytes data)");
    bytes32 private constant _OPEN_POSITION_REQUEST_HASH =
        keccak256("OpenPositionRequest(uint256 id,address currency,address targetCurrency,uint256 downPayment,uint256 principal,uint256 minTargetAmount,uint256 expiration,uint256 fee,FunctionCallData[] functionCallDataList,Position existingPosition,address referrer)FunctionCallData(address to,uint256 value,bytes data)Position(uint256 id,address trader,address currency,address collateralCurrency,uint256 lastFundingTimestamp,uint256 downPayment,uint256 principal,uint256 collateralAmount,uint256 feesToBePaid)");
    bytes32 private constant _POSITION_HASH =
        keccak256("Position(uint256 id,address trader,address currency,address collateralCurrency,uint256 lastFundingTimestamp,uint256 downPayment,uint256 principal,uint256 collateralAmount,uint256 feesToBePaid)");
    bytes32 private constant _CLOSE_POSITION_REQUEST_HASH =
        keccak256("ClosePositionRequest(uint256 expiration,uint256 interest,uint256 amount,Position position,FunctionCallData[] functionCallDataList,address referrer)FunctionCallData(address to,uint256 value,bytes data)Position(uint256 id,address trader,address currency,address collateralCurrency,uint256 lastFundingTimestamp,uint256 downPayment,uint256 principal,uint256 collateralAmount,uint256 feesToBePaid)");
    bytes32 private constant _CLOSE_POSITION_ORDER_HASH =
        keccak256("ClosePositionOrder(uint8 orderType,uint256 positionId,uint256 createdAt,uint256 expiration,uint256 makerAmount,uint256 takerAmount,uint256 executionFee)");
    bytes32 private constant _ADD_COLLATERAL_REQUEST_HASH =
        keccak256("AddCollateralRequest(uint256 amount,uint256 interest,uint256 expiration,Position position)Position(uint256 id,address trader,address currency,address collateralCurrency,uint256 lastFundingTimestamp,uint256 downPayment,uint256 principal,uint256 collateralAmount,uint256 feesToBePaid)");
    bytes32 private constant _REMOVE_COLLATERAL_REQUEST_HASH =
        keccak256("RemoveCollateralRequest(uint256 amount,uint256 expiration,Position position)Position(uint256 id,address trader,address currency,address collateralCurrency,uint256 lastFundingTimestamp,uint256 downPayment,uint256 principal,uint256 collateralAmount,uint256 feesToBePaid)");

    /// @dev hashes the given FunctionCallData list
    /// @param functionCallDataList The list of function call data to hash
    function hashFunctionCallDataList(IWasabiPerps.FunctionCallData[] memory functionCallDataList) internal pure returns (bytes32) {
        uint256 length = functionCallDataList.length;
        bytes32[] memory functionCallDataHashes = new bytes32[](length);
        for (uint256 i = 0; i < length; ++i) {
            functionCallDataHashes[i] = keccak256(
                abi.encode(
                    _FUNCTION_CALL_DATA_HASH,
                    functionCallDataList[i].to,
                    functionCallDataList[i].value,
                    keccak256(functionCallDataList[i].data)
                )
            );
        }
        return keccak256(abi.encodePacked(functionCallDataHashes));
    }

    /// @dev Hashes the given Position
    /// @param _position the position
    function hash(IWasabiPerps.Position memory _position) internal pure returns (bytes32) {
        return keccak256(abi.encode(
            _POSITION_HASH,
            _position.id,
            _position.trader,
            _position.currency,
            _position.collateralCurrency,
            _position.lastFundingTimestamp,
            _position.downPayment,
            _position.principal,
            _position.collateralAmount,
            _position.feesToBePaid
        ));
    }

    /// @dev Hashes the given OpenPositionRequest
    /// @param _request The request to hash
    function hash(IWasabiPerps.OpenPositionRequest memory _request) internal pure returns (bytes32) {
        return keccak256(abi.encode(
            _OPEN_POSITION_REQUEST_HASH,
            _request.id,
            _request.currency,
            _request.targetCurrency,
            _request.downPayment,
            _request.principal,
            _request.minTargetAmount,
            _request.expiration,
            _request.fee,
            hashFunctionCallDataList(_request.functionCallDataList),
            hash(_request.existingPosition),
            _request.referrer
        ));
    }

    /// @dev Hashes the given AddCollateralRequest
    /// @param _request The request to hash
    function hash(IWasabiPerps.AddCollateralRequest memory _request) internal pure returns (bytes32) {
        return keccak256(abi.encode(
            _ADD_COLLATERAL_REQUEST_HASH,
            _request.amount,
            _request.interest,
            _request.expiration,
            hash(_request.position)
        ));
    }

    /// @dev Hashes the given RemoveCollateralRequest
    /// @param _request The request to hash
    function hash(IWasabiPerps.RemoveCollateralRequest memory _request) internal pure returns (bytes32) {
        return keccak256(abi.encode(
            _REMOVE_COLLATERAL_REQUEST_HASH,
            _request.amount,
            _request.expiration,
            hash(_request.position)
        ));
    }

    /// @dev Hashes the given ClosePositionRequest
    /// @param _request The request to hash
    function hash(IWasabiPerps.ClosePositionRequest memory _request) internal pure returns (bytes32) {
        return keccak256(abi.encode(
            _CLOSE_POSITION_REQUEST_HASH,
            _request.expiration,
            _request.interest,
            _request.amount,
            hash(_request.position),
            hashFunctionCallDataList(_request.functionCallDataList),
            _request.referrer
        ));
    }

    /// @dev Hashes the given ClosePositionOrder
    /// @param _order The order to hash
    function hash(IWasabiPerps.ClosePositionOrder memory _order) internal pure returns (bytes32) {
        return keccak256(abi.encode(
            _CLOSE_POSITION_ORDER_HASH,
            _order.orderType,
            _order.positionId,
            _order.createdAt,
            _order.expiration,
            _order.makerAmount,
            _order.takerAmount,
            _order.executionFee
        ));
    }
}