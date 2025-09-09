pragma solidity >=0.8.0 <0.9.0;
// SPDX-License-Identifier: MIT

import { TellerV2 } from "../contracts/TellerV2.sol";
import "../contracts/mock/WethMock.sol";
import "../contracts/interfaces/IMarketRegistry.sol";
import "../contracts/interfaces/ITellerV2.sol";
import "../contracts/interfaces/ITellerV2Context.sol";
import { Collateral } from "../contracts/interfaces/escrow/ICollateralEscrowV1.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import { PaymentType } from "../contracts/libraries/V2Calculations.sol";

contract User {
    address public immutable tellerV2;

    constructor(address _tellerV2 /*, WethMock _wethMock*/) {
        tellerV2 = _tellerV2;
    }

    function addAllowance(
        address _assetContractAddress,
        address _spender,
        uint256 _amount
    ) public {
        IERC20(_assetContractAddress).approve(_spender, _amount);
    }

    function createMarketSimple(
        address marketRegistry,
        uint32 _paymentCycleDuration,
        uint32 _paymentDefaultDuration,
        uint32 _bidExpirationTime,
        uint16 _feePercent,
        bool _requireLenderAttestation,
        bool _requireBorrowerAttestation,
        string calldata _uri
    ) public returns (uint256) {
        return
            IMarketRegistry(marketRegistry).createMarket(
                address(this),
                _paymentCycleDuration,
                _paymentDefaultDuration,
                _bidExpirationTime,
                _feePercent,
                _requireLenderAttestation,
                _requireBorrowerAttestation,
                _uri
            );
    }

    function createMarket(
        address marketRegistry,
        uint32 _paymentCycleDuration,
        uint32 _paymentDefaultDuration,
        uint32 _bidExpirationTime,
        uint16 _feePercent,
        bool _requireLenderAttestation,
        bool _requireBorrowerAttestation,
        PaymentType _paymentType,
        PaymentCycleType _paymentCycleType,
        string calldata _uri
    ) public returns (uint256) {
        return
            IMarketRegistry(marketRegistry).createMarket(
                address(this),
                _paymentCycleDuration,
                _paymentDefaultDuration,
                _bidExpirationTime,
                _feePercent,
                _requireLenderAttestation,
                _requireBorrowerAttestation,
                _paymentType,
                _paymentCycleType,
                _uri
            );
    }

    function acceptBid(uint256 _bidId) public {
        ITellerV2(tellerV2).lenderAcceptBid(_bidId);
    }

    function submitBid(
        address _lendingToken,
        uint256 _marketplaceId,
        uint256 _principal,
        uint32 _duration,
        uint16 _APR,
        string calldata _metadataURI,
        address _receiver
    ) public returns (uint256) {
        return
            ITellerV2(tellerV2).submitBid(
                _lendingToken,
                _marketplaceId,
                _principal,
                _duration,
                _APR,
                _metadataURI,
                _receiver
            );
    }

    function submitCollateralBid(
        address _lendingToken,
        uint256 _marketplaceId,
        uint256 _principal,
        uint32 _duration,
        uint16 _APR,
        string calldata _metadataURI,
        address _receiver,
        Collateral[] calldata _collateralInfo
    ) public returns (uint256) {
        return
            ITellerV2(tellerV2).submitBid(
                _lendingToken,
                _marketplaceId,
                _principal,
                _duration,
                _APR,
                _metadataURI,
                _receiver,
                _collateralInfo
            );
    }

    function repayLoanFull(uint256 _bidId) public {
        return ITellerV2(tellerV2).repayLoanFull(_bidId);
    }

    function setTrustedMarketForwarder(uint256 _marketId, address _forwarder)
        external
    {
        ITellerV2Context(tellerV2).setTrustedMarketForwarder(
            _marketId,
            _forwarder
        );
    }

    function approveMarketForwarder(uint256 _marketId, address _forwarder)
        external
    {
        ITellerV2Context(tellerV2).approveMarketForwarder(
            _marketId,
            _forwarder
        );
    }

    receive() external payable {}
}
