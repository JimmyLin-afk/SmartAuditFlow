// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { Testable } from "./Testable.sol";

import { TellerV2 } from "../contracts/TellerV2.sol";
import { MarketRegistry } from "../contracts/MarketRegistry.sol";

import "../contracts/TellerV2Context.sol";

import "../contracts/TellerV2Storage.sol";

import "../contracts/interfaces/IMarketRegistry.sol";

import "../contracts/EAS/TellerAS.sol";

import "../contracts/mock/WethMock.sol";
import "../contracts/interfaces/IWETH.sol";

import { User } from "./Test_Helpers.sol";
import { PaymentType, PaymentCycleType } from "../contracts/libraries/V2Calculations.sol";

contract MarketRegistry_Override is MarketRegistry {
    using EnumerableSet for EnumerableSet.AddressSet;

    address globalMarketOwner;
    address globalFeeRecipient;

    bool public attestStakeholderWasCalled;
    bool public attestStakeholderVerificationWasCalled;
    bool public attestStakeholderViaDelegationWasCalled;
    bool public revokeStakeholderWasCalled;
    bool public revokeStakeholderVerificationWasCalled;

    constructor() MarketRegistry() {}

    function stubMarket(uint256 marketId, address marketOwner) public {
        markets[marketId].owner = marketOwner;
        markets[marketId].marketplaceFeePercent = 500;
        markets[marketId].paymentCycleDuration = 500;
        markets[marketId].paymentDefaultDuration = 500;
        markets[marketId].bidExpirationTime = 500;
    }

    function setMarketOwner(address _owner) public {
        globalMarketOwner = _owner;
    }

    function attestStakeholder(
        uint256 _marketId,
        address _stakeholderAddress,
        uint256 _expirationTime,
        bool _isLender
    ) public {
        super._attestStakeholder(
            _marketId,
            _stakeholderAddress,
            _expirationTime,
            _isLender
        );
    }

    function revokeStakeholder(
        uint256 _marketId,
        address _stakeholderAddress,
        bool _isLender
    ) public {
        super._revokeStakeholder(_marketId, _stakeholderAddress, _isLender);
    }

    function attestStakeholderVerification(
        uint256 _marketId,
        address _stakeholderAddress,
        bytes32 _uuid,
        bool _isLender
    ) public {
        super._attestStakeholderVerification(
            _marketId,
            _stakeholderAddress,
            _uuid,
            _isLender
        );
    }

    function revokeStakeholderVerification(
        uint256 _marketId,
        address _stakeholderAddress,
        bool _isLender
    ) public {
        super._revokeStakeholderVerification(
            _marketId,
            _stakeholderAddress,
            _isLender
        );
    }

    function forceVerifyLenderForMarket(uint256 _marketId, address guy) public {
        markets[_marketId].verifiedLendersForMarket.add(guy);
    }

    function forceVerifyBorrowerForMarket(uint256 _marketId, address guy)
        public
    {
        markets[_marketId].verifiedBorrowersForMarket.add(guy);
    }

    function marketVerifiedLendersContains(uint256 _marketId, address guy)
        public
        returns (bool)
    {
        return markets[_marketId].verifiedLendersForMarket.contains(guy);
    }

    function marketVerifiedBorrowersContains(uint256 _marketId, address guy)
        public
        returns (bool)
    {
        return markets[_marketId].verifiedBorrowersForMarket.contains(guy);
    }

    function getLenderAttestationId(uint256 _marketId, address guy)
        public
        returns (bytes32)
    {
        return markets[_marketId].lenderAttestationIds[guy];
    }

    function getBorrowerAttestationId(uint256 _marketId, address guy)
        public
        returns (bytes32)
    {
        return markets[_marketId].borrowerAttestationIds[guy];
    }

    /*
    @notice returns the actual value in the markets storage mapping, not globalMarketOwner the override
    */
    function getMarketOwner(uint256 marketId)
        public
        view
        override
        returns (address)
    {
        return super._getMarketOwner(marketId);
    }

    function setFeeRecipient(uint256 _marketId, address _feeRecipient) public {
        markets[_marketId].feeRecipient = _feeRecipient;
    }

    function isVerified(address _stakeholderAddress, uint256 _marketId)
        public
        returns (bool isVerified_, bytes32 uuid_)
    {
        (isVerified_, uuid_) = super._isVerified(
            _stakeholderAddress,
            markets[_marketId].lenderAttestationRequired,
            markets[_marketId].lenderAttestationIds,
            markets[_marketId].verifiedLendersForMarket
        );
    }

    //overrides

    function _getMarketOwner(uint256 marketId)
        internal
        view
        override
        returns (address)
    {
        return globalMarketOwner;
    }

    function _attestStakeholder(
        uint256 _marketId,
        address _stakeholderAddress,
        uint256 _expirationTime,
        bool _isLender
    ) internal override {
        attestStakeholderWasCalled = true;
    }

    function _attestStakeholderVerification(
        uint256 _marketId,
        address _stakeholderAddress,
        bytes32 _uuid,
        bool _isLender
    ) internal override {
        attestStakeholderVerificationWasCalled = true;
    }

    function _attestStakeholderViaDelegation(
        uint256 _marketId,
        address _stakeholderAddress,
        uint256 _expirationTime,
        bool _isLender,
        uint8 _v,
        bytes32 _r,
        bytes32 _s
    ) internal override {
        attestStakeholderViaDelegationWasCalled = true;
    }

    function _revokeStakeholder(
        uint256 _marketId,
        address _stakeholderAddress,
        bool _isLender
    ) internal override {
        revokeStakeholderWasCalled = true;
    }

    function _revokeStakeholderVerification(
        uint256 _marketId,
        address _stakeholderAddress,
        bool _isLender
    ) internal override returns (bytes32 uuid_) {
        revokeStakeholderVerificationWasCalled = true;
    }

    function _isVerified(
        address _stakeholderAddress,
        bool _attestationRequired,
        mapping(address => bytes32) storage _stakeholderAttestationIds,
        EnumerableSet.AddressSet storage _verifiedStakeholderForMarket
    ) internal view override returns (bool isVerified_, bytes32 uuid_) {
        isVerified_ = true;
        uuid_ = bytes32("0x42");
    }
}
