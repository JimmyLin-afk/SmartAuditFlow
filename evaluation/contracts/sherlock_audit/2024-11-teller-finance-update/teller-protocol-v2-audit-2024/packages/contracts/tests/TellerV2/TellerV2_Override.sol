// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { TellerV2, Bid, BidState, Collateral, Payment, LoanDetails, Terms } from "../../contracts/TellerV2.sol";

import "../../contracts/interfaces/IMarketRegistry.sol";
import "../../contracts/interfaces/IReputationManager.sol";
import "../../contracts/interfaces/ICollateralManager.sol";
import "../../contracts/interfaces/ILenderManager.sol";

contract TellerV2_Override is TellerV2 {
    bool public submitBidWasCalled;
    bool public cancelBidWasCalled;
    bool public repayLoanWasCalled;
    address public mockMsgSenderForMarket;

    constructor() TellerV2(address(0)) {}

  
    function mock_setBid(uint256 bidId, Bid memory bid) public {
        bids[bidId] = bid;
    }

    function mock_addUriToMapping(uint256 bidId, string memory uri) public {
        uris[bidId] = uri;
    }

    function mock_setLenderManager(address _lenderManager) public {
        lenderManager = ILenderManager(_lenderManager);
    }


     function setProtocolPausingManagerSuper(address _protocolPausingManager) public {
         super._setProtocolPausingManager(_protocolPausingManager);
    }


    function setLenderManagerSuper(address _lenderManager) public initializer {
        lenderManager = ILenderManager(_lenderManager);
    }

    function setMarketRegistrySuper(address _marketRegistry) public {
        marketRegistry = IMarketRegistry(_marketRegistry);
    }

    function setCollateralManagerSuper(address _collateralManager) public {
        collateralManager = ICollateralManager(_collateralManager);
    }

    function setReputationManagerSuper(address _reputationManager) public {
        reputationManager = IReputationManager(_reputationManager);
    }

    function mock_setBidState(uint256 bidId, BidState state) public {
        bids[bidId].state = state;
    }

    function mock_setBidPaymentCycleType(
        uint256 bidId,
        PaymentCycleType paymentCycleType
    ) public {
        bidPaymentCycleType[bidId] = PaymentCycleType(paymentCycleType);
    }

    function mock_setBidLastRepaidTimestamp(
        uint256 bidId,
        uint32 lastRepaidTimestamp
    ) public {
        bids[bidId].loanDetails.lastRepaidTimestamp = lastRepaidTimestamp;
    }

    function mock_setBidDefaultDuration(uint256 bidId, uint32 defaultDuration)
        public
    {
        bidDefaultDuration[bidId] = defaultDuration;
    }

    function mock_setBidExpirationTime(uint256 bidId, uint32 expirationTime)
        public
    {
        bidExpirationTime[bidId] = expirationTime;
    }

    function mock_initialize() public initializer {
        __ProtocolFee_init(0);

      //  __Pausable_init();
    }

    function setMockMsgSenderForMarket(address _sender) public {
        mockMsgSenderForMarket = _sender;
    }

    function _submitBidSuper(
        address _lendingToken,
        uint256 _marketplaceId,
        uint256 _principal,
        uint32 _duration,
        uint16 _APR,
        string calldata _metadataURI,
        address _receiver
    ) public returns (uint256) {
        return
            super._submitBid(
                _lendingToken,
                _marketplaceId,
                _principal,
                _duration,
                _APR,
                _metadataURI,
                _receiver
            );
    }

    function _cancelBidSuper(uint256 bidId) public {
        super._cancelBid(bidId);
    }

    function _repayLoanSuper(
        uint256 _bidId,
        Payment memory _payment,
        uint256 _owedAmount,
        bool _shouldWithdrawCollateral
    ) public {
        super._repayLoan(
            _bidId,
            _payment,
            _owedAmount,
            _shouldWithdrawCollateral
        );
    }

    function _canLiquidateLoanSuper(uint256 _bidId, uint32 _liquidationDelay)
        public
        view
        returns (bool)
    {
        return _isLoanDefaulted(_bidId, _liquidationDelay);
    }

    /*

    Overrides 

    */

    function _msgSenderForMarket(uint256 _marketId)
        internal
        view
        override
        returns (address)
    {
        if (mockMsgSenderForMarket != address(0)) {
            return mockMsgSenderForMarket;
        }
        return super._msgSenderForMarket(_marketId);
    }

    function _submitBid(
        address _lendingToken,
        uint256 _marketplaceId,
        uint256 _principal,
        uint32 _duration,
        uint16 _APR,
        string calldata _metadataURI,
        address _receiver
    ) internal override returns (uint256 _bidId) {
        submitBidWasCalled = true;
        _bidId = 0;
    }

    function _cancelBid(uint256 _bidId) internal override {
        cancelBidWasCalled = true;
    }

    function _repayLoan(
        uint256 _bidId,
        Payment memory _payment,
        uint256 _owedAmount,
        bool _shouldWithdrawCollateral
    ) internal override {
        repayLoanWasCalled = true;
    }
}
