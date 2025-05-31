// SPDX-License-Identifier: AGPL-3.0-or-later
pragma solidity ^0.8.7;

import { IERC20 }             from "../modules/erc20/src/interfaces/IERC20.sol";
import { IMapleProxyFactory } from "../modules/maple-proxy-factory/contracts/interfaces/IMapleProxyFactory.sol";

import { ERC20Helper } from "../modules/erc20-helper/src/ERC20Helper.sol";

import { IMapleLoan } from "./interfaces/IMapleLoan.sol";
import { IMapleGlobalsLike } from "./interfaces/Interfaces.sol";

import { MapleLoanInternals } from "./MapleLoanInternals.sol";

/// @title MapleLoan implements a primitive loan with additional functionality, and is intended to be proxied.
contract MapleLoan is IMapleLoan, MapleLoanInternals {

    modifier whenProtocolNotPaused() {
        require(!isProtocolPaused(), "ML:PROTOCOL_PAUSED");
        _;
    }

    /********************************/
    /*** Administrative Functions ***/
    /********************************/

    function migrate(address migrator_, bytes calldata arguments_) external override {
        require(msg.sender == _factory(),        "ML:M:NOT_FACTORY");
        require(_migrate(migrator_, arguments_), "ML:M:FAILED");
    }

    function setImplementation(address newImplementation_) external override {
        require(msg.sender == _factory(),               "ML:SI:NOT_FACTORY");
        require(_setImplementation(newImplementation_), "ML:SI:FAILED");
    }

    function upgrade(uint256 toVersion_, bytes calldata arguments_) external override {
        require(msg.sender == _borrower, "ML:U:NOT_BORROWER");

        emit Upgraded(toVersion_, arguments_);

        IMapleProxyFactory(_factory()).upgradeInstance(toVersion_, arguments_);
    }

    /************************/
    /*** Borrow Functions ***/
    /************************/

    function acceptBorrower() external override {
        require(msg.sender == _pendingBorrower, "ML:AB:NOT_PENDING_BORROWER");

        _pendingBorrower = address(0);

        emit BorrowerAccepted(_borrower = msg.sender);
    }

    function closeLoan(uint256 amount_) external override returns (uint256 principal_, uint256 interest_) {
        // The amount specified is an optional amount to be transfer from the caller, as a convenience for EOAs.
        require(amount_ == uint256(0) || ERC20Helper.transferFrom(_fundsAsset, msg.sender, address(this), amount_), "ML:CL:TRANSFER_FROM_FAILED");

        ( principal_, interest_ ) = _closeLoan();

        emit LoanClosed(principal_, interest_);
    }

    function drawdownFunds(uint256 amount_, address destination_) external override whenProtocolNotPaused returns (uint256 collateralPosted_) {
        require(msg.sender == _borrower, "ML:DF:NOT_BORROWER");

        emit FundsDrawnDown(amount_, destination_);

        // Post additional collateral required to facilitate this drawdown, if needed.
        uint256 additionalCollateralRequired = getAdditionalCollateralRequiredFor(amount_);

        if (additionalCollateralRequired > uint256(0)) {
            // Determine collateral currently unaccounted for.
            uint256 unaccountedCollateral = _getUnaccountedAmount(_collateralAsset);

            // Post required collateral, specifying then amount lacking as the optional amount to be transferred from.
            collateralPosted_ = postCollateral(
                additionalCollateralRequired > unaccountedCollateral ? additionalCollateralRequired - unaccountedCollateral : uint256(0)
            );
        }

        _drawdownFunds(amount_, destination_);
    }

    function makePayment(uint256 amount_) external override returns (uint256 principal_, uint256 interest_) {
        // The amount specified is an optional amount to be transfer from the caller, as a convenience for EOAs.
        require(amount_ == uint256(0) || ERC20Helper.transferFrom(_fundsAsset, msg.sender, address(this), amount_), "ML:MP:TRANSFER_FROM_FAILED");

        ( principal_, interest_ ) = _makePayment();

        emit PaymentMade(principal_, interest_);
    }

    function postCollateral(uint256 amount_) public override whenProtocolNotPaused returns (uint256 collateralPosted_) {
        // The amount specified is an optional amount to be transfer from the caller, as a convenience for EOAs.
        require(
            amount_ == uint256(0) || ERC20Helper.transferFrom(_collateralAsset, msg.sender, address(this), amount_),
            "ML:PC:TRANSFER_FROM_FAILED"
        );

        emit CollateralPosted(collateralPosted_ = _postCollateral());
    }

    function proposeNewTerms(address refinancer_, bytes[] calldata calls_) external override whenProtocolNotPaused {
        require(msg.sender == _borrower, "ML:PNT:NOT_BORROWER");

        emit NewTermsProposed(_proposeNewTerms(refinancer_, calls_), refinancer_, calls_);
    }

    function removeCollateral(uint256 amount_, address destination_) external override whenProtocolNotPaused {
        require(msg.sender == _borrower, "ML:RC:NOT_BORROWER");

        emit CollateralRemoved(amount_, destination_);

        _removeCollateral(amount_, destination_);
    }

    function returnFunds(uint256 amount_) public override whenProtocolNotPaused returns (uint256 fundsReturned_) {
        // The amount specified is an optional amount to be transfer from the caller, as a convenience for EOAs.
        require(amount_ == uint256(0) || ERC20Helper.transferFrom(_fundsAsset, msg.sender, address(this), amount_), "ML:RF:TRANSFER_FROM_FAILED");

        emit FundsReturned(fundsReturned_ = _returnFunds());
    }

    function setPendingBorrower(address pendingBorrower_) external override {
        require(msg.sender == _borrower, "ML:SPB:NOT_BORROWER");

        emit PendingBorrowerSet(_pendingBorrower = pendingBorrower_);
    }

    /**********************/
    /*** Lend Functions ***/
    /**********************/

    function acceptLender() external override {
        require(msg.sender == _pendingLender, "ML:AL:NOT_PENDING_LENDER");

        _pendingLender = address(0);

        emit LenderAccepted(_lender = msg.sender);
    }

    function acceptNewTerms(address refinancer_, bytes[] calldata calls_, uint256 amount_) external override whenProtocolNotPaused {
        require(msg.sender == _lender, "ML:ANT:NOT_LENDER");

        // The amount specified is an optional amount to be transfer from the caller, as a convenience for EOAs.
        require(amount_ == uint256(0) || ERC20Helper.transferFrom(_fundsAsset, msg.sender, address(this), amount_), "ML:ACT:TRANSFER_FROM_FAILED");

        emit NewTermsAccepted(_acceptNewTerms(refinancer_, calls_), refinancer_, calls_);

        uint256 extra = _getUnaccountedAmount(_fundsAsset);

        // NOTE: This block ensures unaccounted funds (pre-existing or due to over-funding) gets redirected to the lender.
        if (extra > uint256(0)) {
            emit FundsRedirected(extra, _lender);
            require(ERC20Helper.transfer(_fundsAsset, _lender, extra), "ML:ANT:TRANSFER_FAILED");
        }
    }

    function claimFunds(uint256 amount_, address destination_) external override whenProtocolNotPaused {
        require(msg.sender == _lender, "ML:CF:NOT_LENDER");

        emit FundsClaimed(amount_, destination_);

        _claimFunds(amount_, destination_);
    }

    function fundLoan(address lender_, uint256 amount_) external override whenProtocolNotPaused returns (uint256 fundsLent_) {
        // The amount specified is an optional amount to be transferred from the caller, as a convenience for EOAs.
        require(amount_ == uint256(0) || ERC20Helper.transferFrom(_fundsAsset, msg.sender, address(this), amount_), "ML:FL:TRANSFER_FROM_FAILED");

        // If the loan is not active, fund it.
        if (_nextPaymentDueDate == uint256(0)) {
            emit Funded(lender_, fundsLent_ = _fundLoan(lender_), _nextPaymentDueDate);
        }

        uint256 extra = _getUnaccountedAmount(_fundsAsset);

        // NOTE: This block is not only a stopgap solution to allow a LiquidityLockerV1 to send funds to a DebtLocker, while maintaining PoolV1 accounting,
        //       but also ensures unaccounted funds (pre-existing or due to over-funding) gets redirected to the lender.
        if (extra > uint256(0)) {
            emit FundsRedirected(extra, _lender);
            require(ERC20Helper.transfer(_fundsAsset, _lender, extra), "ML:FL:TRANSFER_FAILED");
        }
    }

    function repossess(address destination_) external override whenProtocolNotPaused returns (uint256 collateralRepossessed_, uint256 fundsRepossessed_) {
        require(msg.sender == _lender, "ML:R:NOT_LENDER");

        ( collateralRepossessed_, fundsRepossessed_ ) = _repossess(destination_);

        emit Repossessed(collateralRepossessed_, fundsRepossessed_, destination_);
    }

    function setPendingLender(address pendingLender_) external override {
        require(msg.sender == _lender, "ML:SPL:NOT_LENDER");

        emit PendingLenderSet(_pendingLender = pendingLender_);
    }

    /*******************************/
    /*** Miscellaneous Functions ***/
    /*******************************/

    function skim(address token_, address destination_) external override whenProtocolNotPaused returns (uint256 skimmed_) {
        require((msg.sender == _borrower) || (msg.sender == _lender),    "L:S:NO_AUTH");
        require((token_ != _fundsAsset) && (token_ != _collateralAsset), "L:S:INVALID_TOKEN");

        emit Skimmed(token_, skimmed_ = IERC20(token_).balanceOf(address(this)), destination_);

        require(ERC20Helper.transfer(token_, destination_, skimmed_), "L:S:TRANSFER_FAILED");
    }

    /**********************/
    /*** View Functions ***/
    /**********************/

    function getAdditionalCollateralRequiredFor(uint256 drawdown_) public view override returns (uint256 collateral_) {
        // Determine the collateral needed in the contract for a reduced drawable funds amount.
        uint256 collateralNeeded = _getCollateralRequiredFor(_principal, _drawableFunds - drawdown_, _principalRequested, _collateralRequired);

        return collateralNeeded > _collateral ? collateralNeeded - _collateral : uint256(0);
    }

    function getEarlyPaymentBreakdown() external view override returns (uint256 principal_, uint256 interest_) {
        ( principal_, interest_ ) = _getEarlyPaymentBreakdown();
    }

    function getNextPaymentBreakdown() external view override returns (uint256 principal_, uint256 interest_) {
        ( principal_, interest_ ) = _getNextPaymentBreakdown();
    }

    function isProtocolPaused() public view override returns (bool paused_) {
        return IMapleGlobalsLike(IMapleProxyFactory(_factory()).mapleGlobals()).protocolPaused();
    }

    /****************************/
    /*** State View Functions ***/
    /****************************/

    function borrower() external view override returns (address borrower_) {
        return _borrower;
    }

    function claimableFunds() external view override returns (uint256 claimableFunds_) {
        return _claimableFunds;
    }

    function collateral() external view override returns (uint256 collateral_) {
        return _collateral;
    }

    function collateralAsset() external view override returns (address collateralAsset_) {
        return _collateralAsset;
    }

    function collateralRequired() external view override returns (uint256 collateralRequired_) {
        return _collateralRequired;
    }

    function drawableFunds() external view override returns (uint256 drawableFunds_) {
        return _drawableFunds;
    }

    function earlyFeeRate() external view override returns (uint256 earlyFeeRate_) {
        return _earlyFeeRate;
    }

    function endingPrincipal() external view override returns (uint256 endingPrincipal_) {
        return _endingPrincipal;
    }

    function excessCollateral() external view override returns (uint256 excessCollateral_) {
        uint256 collateralNeeded = _getCollateralRequiredFor(_principal, _drawableFunds, _principalRequested, _collateralRequired);

        return _collateral > collateralNeeded ? _collateral - collateralNeeded : uint256(0);
    }

    function factory() external view override returns (address factory_) {
        return _factory();
    }

    function fundsAsset() external view override returns (address fundsAsset_) {
        return _fundsAsset;
    }

    function gracePeriod() external view override returns (uint256 gracePeriod_) {
        return _gracePeriod;
    }

    function implementation() external view override returns (address implementation_) {
        return _implementation();
    }

    function interestRate() external view override returns (uint256 interestRate_) {
        return _interestRate;
    }

    function lateFeeRate() external view override returns (uint256 lateFeeRate_) {
        return _lateFeeRate;
    }

    function lateInterestPremium() external view override returns (uint256 lateInterestPremium_) {
        return _lateInterestPremium;
    }

    function lender() external view override returns (address lender_) {
        return _lender;
    }

    function nextPaymentDueDate() external view override returns (uint256 nextPaymentDueDate_) {
        return _nextPaymentDueDate;
    }

    function paymentInterval() external view override returns (uint256 paymentInterval_) {
        return _paymentInterval;
    }

    function paymentsRemaining() external view override returns (uint256 paymentsRemaining_) {
        return _paymentsRemaining;
    }

    function pendingBorrower() external view override returns (address pendingBorrower_) {
        return _pendingBorrower;
    }

    function pendingLender() external view override returns (address pendingLender_) {
        return _pendingLender;
    }

    function principalRequested() external view override returns (uint256 principalRequested_) {
        return _principalRequested;
    }

    function principal() external view override returns (uint256 principal_) {
        return _principal;
    }

    // NOTE: This is needed for `fundLoan` call from PoolV1.
    function superFactory() external view override returns (address superFactory_) {
        return _factory();
    }

}