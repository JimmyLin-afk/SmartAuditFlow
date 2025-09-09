// SPDX-License-Identifier: BSD-3-Clause
pragma solidity 0.8.22;

import "./CToken.sol";

/**
 * @title Mach's CSonic Contract
 * @notice CToken which wraps Sonic
 * @author Mach
 */
contract CSonic is CToken {
    /**
     * @notice Construct a new CSonic money market
     * @param comptroller_ The address of the Comptroller
     * @param interestRateModel_ The address of the interest rate model
     * @param initialExchangeRateMantissa_ The initial exchange rate, scaled by 1e18
     * @param name_ ERC-20 name of this token
     * @param symbol_ ERC-20 symbol of this token
     * @param decimals_ ERC-20 decimal precision of this token
     * @param admin_ Address of the administrator of this token
     */
    constructor(
        ComptrollerInterface comptroller_,
        InterestRateModel interestRateModel_,
        uint256 initialExchangeRateMantissa_,
        string memory name_,
        string memory symbol_,
        uint8 decimals_,
        address payable admin_
    ) {
        // Creator of the contract is admin during initialization
        admin = payable(msg.sender);

        initialize(comptroller_, interestRateModel_, initialExchangeRateMantissa_, name_, symbol_, decimals_);

        // Set the proper admin now that initialization is done
        admin = admin_;
    }

    /**
     * User Interface **
     */

    /**
     * @notice Sender supplies assets into the market and receives cTokens in exchange
     * @dev Reverts upon any failure
     */
    function mint() external payable {
        mintInternal(msg.value);
    }

    /**
     * @notice Sender supplies assets into the market, enables it as collateral, and receives cTokens in exchange
     * @dev Accrues interest whether or not the operation succeeds, unless reverted
     * @return uint 0=success, otherwise a failure (see ErrorReporter.sol for details)
     */
    function mintAsCollateral() external payable returns (uint256) {
        address cToken = address(this);

        // Check if cToken is already used as collateral
        bool isCollateral = comptroller.checkMembership(msg.sender, cToken);
        mintInternal(msg.value);

        // If cToken is not used as collateral, enter market
        if (!isCollateral) {
            uint256 err = comptroller.enterMarketForCToken(cToken, msg.sender);
            if (err != NO_ERROR) {
                revert EnterMarketComptrollerRejection(err);
            }
        }

        return NO_ERROR;
    }

    /**
     * @notice Sender redeems cTokens in exchange for the underlying asset
     * @dev Accrues interest whether or not the operation succeeds, unless reverted
     * @param redeemTokens The number of cTokens to redeem into underlying
     * @return uint 0=success, otherwise a failure (see ErrorReporter.sol for details)
     */
    function redeem(uint256 redeemTokens) external returns (uint256) {
        redeemInternal(redeemTokens);
        return NO_ERROR;
    }

    /**
     * @notice Sender redeems cTokens in exchange for a specified amount of underlying asset
     * @dev Accrues interest whether or not the operation succeeds, unless reverted
     * @param redeemAmount The amount of underlying to redeem
     * @return uint 0=success, otherwise a failure (see ErrorReporter.sol for details)
     */
    function redeemUnderlying(uint256 redeemAmount) external returns (uint256) {
        redeemUnderlyingInternal(redeemAmount);
        return NO_ERROR;
    }

    /**
     * @notice Sender borrows assets from the protocol to their own address
     * @param borrowAmount The amount of the underlying asset to borrow
     * @return uint 0=success, otherwise a failure (see ErrorReporter.sol for details)
     */
    function borrow(uint256 borrowAmount) external returns (uint256) {
        borrowInternal(borrowAmount);
        return NO_ERROR;
    }

    /**
     * @notice Sender repays their own borrow
     * @dev Reverts upon any failure
     */
    function repayBorrow() external payable {
        repayBorrowInternal(msg.value);
    }

    /**
     * @notice Sender repays a borrow belonging to borrower
     * @dev Reverts upon any failure
     * @param borrower the account with the debt being payed off
     */
    function repayBorrowBehalf(address borrower) external payable {
        repayBorrowBehalfInternal(borrower, msg.value);
    }

    /**
     * @notice The sender liquidates the borrowers collateral.
     *  The collateral seized is transferred to the liquidator.
     * @dev Reverts upon any failure
     * @param borrower The borrower of this cToken to be liquidated
     * @param cTokenCollateral The market in which to seize collateral from the borrower
     */
    function liquidateBorrow(address borrower, CToken cTokenCollateral) external payable {
        liquidateBorrowInternal(borrower, msg.value, cTokenCollateral);
    }

    /**
     * @notice The sender adds to reserves.
     * @return uint 0=success, otherwise a failure (see ErrorReporter.sol for details)
     */
    function _addReserves() external payable returns (uint256) {
        return _addReservesInternal(msg.value);
    }

    /**
     * @notice Send Sonic to CSonic to mint
     */
    receive() external payable {
        mintInternal(msg.value);
    }

    /**
     * Safe Token **
     */

    /**
     * @notice Gets balance of this contract in terms of Sonic, before this message
     * @dev This excludes the value of the current message, if any
     * @return The quantity of Sonic owned by this contract
     */
    function getCashPrior() internal view override returns (uint256) {
        return address(this).balance - msg.value;
    }

    /**
     * @notice Admin function to sweep any ERC-20 token to the admin
     * @param token The address of the ERC-20 token to sweep
     */
    function sweepToken(EIP20NonStandardInterface token) external {
        require(msg.sender == admin, "CSonic::sweepToken: only admin can sweep tokens");
        uint256 balance = token.balanceOf(address(this));
        require(balance > 0, "CSonic::sweepToken: no balance to sweep");
        token.transfer(admin, balance);
    }

    /**
     * @notice Perform the actual transfer in, which is a no-op
     * @param from Address sending the Sonic
     * @param amount Amount of Sonic being sent
     * @return The actual amount of Sonic transferred
     */
    function doTransferIn(address from, uint256 amount) internal override returns (uint256) {
        // Sanity checks
        require(msg.sender == from, "sender mismatch");
        require(msg.value == amount, "value mismatch");
        return amount;
    }

    function doTransferOut(address payable to, uint256 amount) internal virtual override {
        /* Send the Sonic, with minimal gas and revert on failure */
        to.transfer(amount);
    }
}
