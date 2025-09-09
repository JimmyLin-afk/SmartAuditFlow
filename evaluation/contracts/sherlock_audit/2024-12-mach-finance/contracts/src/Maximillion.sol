// SPDX-License-Identifier: BSD-3-Clause
pragma solidity 0.8.22;

import "./CSonic.sol";

/**
 * @title Mach's Maximillion Contract
 * @author Mach
 */
contract Maximillion {
    /**
     * @notice The default cSonic market to repay in
     */
    CSonic public cSonic;

    /**
     * @notice Construct a Maximillion to repay max in a CSonic market
     */
    constructor(CSonic cSonic_) public {
        cSonic = cSonic_;
    }

    /**
     * @notice msg.sender sends Sonic to repay an account's borrow in the cSonic market
     * @dev The provided Sonic is applied towards the borrow balance, any excess is refunded
     * @param borrower The address of the borrower account to repay on behalf of
     */
    function repayBehalf(address borrower) public payable {
        repayBehalfExplicit(borrower, cSonic);
    }

    /**
     * @notice msg.sender sends Sonic to repay an account's borrow in a cSonic market
     * @dev The provided Sonic is applied towards the borrow balance, any excess is refunded
     * @param borrower The address of the borrower account to repay on behalf of
     * @param cSonic_ The address of the cSonic contract to repay in
     */
    function repayBehalfExplicit(address borrower, CSonic cSonic_) public payable {
        uint256 received = msg.value;
        uint256 borrows = cSonic_.borrowBalanceCurrent(borrower);
        if (received > borrows) {
            cSonic_.repayBorrowBehalf{value: borrows}(borrower);
            payable(msg.sender).transfer(received - borrows);
        } else {
            cSonic_.repayBorrowBehalf{value: received}(borrower);
        }
    }
}
