// SPDX-License-Identifier: BSD-3-Clause
pragma solidity 0.8.22;

import {CToken} from "../CToken.sol";

interface IRewardDistributor {
    /**
     * Comptroller calls these functions to update the index and disburse rewards
     */
    function updateSupplyIndexAndDisburseSupplierRewards(CToken _cToken, address _supplier) external;

    function updateBorrowIndexAndDisburseBorrowerRewards(CToken _cToken, address _borrower) external;

    function disburseSupplierRewards(CToken _cToken, address _supplier) external;

    function disburseBorrowerRewards(CToken _cToken, address _borrower) external;

    function updateBorrowIndex(CToken _cToken) external;

    function updateSupplyIndex(CToken _cToken) external;
}
