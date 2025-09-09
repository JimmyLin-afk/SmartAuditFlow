// SPDX-License-Identifier: BSD-3-Clause
pragma solidity 0.8.22;

import {IRewardDistributor} from "../../src/Rewards/IRewardDistributor.sol";
import {CToken} from "../../src/CToken.sol";

contract MockRewardDistributor is IRewardDistributor {
    uint256 public updateSupplyIndexCount;
    uint256 public updateBorrowIndexCount;

    uint256 public disburseSupplierRewardsCount;
    uint256 public disburseBorrowerRewardsCount;

    uint256 public updateSupplyIndexAndDisburseSupplierRewardsCount;
    uint256 public updateBorrowIndexAndDisburseBorrowerRewardsCount;

    constructor() {}

    function updateSupplyIndexAndDisburseSupplierRewards(CToken cToken, address supplier) external {
        updateSupplyIndexAndDisburseSupplierRewardsCount++;
    }

    function updateBorrowIndexAndDisburseBorrowerRewards(CToken cToken, address borrower) external {
        updateBorrowIndexAndDisburseBorrowerRewardsCount++;
    }

    function disburseSupplierRewards(CToken cToken, address supplier) external {
        disburseSupplierRewardsCount++;
    }

    function disburseBorrowerRewards(CToken cToken, address borrower) external {
        disburseBorrowerRewardsCount++;
    }

    function updateBorrowIndex(CToken cToken) external {
        updateBorrowIndexCount++;
    }

    function updateSupplyIndex(CToken cToken) external {
        updateSupplyIndexCount++;
    }
}
