// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/utils/Address.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "../contracts/TellerV2MarketForwarder_G1.sol";

import "../contracts/TellerV2Context.sol";

import { Testable } from "./Testable.sol";

import { Collateral, CollateralType } from "../contracts/interfaces/escrow/ICollateralEscrowV1.sol";

import { User } from "./Test_Helpers.sol";

import "../contracts/mock/MarketRegistryMock.sol";
import "../contracts/mock/CollateralManagerMock.sol";

import "../contracts/MarketLiquidityRewards.sol";

contract MarketLiquidityRewards_Override is MarketLiquidityRewards {
    uint256 immutable startTime = 1678122531;

    //  address tokenAddress;
    uint256 marketId;
    uint256 maxAmount;

    address[] emptyArray;
    address[] borrowersArray;

    uint32 maxLoanDuration;
    uint16 minInterestRate;
    uint32 expiration;

    bool public verifyLoanStartTimeWasCalled;

    bool public verifyRewardRecipientWasCalled;
    bool public verifyCollateralAmountWasCalled;

    constructor(
        address tellerV2,
        address marketRegistry,
        address collateralManager
    ) MarketLiquidityRewards(tellerV2, marketRegistry, collateralManager) {}

    function setAllocation(
        uint256 _allocationId,
        RewardAllocation memory _allocation
    ) public {
        allocatedRewards[_allocationId] = _allocation;
    }

    function setAllocatedAmount(uint256 _allocationId, uint256 _amount) public {
        allocatedRewards[_allocationId].rewardTokenAmount = _amount;
    }

    function calculateRewardAmount(
        uint256 loanPrincipal,
        uint32 loanDuration,
        uint256 principalTokenDecimals,
        uint256 rewardPerLoanPrincipalAmount
    ) public view returns (uint256) {
        return
            super._calculateRewardAmount(
                loanPrincipal,
                loanDuration,
                principalTokenDecimals,
                rewardPerLoanPrincipalAmount
            );
    }

    function requiredCollateralAmount(
        uint256 loanPrincipal,
        uint256 principalTokenDecimals,
        uint256 collateralTokenDecimals,
        uint256 minimumCollateralPerPrincipal
    ) public view returns (uint256) {
        return
            super._requiredCollateralAmount(
                loanPrincipal,
                principalTokenDecimals,
                collateralTokenDecimals,
                minimumCollateralPerPrincipal
            );
    }

    function decrementAllocatedAmount(
        uint256 _allocationId,
        uint256 _tokenAmount
    ) public {
        super._decrementAllocatedAmount(_allocationId, _tokenAmount);
    }

    function verifyLoanStartTime(uint32 a, uint32 b, uint32 c) public {
        super._verifyLoanStartTime(a, b, c);
    }

    function verifyAndReturnRewardRecipient(
        AllocationStrategy allocationStrategy,
        BidState bidState,
        address borrower,
        address lender
    ) public returns (address) {
        return
            super._verifyAndReturnRewardRecipient(
                allocationStrategy,
                bidState,
                borrower,
                lender
            );
    }

    //overrides

    function allocateRewards(
        MarketLiquidityRewards.RewardAllocation calldata _allocation
    ) public override returns (uint256 allocationId_) {
        super.allocateRewards(_allocation);
    }

    function increaseAllocationAmount(
        uint256 _allocationId,
        uint256 _tokenAmount
    ) public override {
        super.increaseAllocationAmount(_allocationId, _tokenAmount);
    }

    function deallocateRewards(uint256 _allocationId, uint256 _tokenAmount)
        public
        override
    {
        super.deallocateRewards(_allocationId, _tokenAmount);
    }

    function _verifyAndReturnRewardRecipient(
        AllocationStrategy strategy,
        BidState bidState,
        address borrower,
        address lender
    ) internal override returns (address rewardRecipient) {
        verifyRewardRecipientWasCalled = true;
        return address(borrower);
    }

    function _verifyCollateralAmount(
        address _collateralTokenAddress,
        uint256 _collateralAmount,
        address _principalTokenAddress,
        uint256 _principalAmount,
        uint256 _minimumCollateralPerPrincipalAmount
    ) internal override {
        verifyCollateralAmountWasCalled = true;
    }

    function _verifyLoanStartTime(
        uint32 loanStartTime,
        uint32 minStartTime,
        uint32 maxStartTime
    ) internal override {
        verifyLoanStartTimeWasCalled = true;
    }
}
