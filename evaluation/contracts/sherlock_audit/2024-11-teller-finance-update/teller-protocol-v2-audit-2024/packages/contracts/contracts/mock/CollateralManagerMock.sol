pragma solidity ^0.8.0;

// SPDX-License-Identifier: MIT

import "../interfaces/ICollateralManager.sol";

contract CollateralManagerMock is ICollateralManager {
    bool public committedCollateralValid = true;
    bool public deployAndDepositWasCalled;

    function commitCollateral(
        uint256 _bidId,
        Collateral[] calldata _collateralInfo
    ) external returns (bool validation_) {
        validation_ = committedCollateralValid;
    }

    function commitCollateral(
        uint256 _bidId,
        Collateral calldata _collateralInfo
    ) external returns (bool validation_) {
        validation_ = committedCollateralValid;
    }

    function checkBalances(
        address _borrowerAddress,
        Collateral[] calldata _collateralInfo
    ) external returns (bool validated_, bool[] memory checks_) {
        validated_ = true;
        checks_ = new bool[](0);
    }

    /**
     * @notice Deploys a new collateral escrow.
     * @param _bidId The associated bidId of the collateral escrow.
     */
    function deployAndDeposit(uint256 _bidId) external {
        deployAndDepositWasCalled = true;
    }

    /**
     * @notice Gets the address of a deployed escrow.
     * @notice _bidId The bidId to return the escrow for.
     * @return The address of the escrow.
     */
    function getEscrow(uint256 _bidId) external view returns (address) {
        return address(0);
    }

    /**
     * @notice Gets the collateral info for a given bid id.
     * @param _bidId The bidId to return the collateral info for.
     */
    function getCollateralInfo(uint256 _bidId)
        external
        view
        returns (Collateral[] memory collateral_)
    {
        collateral_ = new Collateral[](0);
    }

    function getCollateralAmount(uint256 _bidId, address collateralAssetAddress)
        external
        view
        returns (uint256 _amount)
    {
        return 500;
    }

    /**
     * @notice Withdraws deposited collateral from the created escrow of a bid.
     * @param _bidId The id of the bid to withdraw collateral for.
     */
    function withdraw(uint256 _bidId) external {}

    /**
     * @notice Re-checks the validity of a borrower's collateral balance committed to a bid.
     * @param _bidId The id of the associated bid.
     * @return validation_ Boolean indicating if the collateral balance was validated.
     */
    function revalidateCollateral(uint256 _bidId) external returns (bool) {
        return true;
    }

    function lenderClaimCollateral(uint256 _bidId) external {}

    function lenderClaimCollateralWithRecipient(uint256 _bidId, address _collateralRecipient) external {}


    /**
     * @notice Sends the deposited collateral to a liquidator of a bid.
     * @notice Can only be called by the protocol.
     * @param _bidId The id of the liquidated bid.
     * @param _liquidatorAddress The address of the liquidator to send the collateral to.
     */
    function liquidateCollateral(uint256 _bidId, address _liquidatorAddress)
        external
    {}

    function forceSetCommitCollateralValidation(bool _validation) external {
        committedCollateralValid = _validation;
    }
}
