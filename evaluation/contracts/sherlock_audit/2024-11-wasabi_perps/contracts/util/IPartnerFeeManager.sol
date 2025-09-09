// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

interface IPartnerFeeManager {
    error AddressNotPartner();
    error CallerNotPool();
    error InvalidFeeShareBips();

    event FeesAccrued(address indexed partner, address indexed feeToken, uint256 amount);
    event FeesClaimed(address indexed partner, address indexed feeToken, uint256 amount);

    /// @dev Returns true if the given address is a partner
    /// @param partner the address to check 
    /// @return true if the given address is a partner, false otherwise
    function isPartner(address partner) external view returns (bool);

    /// @dev Returns the accrued fees for the given partner and fee token
    /// @param partner the partner to get accrued fees for
    /// @param feeToken the fee token to get accrued fees for
    /// @return the accrued fees for the given partner and fee token
    function getAccruedFees(address partner, address feeToken) external view returns (uint256);

    /// @dev Computes the portion of the total fees that will be paid to the given partner
    /// @param partner the partner to compute fees for
    /// @param totalFees the total fees for the trade
    /// @return the partner fees for the given partner and total fees
    function computePartnerFees(address partner, uint256 totalFees) external view returns (uint256);

    /// @dev Accrues fees for the given partner and fee token
    /// @param partner the partner to accrue fees to
    /// @param feeToken the fee token to accrue fees in
    /// @param partnerFees the fees to accrue to the partner
    function accrueFees(address partner, address feeToken, uint256 partnerFees) external;

    /// @dev Claims fees for the given fee tokens if the caller is a partner
    /// @param feeTokens the fee tokens to claim
    function claimFees(address[] calldata feeTokens) external;

    /// @dev Sets the partner fee share percentage
    /// @param partner the partner to set the fee share for
    /// @param feeShareBips the fee share in basis points
    function setFeeShareBips(address partner, uint256 feeShareBips) external;

    /// @dev Adds fees to the accrued fees for the given partner and fee token
    /// @param partner the partner to accrue fees to
    /// @param feeToken the fee token to accrue fees in
    /// @param amount the amount of fees to accrue to the partner
    function adminAddFees(address partner, address feeToken, uint256 amount) external;
}
