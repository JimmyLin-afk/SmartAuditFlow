// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

interface IStrategy {
    error OnlyVault();

    /// @dev The vault that the strategy is attached to
    function vault() external view returns (address);

    /// @dev The asset that the strategy deposits and withdraws
    function asset() external view returns (address);

    /// @dev The collateral asset that the strategy holds
    function collateralAsset() external view returns (address);

    /// @dev Deposits the asset into the strategy
    /// @notice The vault should transfer the asset to the strategy before calling this function
    /// @param amount The amount of asset to deposit
    /// @return collateral The collateral asset that the strategy holds
    /// @return collateralIncreased The amount of collateral that the strategy received
    function deposit(uint256 amount) external returns (address collateral, uint256 collateralIncreased);

    /// @dev Withdraws the asset from the strategy
    /// @param amount The amount of collateral to withdraw
    /// @return collateral The collateral asset that the strategy holds
    /// @return collateralSold The amount of collateral that the strategy sold
    function withdraw(uint256 amount) external returns (address collateral, uint256 collateralSold);

    /// @dev Gets the interest earned since the last time interest was claimed
    /// @param lastObservedAmount The current strategy debt stored in the vault
    /// @return interestReceived The amount to increment the strategy debt by
    function getNewInterest(uint256 lastObservedAmount) external view returns (uint256 interestReceived);

    /// @dev Returns the current APR of the strategy, expressed in bps
    function getAPR() external view returns (uint256);
}
