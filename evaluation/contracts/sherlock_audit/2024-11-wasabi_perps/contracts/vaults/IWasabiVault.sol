// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/interfaces/IERC4626.sol";

interface IWasabiVault is IERC4626  {
    error AlreadyMigrated(); // 0xca1c3cbc
    error PrincipalTooHigh(); // 0xd7cdb444
    error InsufficientAvailablePrincipal(); // 0x836ee0c2
    error InsufficientPrincipalRepaid(); // 0xb0f8fc9b
    error CannotClaimNonYieldBearingAsset(address _asset); // 0x4cfaa278
    error EthTransferFailed(); // 0x6d963f88
    error CannotDepositEth(); // 0x2e15428f
    error CallerNotPool(); // 0xe9211597
    error InvalidEthAmount(); // 0x0772327b
    error InvalidAmount(); // 0x2c5211c6
    error NoDustToClean(); // 0x37e34f38
    error AmountExceedsDebt(); // 0x64ddcf37
    error InvalidStrategy(); // 0x4e236e9a
    error InterestFeeTooHigh(); // 0x8e395cd1

    event NativeYieldClaimed(
        address token,
        uint256 amount
    );

    event DepositCapUpdated(
        uint256 newDepositCap
    );

    event InterestFeeBipsUpdated(
        uint256 newInterestFeeBips
    );

    event InterestReceived(
        uint256 interestReceived,
        uint256 interestFeeShares,
        address feeReceiver
    );

    event StrategyDeposit(
        address strategy,
        address collateral,
        uint256 amountDeposited,
        uint256 collateralReceived
    );

    event StrategyWithdraw(
        address strategy,
        address collateral,
        uint256 amountWithdraw,
        uint256 collateralSold
    );

    event StrategyClaim(
        address strategy,
        address collateral,
        uint256 amount
    );

    /// @dev Deposits ETH into the vault (only WETH vault)
    function depositEth(address receiver) external payable returns (uint256);
    
    /// @dev Returns the long or short pool address
    /// @param _long True for long, false for short
    function getPoolAddress(bool _long) external view returns (address);

    /// @dev Returns the current deposit cap
    function getDepositCap() external view returns (uint256);

    /// @dev Called by the pools to borrow assets when a position is opened
    /// @param _amount The amount of assets to borrow
    function borrow(uint256 _amount) external;

    /// @dev Called by the pools to repay assets when a position is closed
    /// @param _totalRepaid The amount of assets being repaid
    /// @param _principal The amount original principal borrowed
    /// @param _isLiquidation Flag to indicate if the repayment is due to liquidation and can cause bad debt
    function recordRepayment(uint256 _totalRepaid, uint256 _principal, bool _isLiquidation) external;

    /// @dev Called by the admin to deposit assets from this vault into a strategy
    /// @param _strategy The address of the strategy account
    /// @param _depositAmount The amount of assets to deposit into the strategy
    function strategyDeposit(address _strategy, uint256 _depositAmount) external;

    /// @dev Called by the admin to withdraw assets from a strategy back to this vault
    /// @param _strategy The address of the strategy
    /// @param _withdrawAmount The amount of assets to withdraw from the strategy
    function strategyWithdraw(address _strategy, uint256 _withdrawAmount) external;

    /// @dev Called by the admin to record interest earned from a strategy, without paying it out yet
    /// @param _strategy The address of the strategy
    function strategyClaim(address _strategy) external;

    /// @dev Called by the admin to donate assets to the vault, which is recorded as interest
    /// @param _amount The amount of assets to donate
    function donate(uint256 _amount) external;

    /// @dev Called by the admin to remove any leftover assets if `totalSupply` is 0 and `totalAssetValue` is > 0
    function cleanDust() external;

    /// @dev Validates that the leverage is within the maximum allowed by the DebtController
    /// @param _downPayment The down payment amount
    /// @param _total The total value of the position in the same currency as the down payment
    /// @notice For shorts, _total is the collateral amount, for longs it is the down payment + principal
    function checkMaxLeverage(uint256 _downPayment, uint256 _total) external view;

    /// @dev Sets the cap on the amount of assets that can be deposited by all users
    /// @param _newDepositCap The new deposit cap
    function setDepositCap(uint256 _newDepositCap) external;

    /// @dev Sets the fee charged on interest in basis points
    /// @param _newInterestFeeBips The new interest fee in basis points
    function setInterestFeeBips(uint256 _newInterestFeeBips) external;
}