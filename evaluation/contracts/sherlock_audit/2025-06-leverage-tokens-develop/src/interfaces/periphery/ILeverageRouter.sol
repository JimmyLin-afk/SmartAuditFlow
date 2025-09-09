// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

// Dependency imports
import {IMorpho} from "@morpho-blue/interfaces/IMorpho.sol";

// Internal imports
import {ILeverageManager} from "../ILeverageManager.sol";
import {ILeverageToken} from "../ILeverageToken.sol";
import {ISwapAdapter} from "./ISwapAdapter.sol";
import {ActionDataV2} from "src/types/DataTypes.sol";

interface ILeverageRouter {
    /// @notice Error thrown when the collateral from the swap + the collateral from the sender is less than the collateral required for the deposit
    /// @param available The collateral from the swap + the collateral from the sender, available for the deposit
    /// @param required The collateral required for the deposit
    error InsufficientCollateralForDeposit(uint256 available, uint256 required);

    /// @notice Error thrown when the cost of a swap exceeds the maximum allowed cost
    /// @param actualCost The actual cost of the swap
    /// @param maxCost The maximum allowed cost of the swap
    error MaxSwapCostExceeded(uint256 actualCost, uint256 maxCost);

    /// @notice Error thrown when the caller is not authorized to execute a function
    error Unauthorized();

    /// @notice Converts an amount of equity to an amount of collateral for a LeverageToken, based on the current
    /// collateral ratio of the LeverageToken
    /// @param token LeverageToken to convert equity to collateral for
    /// @param equityInCollateralAsset Amount of equity to convert to collateral, denominated in the collateral asset of the LeverageToken
    /// @return collateral Amount of collateral that correspond to the equity amount
    function convertEquityToCollateral(ILeverageToken token, uint256 equityInCollateralAsset)
        external
        view
        returns (uint256 collateral);

    /// @notice The LeverageManager contract
    /// @return _leverageManager The LeverageManager contract
    function leverageManager() external view returns (ILeverageManager _leverageManager);

    /// @notice The Morpho core protocol contract
    /// @return _morpho The Morpho core protocol contract
    function morpho() external view returns (IMorpho _morpho);

    /// @notice Previews the deposit function call for an amount of equity and returns all required data
    /// @param token LeverageToken to preview deposit for
    /// @param collateralFromSender The amount of collateral from the sender to deposit
    /// @return previewData Preview data for deposit
    ///         - collateral Total amount of collateral that will be added to the LeverageToken (including collateral from swapping flash loaned debt)
    ///         - debt Amount of debt that will be borrowed
    ///         - shares Amount of shares that will be minted
    ///         - tokenFee Amount of shares that will be charged for the deposit that are given to the LeverageToken
    ///         - treasuryFee Amount of shares that will be charged for the deposit that are given to the treasury
    function previewDeposit(ILeverageToken token, uint256 collateralFromSender)
        external
        view
        returns (ActionDataV2 memory);

    /// @notice The swap adapter contract used to facilitate swaps
    /// @return _swapper The swap adapter contract
    function swapper() external view returns (ISwapAdapter _swapper);

    /// @notice Deposits collateral into a LeverageToken and mints shares to the sender. Any surplus debt received from
    /// the deposit of (collateralFromSender + debt swapped to collateral) is given to the sender.
    /// @param leverageToken LeverageToken to deposit into
    /// @param collateralFromSender Collateral asset amount from the sender to deposit
    /// @param flashLoanAmount Amount of debt to flash loan, which is swapped to collateral and used to deposit into the LeverageToken
    /// @param minShares Minimum number of shares expected to be received by the sender
    /// @param swapContext Swap context to use for the swap (which DEX to use, the route, tick spacing, etc.)
    function deposit(
        ILeverageToken leverageToken,
        uint256 collateralFromSender,
        uint256 flashLoanAmount,
        uint256 minShares,
        ISwapAdapter.SwapContext memory swapContext
    ) external;

    /// @notice Redeems equity of a LeverageToken by repaying debt and burning shares
    /// @param token LeverageToken to redeem
    /// @param equityInCollateralAsset The amount of equity to receive by redeeming LeverageToken. Denominated in the collateral
    ///        asset of the LeverageToken
    /// @param maxShares Maximum shares (LeverageTokens) to redeem
    /// @param maxSwapCostInCollateralAsset The maximum amount of equity to pay for the redeem of the LeverageToken
    ///        to use to help repay the debt flash loan due to the swap of debt to collateral being unfavorable
    /// @param swapContext Swap context to use for the swap (which DEX to use, the route, tick spacing, etc.)
    function redeem(
        ILeverageToken token,
        uint256 equityInCollateralAsset,
        uint256 maxShares,
        uint256 maxSwapCostInCollateralAsset,
        ISwapAdapter.SwapContext memory swapContext
    ) external;
}
