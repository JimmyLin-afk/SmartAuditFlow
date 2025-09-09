// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

// Dependency imports
import {Math} from "@openzeppelin/contracts/utils/math/Math.sol";
import {IMorpho} from "@morpho-blue/interfaces/IMorpho.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

// Internal imports
import {ILendingAdapter} from "../interfaces/ILendingAdapter.sol";
import {ILeverageManager} from "../interfaces/ILeverageManager.sol";
import {ILeverageToken} from "../interfaces/ILeverageToken.sol";
import {ISwapAdapter} from "../interfaces/periphery/ISwapAdapter.sol";
import {ILeverageRouter} from "../interfaces/periphery/ILeverageRouter.sol";
import {ActionData, ActionDataV2, ExternalAction} from "../types/DataTypes.sol";

/**
 * @dev The LeverageRouter contract is an immutable periphery contract that facilitates the use of flash loans and a swap adapter
 * to deposit and redeem equity from LeverageTokens.
 *
 * The high-level deposit flow is as follows:
 *   1. The sender calls `deposit` with the amount of collateral from the sender to deposit, the amount of debt to flash loan
 *      (which will be swapped to collateral), the minimum amount of shares to receive, and the swap context
 *   2. The LeverageRouter will flash loan the debt asset amount and swap it to collateral
 *   3. The LeverageRouter will use the collateral from the swapped debt and the collateral from the sender for the deposit
 *      into the LeverageToken, receiving LeverageToken shares and debt in return
 *   4. The LeverageRouter will use the debt received from the deposit to repay the flash loan
 *   6. The LeverageRouter will transfer the LeverageToken shares and any surplus debt assets to the sender
 *
 * The high-level redeem flow is the same as the deposit flow, but in reverse.
 */
contract LeverageRouter is ILeverageRouter {
    /// @notice Deposit related parameters to pass to the Morpho flash loan callback handler for deposits
    struct DepositParams {
        // Address of the sender of the deposit
        address sender;
        // LeverageToken to deposit into
        ILeverageToken leverageToken;
        // Amount of collateral from the sender to deposit
        uint256 collateralFromSender;
        // Minimum amount of shares (LeverageTokens) to receive
        uint256 minShares;
        // Swap context for the swap of flash loaned debt to collateral
        ISwapAdapter.SwapContext swapContext;
    }

    /// @notice Redeem related parameters to pass to the Morpho flash loan callback handler for redeems
    struct RedeemParams {
        // LeverageToken to redeem from
        ILeverageToken token;
        // Amount of equity to receive by redeeming, denominated in the collateral asset
        uint256 equityInCollateralAsset;
        // Amount of LeverageToken shares to redeem for the equity
        uint256 shares;
        // Maximum amount of shares (LeverageTokens) to be burned during the redeem
        uint256 maxShares;
        // Maximum cost to the sender for the swap of debt to collateral during the redeem to repay the flash loan,
        // denominated in the collateral asset. This cost is applied to the equity being received
        uint256 maxSwapCostInCollateralAsset;
        // Address of the sender of the redeem, whose shares will be burned and the equity will be transferred to
        address sender;
        // Swap context for the debt swap
        ISwapAdapter.SwapContext swapContext;
    }

    /// @notice Morpho flash loan callback data to pass to the Morpho flash loan callback handler
    struct MorphoCallbackData {
        ExternalAction action;
        bytes data;
    }

    /// @inheritdoc ILeverageRouter
    ILeverageManager public immutable leverageManager;

    /// @inheritdoc ILeverageRouter
    IMorpho public immutable morpho;

    /// @inheritdoc ILeverageRouter
    ISwapAdapter public immutable swapper;

    /// @notice Creates a new LeverageRouter
    /// @param _leverageManager The LeverageManager contract
    /// @param _morpho The Morpho core protocol contract
    /// @param _swapper The Swapper contract
    constructor(ILeverageManager _leverageManager, IMorpho _morpho, ISwapAdapter _swapper) {
        leverageManager = _leverageManager;
        morpho = _morpho;
        swapper = _swapper;
    }

    /// @inheritdoc ILeverageRouter
    function convertEquityToCollateral(ILeverageToken token, uint256 equityInCollateralAsset)
        public
        view
        returns (uint256 collateral)
    {
        uint256 collateralRatio = leverageManager.getLeverageTokenState(token).collateralRatio;
        ILendingAdapter lendingAdapter = leverageManager.getLeverageTokenLendingAdapter(token);
        uint256 baseRatio = leverageManager.BASE_RATIO();

        if (lendingAdapter.getCollateral() == 0 && lendingAdapter.getDebt() == 0) {
            uint256 initialCollateralRatio = leverageManager.getLeverageTokenInitialCollateralRatio(token);
            collateral = Math.mulDiv(
                equityInCollateralAsset, initialCollateralRatio, initialCollateralRatio - baseRatio, Math.Rounding.Ceil
            );
        } else if (collateralRatio == type(uint256).max) {
            collateral = equityInCollateralAsset;
        } else {
            collateral =
                Math.mulDiv(equityInCollateralAsset, collateralRatio, collateralRatio - baseRatio, Math.Rounding.Ceil);
        }

        return collateral;
    }

    /// @inheritdoc ILeverageRouter
    function previewDeposit(ILeverageToken token, uint256 collateralFromSender)
        external
        view
        returns (ActionDataV2 memory previewData)
    {
        uint256 collateral = convertEquityToCollateral(token, collateralFromSender);
        return leverageManager.previewDeposit(token, collateral);
    }

    /// @inheritdoc ILeverageRouter
    function deposit(
        ILeverageToken leverageToken,
        uint256 collateralFromSender,
        uint256 flashLoanAmount,
        uint256 minShares,
        ISwapAdapter.SwapContext memory swapContext
    ) external {
        bytes memory depositData = abi.encode(
            DepositParams({
                sender: msg.sender,
                leverageToken: leverageToken,
                collateralFromSender: collateralFromSender,
                minShares: minShares,
                swapContext: swapContext
            })
        );

        morpho.flashLoan(
            address(leverageManager.getLeverageTokenDebtAsset(leverageToken)),
            flashLoanAmount,
            abi.encode(MorphoCallbackData({action: ExternalAction.Mint, data: depositData}))
        );
    }

    /// @inheritdoc ILeverageRouter
    function redeem(
        ILeverageToken token,
        uint256 equityInCollateralAsset,
        uint256 maxShares,
        uint256 maxSwapCostInCollateralAsset,
        ISwapAdapter.SwapContext memory swapContext
    ) external {
        ActionData memory actionData = leverageManager.previewRedeem(token, equityInCollateralAsset);

        bytes memory redeemData = abi.encode(
            RedeemParams({
                token: token,
                equityInCollateralAsset: equityInCollateralAsset,
                shares: actionData.shares,
                maxShares: maxShares,
                maxSwapCostInCollateralAsset: maxSwapCostInCollateralAsset,
                sender: msg.sender,
                swapContext: swapContext
            })
        );

        // Flash loan the debt asset required to repay the flash loan, and pass the required data to the Morpho flash loan callback handler for the redeem.
        morpho.flashLoan(
            address(leverageManager.getLeverageTokenDebtAsset(token)),
            actionData.debt,
            abi.encode(MorphoCallbackData({action: ExternalAction.Redeem, data: redeemData}))
        );
    }

    /// @notice Morpho flash loan callback function
    /// @param loanAmount Amount of asset flash loaned
    /// @param data Encoded data passed to `morpho.flashLoan`
    function onMorphoFlashLoan(uint256 loanAmount, bytes calldata data) external {
        if (msg.sender != address(morpho)) revert Unauthorized();

        MorphoCallbackData memory callbackData = abi.decode(data, (MorphoCallbackData));

        if (callbackData.action == ExternalAction.Mint) {
            DepositParams memory params = abi.decode(callbackData.data, (DepositParams));
            _depositAndRepayMorphoFlashLoan(params, loanAmount);
        } else if (callbackData.action == ExternalAction.Redeem) {
            RedeemParams memory params = abi.decode(callbackData.data, (RedeemParams));
            _redeemAndRepayMorphoFlashLoan(params, loanAmount);
        }
    }

    /// @notice Executes the deposit into a LeverageToken by flash loaning the debt asset, swapping it to collateral,
    /// depositing into the LeverageToken with the sender's collateral, and using the resulting debt to repay the flash loan.
    /// Any surplus debt assets after repaying the flash loan are given to the sender.
    /// @param params Params for the deposit into a LeverageToken
    /// @param debtLoan Amount of debt asset flash loaned
    function _depositAndRepayMorphoFlashLoan(DepositParams memory params, uint256 debtLoan) internal {
        IERC20 collateralAsset = leverageManager.getLeverageTokenCollateralAsset(params.leverageToken);
        IERC20 debtAsset = leverageManager.getLeverageTokenDebtAsset(params.leverageToken);

        // Transfer the collateral from the sender for the deposit
        // slither-disable-next-line arbitrary-send-erc20
        SafeERC20.safeTransferFrom(collateralAsset, params.sender, address(this), params.collateralFromSender);

        // Swap the debt asset received from the flash loan to the collateral asset, used to deposit
        SafeERC20.forceApprove(debtAsset, address(swapper), debtLoan);

        uint256 collateralFromSwap = swapper.swapExactInput(
            debtAsset,
            debtLoan,
            0, // Set to zero because collateral from the sender is used to help with the deposit
            params.swapContext
        );

        uint256 totalCollateral = collateralFromSwap + params.collateralFromSender;

        // Use the collateral from the swap and the collateral from the sender for the deposit into the LeverageToken
        SafeERC20.forceApprove(collateralAsset, address(leverageManager), totalCollateral);

        ActionDataV2 memory actionData =
            leverageManager.deposit(params.leverageToken, totalCollateral, params.minShares);

        // Transfer any surplus debt assets to the sender
        if (debtLoan < actionData.debt) {
            SafeERC20.safeTransfer(debtAsset, params.sender, actionData.debt - debtLoan);
        }

        // Transfer shares received from the deposit to the deposit sender
        SafeERC20.safeTransfer(params.leverageToken, params.sender, actionData.shares);

        // Approve morpho to transfer debt assets to repay the flash loan
        // Note: if insufficient debt is available to repay the flash loan, the transaction will revert when Morpho
        // attempts to transfer the debt assets to repay the flash loan
        SafeERC20.forceApprove(debtAsset, address(morpho), debtLoan);
    }

    /// @notice Executes redeem on a LeverageToken to receive equity and the swap of collateral assets to the debt asset
    /// to repay the flash loan from Morpho
    /// @param params Params for the redeem of equity from a LeverageToken
    /// @param debtLoanAmount Amount of debt asset flash loaned
    function _redeemAndRepayMorphoFlashLoan(RedeemParams memory params, uint256 debtLoanAmount) internal {
        IERC20 collateralAsset = leverageManager.getLeverageTokenCollateralAsset(params.token);
        IERC20 debtAsset = leverageManager.getLeverageTokenDebtAsset(params.token);

        // Transfer the shares from the sender
        // slither-disable-next-line arbitrary-send-erc20
        SafeERC20.safeTransferFrom(params.token, params.sender, address(this), params.shares);

        // Redeem the equity from the leverage token
        SafeERC20.forceApprove(debtAsset, address(leverageManager), debtLoanAmount);
        uint256 collateralWithdrawn =
            leverageManager.redeem(params.token, params.equityInCollateralAsset, params.maxShares).collateral;

        // Swap the collateral asset received from the redeem to the debt asset, used to repay the flash loan
        SafeERC20.forceApprove(collateralAsset, address(swapper), collateralWithdrawn);
        uint256 collateralAmountSwapped =
            swapper.swapExactOutput(collateralAsset, debtLoanAmount, collateralWithdrawn, params.swapContext);

        // Check if the amount of collateral swapped to repay the flash loan is greater than the allowed cost
        uint256 remainingCollateral = collateralWithdrawn - collateralAmountSwapped;
        if (remainingCollateral < params.equityInCollateralAsset - params.maxSwapCostInCollateralAsset) {
            revert MaxSwapCostExceeded(
                params.equityInCollateralAsset - remainingCollateral, params.maxSwapCostInCollateralAsset
            );
        } else if (remainingCollateral > 0) {
            SafeERC20.safeTransfer(collateralAsset, params.sender, remainingCollateral);
        }

        // Approve morpho to transfer assets to repay the flash loan
        SafeERC20.forceApprove(debtAsset, address(morpho), debtLoanAmount);
    }
}
