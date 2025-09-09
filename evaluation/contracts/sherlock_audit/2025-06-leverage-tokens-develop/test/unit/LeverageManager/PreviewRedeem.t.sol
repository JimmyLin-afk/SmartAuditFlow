// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.26;

// Dependency imports
import {Math} from "@openzeppelin/contracts/utils/math/Math.sol";

// Internal imports
import {ActionData, ExternalAction} from "src/types/DataTypes.sol";
import {PreviewActionTest} from "./PreviewAction.t.sol";

contract PreviewRedeemTest is PreviewActionTest {
    function testFuzz_previewRedeem_MatchesPreviewAction(
        uint128 initialCollateral,
        uint128 initialDebtInCollateralAsset,
        uint128 sharesTotalSupply,
        uint128 equityToRedeemInCollateralAsset,
        uint16 treasuryFee
    ) public {
        initialDebtInCollateralAsset = uint128(bound(initialDebtInCollateralAsset, 0, initialCollateral));

        treasuryFee = uint16(bound(treasuryFee, 0, MAX_ACTION_FEE));
        _setTreasuryActionFee(ExternalAction.Redeem, treasuryFee);

        equityToRedeemInCollateralAsset =
            uint128(bound(equityToRedeemInCollateralAsset, 0, initialCollateral - initialDebtInCollateralAsset));

        _prepareLeverageManagerStateForAction(
            MockLeverageManagerStateForAction({
                collateral: initialCollateral,
                debt: initialDebtInCollateralAsset,
                sharesTotalSupply: sharesTotalSupply
            })
        );

        ActionData memory previewActionData =
            leverageManager.exposed_previewAction(leverageToken, equityToRedeemInCollateralAsset, ExternalAction.Redeem);

        ActionData memory actualPreviewData =
            leverageManager.previewRedeem(leverageToken, equityToRedeemInCollateralAsset);

        assertEq(actualPreviewData.collateral, previewActionData.collateral, "Collateral to remove mismatch");
        assertEq(actualPreviewData.debt, previewActionData.debt, "Debt to repay mismatch");
        assertEq(actualPreviewData.shares, previewActionData.shares, "Shares after fee mismatch");
        assertEq(actualPreviewData.tokenFee, previewActionData.tokenFee, "Shares fee mismatch");
        assertEq(actualPreviewData.treasuryFee, previewActionData.treasuryFee, "Treasury fee mismatch");
        assertEq(actualPreviewData.equity, previewActionData.equity, "Equity mismatch");
    }

    function test_previewRedeem_CollateralLessThanTreasuryFee() public {
        uint256 equityToPreview = 3;

        uint256 treasuryFee = 0.8e4; // 80%
        _setTreasuryActionFee(ExternalAction.Redeem, treasuryFee);

        uint256 initialCollateral = 330944644884850719377224828425;
        uint256 initialDebt = 135;
        uint256 sharesTotalSupply = 147701522956517018969156799895542881615;
        _prepareLeverageManagerStateForAction(
            MockLeverageManagerStateForAction({
                collateral: initialCollateral,
                debt: initialDebt,
                sharesTotalSupply: sharesTotalSupply
            })
        );

        uint256 expectedSharesBeforeFees =
            leverageManager.exposed_convertToShares(leverageToken, equityToPreview, ExternalAction.Redeem);
        assertEq(expectedSharesBeforeFees, 1338908412);

        uint256 collateralToRemove = initialCollateral * expectedSharesBeforeFees / sharesTotalSupply;
        assertEq(collateralToRemove, 3);

        // The treasury fee is rounded up, so it's possible for it to be greater than the calculated collateral to be removed
        // which is calculated by rounding down
        uint256 expectedTreasuryFeeBeforeAdjustment = Math.mulDiv(equityToPreview, treasuryFee, 1e4, Math.Rounding.Ceil);
        assertEq(expectedTreasuryFeeBeforeAdjustment, 3);

        ActionData memory previewData = leverageManager.previewRedeem(leverageToken, equityToPreview);

        assertEq(previewData.collateral, collateralToRemove);
        assertEq(
            previewData.treasuryFee, Math.mulDiv(expectedSharesBeforeFees, treasuryFee, MAX_BPS, Math.Rounding.Ceil)
        );
    }
}
