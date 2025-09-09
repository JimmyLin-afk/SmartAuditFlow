// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.26;

// External imports
import {Math} from "@openzeppelin/contracts/utils/math/Math.sol";

// Internal imports
import {ILeverageManager} from "src/interfaces/ILeverageManager.sol";
import {ActionData, ExternalAction} from "src/types/DataTypes.sol";
import {PreviewActionTest} from "./PreviewAction.t.sol";

contract RedeemTest is PreviewActionTest {
    function test_redeem_WithFees() public {
        leverageManager.exposed_setLeverageTokenActionFee(leverageToken, ExternalAction.Redeem, 0.05e4); // 5% fee
        _setTreasuryActionFee(ExternalAction.Redeem, 0.05e4); // 5% fee

        _setManagementFee(feeManagerRole, leverageToken, 0.1e4); // 10% management fee
        feeManager.chargeManagementFee(leverageToken);

        // 1:2 exchange rate
        lendingAdapter.mockConvertCollateralToDebtAssetExchangeRate(2e8);

        MockLeverageManagerStateForAction memory beforeState =
            MockLeverageManagerStateForAction({collateral: 200 ether, debt: 100 ether, sharesTotalSupply: 100 ether});

        _prepareLeverageManagerStateForAction(beforeState);

        uint256 equityToRedeem = 10 ether;
        _testRedeem(equityToRedeem, type(uint256).max);
    }

    function test_redeem_WithoutFees() public {
        MockLeverageManagerStateForAction memory beforeState =
            MockLeverageManagerStateForAction({collateral: 200 ether, debt: 100 ether, sharesTotalSupply: 100 ether});

        _prepareLeverageManagerStateForAction(beforeState);

        uint256 equityToRedeem = 10 ether;
        _testRedeem(equityToRedeem, type(uint256).max);
    }

    function test_redeem_ZeroEquity() public {
        MockLeverageManagerStateForAction memory beforeState =
            MockLeverageManagerStateForAction({collateral: 200 ether, debt: 100 ether, sharesTotalSupply: 100 ether});

        _prepareLeverageManagerStateForAction(beforeState);

        _testRedeem(0, type(uint256).max);
    }

    function testFuzz_redeem_RevertIf_SlippageTooHigh(
        uint128 initialCollateral,
        uint128 initialDebtInCollateralAsset,
        uint128 sharesTotalSupply,
        uint128 equityToRedeemInCollateralAsset,
        uint16 tokenFee,
        uint16 treasuryFee
    ) public {
        tokenFee = uint16(bound(tokenFee, 0, MAX_ACTION_FEE));
        treasuryFee = uint16(bound(treasuryFee, 0, MAX_ACTION_FEE));
        initialDebtInCollateralAsset = uint128(bound(initialDebtInCollateralAsset, 0, initialCollateral));
        sharesTotalSupply = uint128(bound(sharesTotalSupply, 1, type(uint128).max));

        leverageManager.exposed_setLeverageTokenActionFee(leverageToken, ExternalAction.Redeem, tokenFee);
        _setTreasuryActionFee(ExternalAction.Redeem, treasuryFee);

        vm.assume(initialCollateral > initialDebtInCollateralAsset);
        vm.assume(equityToRedeemInCollateralAsset > 0);

        // Preview the redeemal
        uint256 expectedShares = leverageManager.previewRedeem(leverageToken, equityToRedeemInCollateralAsset).shares;

        vm.expectRevert(
            abi.encodeWithSelector(ILeverageManager.SlippageTooHigh.selector, expectedShares, expectedShares - 1)
        );
        leverageManager.redeem(leverageToken, equityToRedeemInCollateralAsset, expectedShares - 1);
    }

    function testFuzz_redeem(
        uint128 initialCollateral,
        uint128 initialDebtInCollateralAsset,
        uint128 sharesTotalSupply,
        uint128 equityToRedeemInCollateralAsset,
        uint16 tokenFee,
        uint16 treasuryFee
    ) public {
        tokenFee = uint16(bound(tokenFee, 0, MAX_ACTION_FEE));
        treasuryFee = uint16(bound(treasuryFee, 0, MAX_ACTION_FEE));
        leverageManager.exposed_setLeverageTokenActionFee(leverageToken, ExternalAction.Redeem, tokenFee);
        _setTreasuryActionFee(ExternalAction.Redeem, treasuryFee);

        // Bound debt to be lower than collateral asset and share total supply to be greater than 0 otherwise redeem can not work
        initialDebtInCollateralAsset = uint128(bound(initialDebtInCollateralAsset, 0, initialCollateral));
        sharesTotalSupply = uint128(bound(sharesTotalSupply, 1, type(uint128).max));

        _prepareLeverageManagerStateForAction(
            MockLeverageManagerStateForAction({
                collateral: initialCollateral,
                debt: initialDebtInCollateralAsset,
                sharesTotalSupply: sharesTotalSupply
            })
        );

        // Ensure redeemal amount doesn't exceed available equity
        equityToRedeemInCollateralAsset =
            uint128(bound(equityToRedeemInCollateralAsset, 0, initialCollateral - initialDebtInCollateralAsset));

        _testRedeem(equityToRedeemInCollateralAsset, type(uint256).max);
    }

    function _testRedeem(uint256 equityToRedeemInCollateralAsset, uint256 maxShares) internal {
        // First preview the redeemal
        ActionData memory previewData = leverageManager.previewRedeem(leverageToken, equityToRedeemInCollateralAsset);

        uint256 shareTotalSupplyBefore = leverageToken.totalSupply();

        vm.assume(previewData.shares <= shareTotalSupplyBefore);

        // This needs to be done this way because initial mock state mints total supply to address(1)
        // In order to keep the same total supply we need to burn and mint the same amount of shares
        vm.startPrank(address(leverageManager));
        leverageToken.burn(address(1), previewData.shares);
        leverageToken.mint(address(this), previewData.shares);
        vm.stopPrank();

        // Mint debt tokens to sender and approve leverage manager
        debtToken.mint(address(this), previewData.debt);
        debtToken.approve(address(leverageManager), previewData.debt);

        uint256 collateralBalanceBefore = collateralToken.balanceOf(address(this));
        uint256 debtBalanceBefore = debtToken.balanceOf(address(this));
        uint256 sharesBalanceBefore = leverageToken.balanceOf(address(this));

        // Execute redeemal
        ActionData memory redeemData = leverageManager.redeem(leverageToken, equityToRedeemInCollateralAsset, maxShares);

        // Verify return values match preview
        assertEq(redeemData.collateral, previewData.collateral);
        assertEq(redeemData.debt, previewData.debt);
        assertEq(redeemData.shares, previewData.shares);
        assertEq(redeemData.tokenFee, previewData.tokenFee);
        assertEq(redeemData.treasuryFee, previewData.treasuryFee);

        // Verify token transfers
        assertEq(collateralToken.balanceOf(address(this)) - collateralBalanceBefore, redeemData.collateral);
        assertEq(debtBalanceBefore - debtToken.balanceOf(address(this)), redeemData.debt);

        // Validate leverage token total supply and balance
        assertEq(leverageToken.totalSupply(), shareTotalSupplyBefore - redeemData.shares + redeemData.treasuryFee);
        assertEq(sharesBalanceBefore - leverageToken.balanceOf(address(this)), redeemData.shares);

        // Verify that the treasury received the treasury action fee
        assertEq(leverageToken.balanceOf(treasury), redeemData.treasuryFee);

        // Verify that if any collateral is returned, the amount of shares burned must be non-zero
        if (redeemData.collateral > 0) {
            assertGt(redeemData.shares, 0);
            assertLt(leverageToken.totalSupply(), shareTotalSupplyBefore);
        }

        // Token action fee should be less than or equal to the equity to redeem
        assertLe(redeemData.tokenFee, equityToRedeemInCollateralAsset);

        // Treasury action fee should be less than or equal to the shares burned from the user
        assertLe(redeemData.treasuryFee, redeemData.shares);

        // Verify the collateral ratio is >= the collateral ratio before the redeem
        // We use the comparison collateralBefore * debtAfter >= collateralAfter * debtBefore, which is equivalent to
        // collateralRatioAfter >= collateralRatioBefore to avoid precision loss from division when calculating collateral
        // ratios
        assertGe(lendingAdapter.getCollateral() * debtBalanceBefore, collateralBalanceBefore * lendingAdapter.getDebt());
    }
}
