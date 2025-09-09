// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.26;

// Dependency imports
import {IOracle} from "@morpho-blue/interfaces/IOracle.sol";
import {Math} from "@openzeppelin/contracts/utils/math/Math.sol";

// Internal imports
import {ExternalAction} from "src/types/DataTypes.sol";
import {ILeverageManager} from "src/interfaces/ILeverageManager.sol";
import {ILendingAdapter} from "src/interfaces/ILendingAdapter.sol";
import {MorphoLendingAdapter} from "src/lending/MorphoLendingAdapter.sol";
import {LeverageManagerTest} from "./LeverageManager.t.sol";
import {ActionData, LeverageTokenState} from "src/types/DataTypes.sol";
import {LeverageManagerHarness} from "test/unit/harness/LeverageManagerHarness.t.sol";

contract LeverageManagerRedeemTest is LeverageManagerTest {
    /// @dev In this block price on oracle 3392.292471591441746049801068
    function testFork_redeem_NoFee() public {
        uint256 equityInCollateralAsset = 10 ether;
        uint256 collateralToAdd = 2 * equityInCollateralAsset;
        _mint(user, equityInCollateralAsset, collateralToAdd);

        LeverageTokenState memory stateBefore = getLeverageTokenState();
        uint256 collateralBefore = morphoLendingAdapter.getCollateral();
        assertEq(stateBefore.collateralRatio, 1999999999970521409); // ~2x CR

        uint256 equityToRedeem = 5 ether;
        ActionData memory previewData = leverageManager.previewRedeem(leverageToken, equityToRedeem);
        _redeem(user, equityToRedeem, previewData.debt);

        LeverageTokenState memory stateAfter = getLeverageTokenState();

        // Ensure that collateral ratio is the same (with some rounding error)
        assertGe(stateAfter.collateralRatio, stateBefore.collateralRatio);
        // Verify the collateral ratio is >= the collateral ratio before the redeem
        // We use the comparison collateralBefore * debtAfter >= collateralAfter * debtBefore, which is equivalent to
        // collateralRatioAfter >= collateralRatioBefore to avoid precision loss from division when calculating collateral
        // ratios
        assertGe(morphoLendingAdapter.getCollateral() * stateBefore.debt, collateralBefore * stateAfter.debt);
        assertEq(stateAfter.collateralRatio, 2000000000058957180);
        assertEq(stateAfter.debt, stateBefore.debt - previewData.debt);

        assertEq(WETH.balanceOf(user), previewData.collateral);
    }

    function testFork_redeem_ZeroAmount() public {
        uint256 equityInCollateralAsset = 10 ether;
        uint256 collateralToAdd = 2 * equityInCollateralAsset;
        _mint(user, equityInCollateralAsset, collateralToAdd);

        ActionData memory previewData = leverageManager.previewRedeem(leverageToken, 0);
        _redeem(user, 0, previewData.debt);

        assertEq(previewData.collateral, 0);
        assertEq(previewData.debt, 0);
        assertEq(previewData.shares, 0);
    }

    function testFork_redeem_FullRedeem() public {
        (,, address oracle,,) = morphoLendingAdapter.marketParams();
        vm.mockCall(address(oracle), abi.encodeWithSelector(IOracle.price.selector), abi.encode(4000e24));

        uint256 equityInCollateralAsset = 10 ether;
        uint256 collateralToAdd = 2 * equityInCollateralAsset;
        uint256 shares = _mint(user, equityInCollateralAsset, collateralToAdd);

        uint256 sharesValue = _convertToAssets(shares);
        uint256 debtToRepay = leverageManager.previewRedeem(leverageToken, sharesValue).debt;
        _redeem(user, sharesValue, debtToRepay);

        // Validate that all shares are burned
        assertEq(leverageToken.totalSupply(), 0);

        // Validate that almost all collateral is redeemn, we round down collateral to redeem so dust can be left
        assertGe(morphoLendingAdapter.getCollateral(), 0);
        assertLe(morphoLendingAdapter.getCollateral(), 2);

        // Validate that entire debt is repaid successfully
        assertEq(morphoLendingAdapter.getDebt(), 0);
    }

    function testFork_redeem_MockPrice() public {
        // Mock ETH price to be 4000 USDC
        (,, address oracle,,) = morphoLendingAdapter.marketParams();
        vm.mockCall(address(oracle), abi.encodeWithSelector(IOracle.price.selector), abi.encode(4000e24));

        uint256 equityInCollateralAsset = 10 ether;
        uint256 collateralToAdd = 2 * equityInCollateralAsset;
        _mint(user, equityInCollateralAsset, collateralToAdd);

        LeverageTokenState memory stateBefore = getLeverageTokenState();
        uint256 collateralBefore = morphoLendingAdapter.getCollateral();
        assertEq(stateBefore.collateralRatio, 1999999999950000000); // ~2x CR

        uint256 equityInCollateralAssetBeforeRedeem = morphoLendingAdapter.getEquityInCollateralAsset();
        assertEq(equityInCollateralAssetBeforeRedeem, 9999999999750000000);

        uint256 equityToRedeem = equityInCollateralAssetBeforeRedeem / 2;
        ActionData memory previewData = leverageManager.previewRedeem(leverageToken, equityToRedeem);
        _redeem(user, equityToRedeem, previewData.debt);

        LeverageTokenState memory stateAfter = getLeverageTokenState();
        uint256 equityInCollateralAssetAfterRedeem = morphoLendingAdapter.getEquityInCollateralAsset();

        // Ensure that collateral ratio is the same (with some rounding error)
        assertGe(stateAfter.collateralRatio, stateBefore.collateralRatio);
        // Verify the collateral ratio is >= the collateral ratio before the redeem
        // We use the comparison collateralBefore * debtAfter >= collateralAfter * debtBefore, which is equivalent to
        // collateralRatioAfter >= collateralRatioBefore to avoid precision loss from division when calculating collateral
        // ratios
        assertGe(morphoLendingAdapter.getCollateral() * stateBefore.debt, collateralBefore * stateAfter.debt);
        assertEq(stateAfter.collateralRatio, 2000000000000000000);

        // Ensure that after redeem debt and collateral is 50% of what was initially after mint
        assertEq(stateAfter.debt, 20000_000000); // 2000 USDC
        assertEq(equityInCollateralAssetAfterRedeem, equityInCollateralAsset / 2);

        assertEq(WETH.balanceOf(user), previewData.collateral);
    }

    function testFork_redeem_PriceChangedBetweenRedeems_CollateralRatioDoesNotChange() public {
        uint256 equityInCollateralAsset = 10 ether;
        uint256 collateralToAdd = 2 * equityInCollateralAsset;
        _mint(user, equityInCollateralAsset, collateralToAdd);

        // Mock ETH price to be 4000 USDC
        (,, address oracle,,) = morphoLendingAdapter.marketParams();
        vm.mockCall(address(oracle), abi.encodeWithSelector(IOracle.price.selector), abi.encode(4000e24));

        LeverageTokenState memory stateBefore = getLeverageTokenState();
        uint256 collateralBefore = morphoLendingAdapter.getCollateral();
        assertEq(stateBefore.collateralRatio, 2358287225224640032); // ~2x CR

        uint256 equityToRedeem = 5 ether;
        ActionData memory previewData = leverageManager.previewRedeem(leverageToken, equityToRedeem);
        _redeem(user, equityToRedeem, previewData.debt);

        LeverageTokenState memory stateAfter = getLeverageTokenState();

        // Ensure that collateral ratio is the same (with some rounding error)
        assertGe(stateAfter.collateralRatio, stateBefore.collateralRatio);
        // Verify the collateral ratio is >= the collateral ratio before the redeem
        // We use the comparison collateralBefore * debtAfter >= collateralAfter * debtBefore, which is equivalent to
        // collateralRatioAfter >= collateralRatioBefore to avoid precision loss from division when calculating collateral
        // ratios
        assertGe(morphoLendingAdapter.getCollateral() * stateBefore.debt, collateralBefore * stateAfter.debt);
        assertEq(stateAfter.collateralRatio, 2358287225265780836);

        assertEq(WETH.balanceOf(user), previewData.collateral);
    }

    function testFork_redeem_fullRedeemComparedToPartialRedeemals() public {
        // Mint some assets initially
        uint256 equityInCollateralAsset = 10 ether;
        uint256 collateralToAdd = 2 * equityInCollateralAsset;
        uint256 sharesAfterMint = _mint(user, equityInCollateralAsset, collateralToAdd);

        // Redeem everything
        uint256 sharesValueAfterMint = _convertToAssets(sharesAfterMint);
        ActionData memory previewDataAfterMint = leverageManager.previewRedeem(leverageToken, sharesValueAfterMint);
        _redeem(user, sharesValueAfterMint, previewDataAfterMint.debt);

        // Mint again to create the same scenario
        sharesAfterMint = _mint(user, equityInCollateralAsset, collateralToAdd);

        // Redeem half of it
        uint256 equityToRedeem = equityInCollateralAsset / 2;
        ActionData memory previewDataFirstTime = leverageManager.previewRedeem(leverageToken, equityToRedeem);
        _redeem(user, equityToRedeem, previewDataFirstTime.debt);

        // Redeem the rest
        equityToRedeem = _convertToAssets(leverageToken.balanceOf(user));
        ActionData memory previewDataSecondTime = leverageManager.previewRedeem(leverageToken, equityToRedeem);
        _redeem(user, equityToRedeem, previewDataSecondTime.debt);

        // Validate that in both cases we get the same amount of collateral and debt
        assertEq(previewDataFirstTime.collateral + previewDataSecondTime.collateral, previewDataAfterMint.collateral);
        assertEq(previewDataFirstTime.debt + previewDataSecondTime.debt, previewDataAfterMint.debt);

        // Validate that collateral token is properly transferred to user
        assertEq(WETH.balanceOf(user), previewDataFirstTime.collateral + previewDataSecondTime.collateral);
        assertLe(previewDataAfterMint.collateral, 2 * equityInCollateralAsset);
    }

    function testFork_redeem_withFee() public {
        uint256 treasuryActionFee = 10_00; // 10%
        leverageManager.setTreasuryActionFee(ExternalAction.Redeem, treasuryActionFee); // 10%

        uint256 tokenActionFee = 10_00; // 10%
        leverageToken =
            _createNewLeverageToken(BASE_RATIO, 2 * BASE_RATIO, 3 * BASE_RATIO, tokenActionFee, tokenActionFee);

        uint256 managementFee = 10_00; // 10%
        leverageManager.setManagementFee(leverageToken, managementFee);

        morphoLendingAdapter =
            MorphoLendingAdapter(address(leverageManager.getLeverageTokenLendingAdapter(leverageToken)));

        uint256 equityInCollateralAsset = 10 ether;
        uint256 collateralToAdd = 2 * equityInCollateralAsset;
        _mint(user, equityInCollateralAsset, collateralToAdd);

        // 10% of equity goes to share dilution (token action fee), so 9 ether shares are minted instead of 10 ether
        assertEq(leverageToken.balanceOf(user), 9 ether);
        assertEq(leverageToken.totalSupply(), 9 ether);

        uint256 equityInCollateralAssetAfterMint = morphoLendingAdapter.getEquityInCollateralAsset();

        // Redeem 50% of equity
        uint256 equityToRedeem = equityInCollateralAssetAfterMint / 2;
        ActionData memory previewData = leverageManager.previewRedeem(leverageToken, equityToRedeem);
        _redeem(user, equityToRedeem, previewData.debt);

        // Half of the equity is redeemn, so half of the total supply is burned + an additional 10% for the token action fee
        // (9/2)*1.1 = 4.95 burned, 4.05 remaining
        assertEq(leverageToken.totalSupply(), 9 ether - 4.95 ether);
        // 10% of the total shares burned are minted to the treasury to cover the treasury fee. 4.95 * 0.1 = 0.495
        assertEq(leverageToken.balanceOf(treasury), 0.495 ether);
        // The user's shares are decreased by the burned shares including token and treasury fees. 9 - (4.5 * 1.1 * 1.1) = 9 - 5.445 = 3.555
        assertEq(leverageToken.balanceOf(user), 3.555 ether);

        assertEq(WETH.balanceOf(user), previewData.collateral); // User receives the collateral asset

        // One year passes
        skip(SECONDS_ONE_YEAR);

        // To redeem the same amount of equity we need to burn more shares because of the share dilution from the
        // management fee and morpho borrow interest
        previewData = leverageManager.previewRedeem(leverageToken, equityToRedeem);
        assertEq(previewData.shares, 5.460965008456074474 ether);

        uint256 userTotalShareValue = Math.mulDiv(
            // An additional 10% of shares are required to be burned from the user to cover the treasury action fee,
            // and an additional 10% of shares are required to be burned from the user to cover the token action fee
            // 1.1 * 1.1 = 1.21
            Math.mulDiv(leverageToken.balanceOf(user), 1e18, 1.21e18, Math.Rounding.Floor),
            morphoLendingAdapter.getEquityInCollateralAsset(),
            leverageManager.getFeeAdjustedTotalSupply(leverageToken),
            Math.Rounding.Floor
        );
        // The share value is less than half of the initial equity minted due to the share dilution from the fees,
        // and morpho borrow interest
        assertEq(userTotalShareValue, 3.254919226259702618 ether);

        previewData = leverageManager.previewRedeem(leverageToken, userTotalShareValue);
        uint256 expectedTreasuryActionFee = Math.mulDiv(
            LeverageManagerHarness(address(leverageManager)).exposed_convertToShares(
                leverageToken,
                // 10% additional shares are burned for the token action fee
                Math.mulDiv(userTotalShareValue, 1.1e18, 1e18, Math.Rounding.Ceil),
                ExternalAction.Redeem
            ),
            // 10% of the shares burned are minted to the treasury to cover the treasury action fee
            0.1e18,
            1e18,
            Math.Rounding.Ceil
        );

        uint256 userWethBalanceBeforeRedeem = WETH.balanceOf(user);

        // Redeem the share equity
        _redeem(user, userTotalShareValue, previewData.debt);

        assertEq(leverageToken.balanceOf(user), 0);
        // Initial balance + management fee + treasury action fee
        assertEq(leverageToken.balanceOf(treasury), 0.495 ether + 0.405 ether + expectedTreasuryActionFee);
        assertEq(leverageToken.totalSupply(), leverageToken.balanceOf(treasury));

        assertEq(WETH.balanceOf(user), userWethBalanceBeforeRedeem + previewData.collateral); // User receives the collateral asset
    }
}
