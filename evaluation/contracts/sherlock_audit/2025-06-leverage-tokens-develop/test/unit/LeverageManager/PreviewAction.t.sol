// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.26;

// Dependency imports
import {Math} from "@openzeppelin/contracts/utils/math/Math.sol";

// Internal imports
import {ExternalAction} from "src/types/DataTypes.sol";
import {ILendingAdapter} from "src/interfaces/ILendingAdapter.sol";
import {IRebalanceAdapter} from "src/interfaces/IRebalanceAdapter.sol";
import {ActionData, LeverageTokenConfig, LeverageTokenState} from "src/types/DataTypes.sol";
import {LeverageManagerTest} from "../LeverageManager/LeverageManager.t.sol";
import {ILeverageManager} from "src/interfaces/ILeverageManager.sol";

contract PreviewActionTest is LeverageManagerTest {
    function setUp() public override {
        super.setUp();

        _createNewLeverageToken(
            manager,
            2e18,
            LeverageTokenConfig({
                lendingAdapter: ILendingAdapter(address(lendingAdapter)),
                rebalanceAdapter: IRebalanceAdapter(address(rebalanceAdapter)),
                mintTokenFee: 0,
                redeemTokenFee: 0
            }),
            address(collateralToken),
            address(debtToken),
            "dummy name",
            "dummy symbol"
        );
    }

    function test_previewAction_WithFee() public {
        _setManagementFee(feeManagerRole, leverageToken, 0.1e4); // 10% management fee
        feeManager.chargeManagementFee(leverageToken);

        _setTreasuryActionFee(feeManagerRole, ExternalAction.Mint, 0.1e4); // 10% fee
        _setTreasuryActionFee(feeManagerRole, ExternalAction.Redeem, 0.1e4); // 10% fee

        leverageManager.exposed_setLeverageTokenActionFee(leverageToken, ExternalAction.Mint, 0.05e4); // 5% fee
        leverageManager.exposed_setLeverageTokenActionFee(leverageToken, ExternalAction.Redeem, 0.05e4); // 5% fee

        // 1:2 exchange rate
        lendingAdapter.mockConvertCollateralToDebtAssetExchangeRate(2e8);

        MockLeverageManagerStateForAction memory beforeState =
            MockLeverageManagerStateForAction({collateral: 100 ether, debt: 100 ether, sharesTotalSupply: 100 ether});

        _prepareLeverageManagerStateForAction(beforeState);

        uint256 equity = 10 ether;
        ActionData memory previewData =
            leverageManager.exposed_previewAction(leverageToken, equity, ExternalAction.Mint);

        assertEq(previewData.collateral, 20 ether);
        assertEq(previewData.debt, 20 ether);
        // 5% fee on equity (1 ether shares), 10% fee on shares (19 ether shares * 0.1 = 1.9 ether shares)
        // 20 ether shares - 1 ether shares - 1.9 ether shares = 17.1 ether shares
        assertEq(previewData.shares, 17.1 ether);
        // 5% fee on equity in collateral asset
        assertEq(previewData.tokenFee, 0.5 ether);
        // 10% fee on shares
        assertEq(previewData.treasuryFee, 1.9 ether);

        previewData = leverageManager.exposed_previewAction(leverageToken, equity, ExternalAction.Redeem);

        assertEq(previewData.collateral, 20 ether);
        assertEq(previewData.debt, 20 ether);
        // 5% fee on equity (1 ether shares), 10% fee on shares (21 ether shares * 0.1 = 2.1 ether shares)
        assertEq(previewData.shares, 23.1 ether);
        // 5% fee on equity in collateral asset
        assertEq(previewData.tokenFee, 0.5 ether);
        // 10% fee on shares
        assertEq(previewData.treasuryFee, 2.1 ether);

        skip(SECONDS_ONE_YEAR);

        previewData = leverageManager.exposed_previewAction(leverageToken, equity, ExternalAction.Mint);

        // 10% management fee affects the shares but everything else is the same
        assertEq(previewData.collateral, 20 ether);
        assertEq(previewData.debt, 20 ether);
        assertEq(previewData.shares, 18.81 ether); // Shares minted are increased by 10% due to management fee diluting share value
        assertEq(previewData.tokenFee, 0.5 ether); // 5% fee on equity in collateral asset
        assertEq(previewData.treasuryFee, 2.09 ether);

        previewData = leverageManager.exposed_previewAction(leverageToken, equity, ExternalAction.Redeem);

        // 10% management fee affects the shares but everything else is the same
        assertEq(previewData.collateral, 20 ether);
        assertEq(previewData.debt, 20 ether);
        assertEq(previewData.shares, 25.41 ether); // Shares burned are increased by 10% due to management fee diluting share value
        assertEq(previewData.tokenFee, 0.5 ether); // 5% fee on equity in collateral asset
        assertEq(previewData.treasuryFee, 2.31 ether);
    }

    function test_previewAction_WithFee_ZeroSharesForEquity() public {
        leverageManager.exposed_setLeverageTokenActionFee(leverageToken, ExternalAction.Mint, 0.05e4); // 5% fee
        leverageManager.exposed_setLeverageTokenActionFee(leverageToken, ExternalAction.Redeem, 0.05e4); // 5% fee

        // 1:2 exchange rate
        lendingAdapter.mockConvertCollateralToDebtAssetExchangeRate(2e8);

        MockLeverageManagerStateForAction memory beforeState =
            MockLeverageManagerStateForAction({collateral: 100 ether, debt: 100 ether, sharesTotalSupply: 10 ether});

        _prepareLeverageManagerStateForAction(beforeState);

        // 0 shares can be minted for 1 wei of equity
        uint256 equity = 1;
        ActionData memory previewData =
            leverageManager.exposed_previewAction(leverageToken, equity, ExternalAction.Mint);

        assertEq(previewData.collateral, 0);
        assertEq(previewData.debt, 0);
        assertEq(previewData.shares, 0);
        assertEq(previewData.tokenFee, 1);
        assertEq(previewData.treasuryFee, 0);

        // 1 share can be burned for 1 wei of equity because of shares rounding up for redeems
        previewData = leverageManager.exposed_previewAction(leverageToken, equity, ExternalAction.Redeem);
        assertEq(previewData.collateral, 10);
        assertEq(previewData.debt, 10);
        assertEq(previewData.shares, 1);
        assertEq(previewData.tokenFee, 1);
        assertEq(previewData.treasuryFee, 0);
    }

    function test_previewAction_WithoutFee() public {
        MockLeverageManagerStateForAction memory beforeState =
            MockLeverageManagerStateForAction({collateral: 100 ether, debt: 50 ether, sharesTotalSupply: 100 ether});

        _prepareLeverageManagerStateForAction(beforeState);

        uint256 equityToAdd = 10 ether;
        ActionData memory previewData =
            leverageManager.exposed_previewAction(leverageToken, equityToAdd, ExternalAction.Mint);

        assertEq(previewData.collateral, 20 ether);
        assertEq(previewData.debt, 10 ether);
        assertEq(previewData.shares, 20 ether);
        assertEq(previewData.tokenFee, 0);
        assertEq(previewData.treasuryFee, 0);

        previewData = leverageManager.exposed_previewAction(leverageToken, equityToAdd, ExternalAction.Redeem);

        assertEq(previewData.collateral, 20 ether);
        assertEq(previewData.debt, 10 ether);
        assertEq(previewData.shares, 20 ether);
        assertEq(previewData.tokenFee, 0);
        assertEq(previewData.treasuryFee, 0);
    }

    function test_previewAction_ZeroEquity() public view {
        uint256 equity = 0;
        ActionData memory previewData =
            leverageManager.exposed_previewAction(leverageToken, equity, ExternalAction.Mint);

        assertEq(previewData.collateral, 0);
        assertEq(previewData.debt, 0);
        assertEq(previewData.shares, 0);
        assertEq(previewData.tokenFee, 0);
        assertEq(previewData.treasuryFee, 0);
        previewData = leverageManager.exposed_previewAction(leverageToken, equity, ExternalAction.Redeem);

        assertEq(previewData.collateral, 0);
        assertEq(previewData.debt, 0);
        assertEq(previewData.shares, 0);
        assertEq(previewData.tokenFee, 0);
        assertEq(previewData.treasuryFee, 0);
    }

    function testFuzz_previewAction_MintZeroSharesTotalSupply(uint128 initialCollateral, uint128 initialDebt) public {
        initialDebt = initialCollateral == 0 ? 0 : uint128(bound(initialDebt, 0, initialCollateral - 1));

        MockLeverageManagerStateForAction memory beforeState =
            MockLeverageManagerStateForAction({collateral: initialCollateral, debt: initialDebt, sharesTotalSupply: 0});

        _prepareLeverageManagerStateForAction(beforeState);

        uint256 equity = 1 ether;

        ActionData memory previewData =
            leverageManager.exposed_previewAction(leverageToken, equity, ExternalAction.Mint);

        // Follows 2x target ratio
        assertEq(previewData.collateral, 2 ether);
        assertEq(previewData.debt, 1 ether);

        uint256 expectedShares = leverageManager.exposed_convertToShares(leverageToken, equity, ExternalAction.Mint);
        assertEq(previewData.shares, expectedShares);
        assertEq(previewData.tokenFee, 0);
        assertEq(previewData.treasuryFee, 0);

        expectedShares = leverageManager.exposed_convertToShares(leverageToken, equity, ExternalAction.Redeem);
        previewData = leverageManager.exposed_previewAction(leverageToken, equity, ExternalAction.Redeem);
        assertEq(previewData.collateral, 2 ether);
        assertEq(previewData.debt, 1 ether);
        assertEq(previewData.shares, expectedShares);
        assertEq(previewData.tokenFee, 0);
        assertEq(previewData.treasuryFee, 0);
    }

    function testFuzz_previewAction(
        uint128 initialCollateral,
        uint128 initialDebtInCollateralAsset,
        uint128 sharesTotalSupply,
        uint128 equityInCollateralAsset,
        uint16 fee,
        uint16 managementFee,
        uint8 actionNum
    ) public {
        ExternalAction action = ExternalAction(actionNum % 2);
        fee = uint16(bound(fee, 0, MAX_ACTION_FEE)); // 0% to 99.99% fee
        leverageManager.exposed_setLeverageTokenActionFee(leverageToken, action, fee);

        managementFee = uint16(bound(managementFee, 0, MAX_MANAGEMENT_FEE)); // 0% to 100% management fee
        _setManagementFee(feeManagerRole, leverageToken, managementFee);

        initialDebtInCollateralAsset = uint128(bound(initialDebtInCollateralAsset, 0, initialCollateral));

        if (initialCollateral == 0 && initialDebtInCollateralAsset == 0) {
            sharesTotalSupply = 0;
        } else {
            sharesTotalSupply = uint128(bound(sharesTotalSupply, 1, type(uint128).max));
        }

        _prepareLeverageManagerStateForAction(
            MockLeverageManagerStateForAction({
                collateral: initialCollateral,
                debt: initialDebtInCollateralAsset, // 1:1 exchange rate for this test
                sharesTotalSupply: sharesTotalSupply
            })
        );

        // Ensure the collateral being added does not result in overflows due to mocked value sizes
        if (action == ExternalAction.Mint) {
            equityInCollateralAsset = uint128(bound(equityInCollateralAsset, 1, type(uint96).max));
        } else {
            equityInCollateralAsset =
                uint128(bound(equityInCollateralAsset, 0, _convertToAssets(sharesTotalSupply, action)));
        }

        // Get state prior to action
        LeverageTokenState memory prevState = leverageManager.getLeverageTokenState(leverageToken);

        ActionData memory previewData =
            leverageManager.exposed_previewAction(leverageToken, equityInCollateralAsset, action);

        // Calculate state after action
        (, uint256 newDebt, uint256 newCollateralRatio) = _getNewLeverageTokenState(
            initialCollateral, initialDebtInCollateralAsset, previewData.collateral, previewData.debt, action
        );

        {
            (uint256 equityForSharesAfterFees, uint256 tokenFee) =
                leverageManager.exposed_computeTokenFee(leverageToken, equityInCollateralAsset, action);

            (uint256 collateralForLeverageToken, uint256 debtForLeverageToken) = leverageManager
                .exposed_computeCollateralAndDebtForAction(leverageToken, equityInCollateralAsset, action);
            uint256 shares = leverageManager.exposed_convertToShares(leverageToken, equityForSharesAfterFees, action);
            uint256 treasuryFee = leverageManager.exposed_computeTreasuryFee(action, shares);

            // Validate if shares, collateral, debt, and fees are properly calculated and returned
            assertEq(previewData.shares, action == ExternalAction.Mint ? shares - treasuryFee : shares + treasuryFee);
            assertEq(previewData.collateral, collateralForLeverageToken);
            assertEq(previewData.debt, debtForLeverageToken);
            assertEq(previewData.tokenFee, tokenFee);
            assertEq(previewData.treasuryFee, treasuryFee);
        }

        // If full redeem is done then the collateral ratio should be max
        if (_isFullRedeem(initialDebtInCollateralAsset, previewData.debt, action)) {
            assertEq(newCollateralRatio, type(uint256).max);
            return;
        }

        // If leverage token was initially empty then action should be done by respecting the target ratio
        if (_isLeverageTokenEmpty(initialCollateral)) {
            assertEq(newCollateralRatio, 2 * _BASE_RATIO());
            return;
        }

        // If initially leverage token had something in collateral but no debt ratio should change for the better
        if (initialDebtInCollateralAsset == 0) {
            assertLe(newCollateralRatio, prevState.collateralRatio);
            return;
        }

        // Otherwise, the action should be done by respecting the current collateral ratio
        // There is some tolerance on collateral ratio due to rounding depending on debt size
        // It is important to calculate tolerance with smaller debt (for mint before action for redeem after action)

        uint256 respectiveDebt = action == ExternalAction.Mint ? initialDebtInCollateralAsset : newDebt;
        uint256 from = action == ExternalAction.Mint ? newCollateralRatio : prevState.collateralRatio;
        uint256 to = action == ExternalAction.Mint ? prevState.collateralRatio : newCollateralRatio;

        assertApproxEqRel(
            from,
            to,
            _getAllowedCollateralRatioSlippage(respectiveDebt),
            "Collateral ratio after action should be within the allowed slippage"
        );
        assertGe(
            newCollateralRatio,
            prevState.collateralRatio,
            "Collateral ratio after action should be greater than or equal to before"
        );
    }

    function _getNewLeverageTokenState(
        uint256 initialCollateral,
        uint256 initialDebtInCollateralAsset,
        uint256 collateralChange,
        uint256 debtChange,
        ExternalAction action
    ) internal view returns (uint256 newCollateral, uint256 newDebt, uint256 newCollateralRatio) {
        debtChange = lendingAdapter.convertDebtToCollateralAsset(debtChange);

        newCollateral =
            action == ExternalAction.Mint ? initialCollateral + collateralChange : initialCollateral - collateralChange;

        newDebt = action == ExternalAction.Mint
            ? initialDebtInCollateralAsset + debtChange
            : initialDebtInCollateralAsset - debtChange;

        newCollateralRatio =
            newDebt != 0 ? Math.mulDiv(newCollateral, _BASE_RATIO(), newDebt, Math.Rounding.Floor) : type(uint256).max;

        return (newCollateral, newDebt, newCollateralRatio);
    }

    function _isFullRedeem(uint256 initialDebt, uint256 debtChange, ExternalAction action)
        internal
        view
        returns (bool)
    {
        return action == ExternalAction.Redeem && initialDebt == lendingAdapter.convertDebtToCollateralAsset(debtChange);
    }

    function _isLeverageTokenEmpty(uint256 collateral) private pure returns (bool) {
        return collateral == 0;
    }
}
