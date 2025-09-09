// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.26;

// Dependency imports
import {Math} from "@openzeppelin/contracts/utils/math/Math.sol";
import {ERC20Mock} from "@openzeppelin/contracts/mocks/token/ERC20Mock.sol";

// Internal imports
import {ExternalAction} from "src/types/DataTypes.sol";
import {ILendingAdapter} from "src/interfaces/ILendingAdapter.sol";
import {ILeverageManager} from "src/interfaces/ILeverageManager.sol";
import {ActionData, LeverageTokenState} from "src/types/DataTypes.sol";
import {PreviewActionTest} from "./PreviewAction.t.sol";

contract MintTest is PreviewActionTest {
    function test_mint() public {
        // collateral:debt is 2:1
        lendingAdapter.mockConvertCollateralToDebtAssetExchangeRate(0.5e8);

        MockLeverageManagerStateForAction memory beforeState =
            MockLeverageManagerStateForAction({collateral: 200 ether, debt: 50 ether, sharesTotalSupply: 100 ether});

        _prepareLeverageManagerStateForAction(beforeState);

        uint256 equityToAddInCollateralAsset = 10 ether;
        _testMint(equityToAddInCollateralAsset, 0, SECONDS_ONE_YEAR);
    }

    function test_mint_WithFees() public {
        leverageManager.exposed_setLeverageTokenActionFee(leverageToken, ExternalAction.Mint, 0.05e4); // 5% fee
        _setTreasuryActionFee(ExternalAction.Mint, 0.1e4); // 10% fee

        _setManagementFee(feeManagerRole, leverageToken, 0.1e4); // 10% management fee
        feeManager.chargeManagementFee(leverageToken);

        MockLeverageManagerStateForAction memory beforeState =
            MockLeverageManagerStateForAction({collateral: 200 ether, debt: 50 ether, sharesTotalSupply: 100 ether});

        _prepareLeverageManagerStateForAction(beforeState);

        uint256 equityToAddInCollateralAsset = 10 ether;
        _testMint(equityToAddInCollateralAsset, 0, SECONDS_ONE_YEAR);
    }

    function testFuzz_mint_SharesTotalSupplyGreaterThanZero(
        uint128 initialCollateral,
        uint128 initialDebtInCollateralAsset,
        uint128 sharesTotalSupply,
        uint128 equityToAddInCollateralAsset
    ) public {
        initialCollateral = uint128(bound(initialCollateral, 1, type(uint128).max));
        initialDebtInCollateralAsset =
            initialCollateral == 1 ? 0 : uint128(bound(initialDebtInCollateralAsset, 1, initialCollateral - 1));
        sharesTotalSupply = uint128(bound(sharesTotalSupply, 1, type(uint128).max));

        _prepareLeverageManagerStateForAction(
            MockLeverageManagerStateForAction({
                collateral: initialCollateral,
                debt: initialDebtInCollateralAsset, // 1:1 exchange rate for this test
                sharesTotalSupply: sharesTotalSupply
            })
        );

        // Ensure the collateral being added does not result in overflows due to mocked value sizes
        equityToAddInCollateralAsset = uint128(bound(equityToAddInCollateralAsset, 1, type(uint96).max));

        uint256 allowedSlippage = _getAllowedCollateralRatioSlippage(initialDebtInCollateralAsset);
        _testMint(equityToAddInCollateralAsset, allowedSlippage, 0);
    }

    function test_mint_EquityToMintIsZero() public {
        // CR is 3x
        _prepareLeverageManagerStateForAction(
            MockLeverageManagerStateForAction({collateral: 9, debt: 3, sharesTotalSupply: 3})
        );

        uint256 equityToAddInCollateralAsset = 0;
        ActionData memory previewData = leverageManager.previewMint(leverageToken, equityToAddInCollateralAsset);

        assertEq(previewData.collateral, 0);
        assertEq(previewData.debt, 0);

        _testMint(equityToAddInCollateralAsset, 0, 0);
    }

    function test_mint_IsEmptyLeverageToken() public {
        MockLeverageManagerStateForAction memory beforeState =
            MockLeverageManagerStateForAction({collateral: 0, debt: 0, sharesTotalSupply: 0});

        _prepareLeverageManagerStateForAction(beforeState);

        uint256 equityToAddInCollateralAsset = 10 ether;
        uint256 collateralToAdd = 20 ether; // 2x CR

        deal(address(collateralToken), address(this), collateralToAdd);
        collateralToken.approve(address(leverageManager), collateralToAdd);

        // Does not revert
        leverageManager.mint(leverageToken, equityToAddInCollateralAsset, equityToAddInCollateralAsset - 1);

        LeverageTokenState memory afterState = leverageManager.getLeverageTokenState(leverageToken);
        assertEq(afterState.collateralInDebtAsset, 20 ether); // 1:1 exchange rate, 2x CR
        assertEq(afterState.debt, 10 ether);
        assertEq(afterState.collateralRatio, 2 * _BASE_RATIO());
    }

    function test_mint_ZeroSharesTotalSupplyWithDust() public {
        MockLeverageManagerStateForAction memory beforeState =
            MockLeverageManagerStateForAction({collateral: 3, debt: 1, sharesTotalSupply: 0});

        _prepareLeverageManagerStateForAction(beforeState);

        uint256 equityToAddInCollateralAsset = 1 ether;
        uint256 expectedCollateralToAdd = 2 ether; // 2x target CR
        uint256 expectedDebtToBorrow = 1 ether;
        uint256 expectedShares = equityToAddInCollateralAsset;

        deal(address(collateralToken), address(this), expectedCollateralToAdd);
        collateralToken.approve(address(leverageManager), expectedCollateralToAdd);

        ActionData memory mintData = leverageManager.mint(leverageToken, equityToAddInCollateralAsset, expectedShares);

        assertEq(mintData.collateral, expectedCollateralToAdd);
        assertEq(mintData.debt, expectedDebtToBorrow);
        assertEq(mintData.shares, expectedShares);
        assertEq(mintData.tokenFee, 0);
        assertEq(mintData.treasuryFee, 0);

        LeverageTokenState memory afterState = leverageManager.getLeverageTokenState(leverageToken);
        assertEq(afterState.collateralInDebtAsset, expectedCollateralToAdd + beforeState.collateral);
        assertEq(afterState.debt, expectedDebtToBorrow + beforeState.debt); // 1:1 collateral to debt exchange rate, 2x target CR
        assertEq(
            afterState.collateralRatio,
            Math.mulDiv(
                expectedCollateralToAdd + beforeState.collateral,
                _BASE_RATIO(),
                expectedDebtToBorrow + beforeState.debt,
                Math.Rounding.Floor
            )
        );
    }

    /// forge-config: default.fuzz.runs = 1
    function testFuzz_mint_RevertIf_SlippageIsTooHigh(uint128 sharesSlippage) public {
        vm.assume(sharesSlippage > 0);

        _prepareLeverageManagerStateForAction(
            MockLeverageManagerStateForAction({collateral: 100 ether, debt: 50 ether, sharesTotalSupply: 10 ether})
        );

        uint256 equityToAddInCollateralAsset = 10 ether;
        ActionData memory previewData = leverageManager.previewMint(leverageToken, equityToAddInCollateralAsset);

        deal(address(collateralToken), address(this), previewData.collateral);
        collateralToken.approve(address(leverageManager), previewData.collateral);

        uint256 minShares = previewData.shares + sharesSlippage; // More than previewed

        vm.expectRevert(
            abi.encodeWithSelector(ILeverageManager.SlippageTooHigh.selector, previewData.shares, minShares)
        );
        leverageManager.mint(leverageToken, equityToAddInCollateralAsset, minShares);
    }

    function _testMint(uint256 equityToAddInCollateralAsset, uint256 collateralRatioDeltaRelative, uint256 deltaTime)
        internal
    {
        skip(deltaTime);

        LeverageTokenState memory beforeState = leverageManager.getLeverageTokenState(leverageToken);
        uint256 beforeSharesTotalSupply = leverageToken.totalSupply();
        uint256 beforeSharesFeeAdjustedTotalSupply = leverageManager.getFeeAdjustedTotalSupply(leverageToken);

        // The assertion for collateral ratio before and after the mint in this helper only makes sense to use
        // if the leverage token has totalSupply > 0 before mint, as a mint of equity into a leverage token with totalSupply = 0
        // will not respect the current collateral ratio of the leverage token, it just uses the target collateral ratio
        require(beforeSharesTotalSupply != 0, "Shares total supply must be non-zero to use _testMint helper function");

        ActionData memory previewData = leverageManager.previewMint(leverageToken, equityToAddInCollateralAsset);

        deal(address(collateralToken), address(this), previewData.collateral);
        collateralToken.approve(address(leverageManager), previewData.collateral);

        ActionData memory expectedMintData = ActionData({
            equity: equityToAddInCollateralAsset,
            collateral: previewData.collateral,
            debt: previewData.debt,
            shares: previewData.shares,
            tokenFee: previewData.tokenFee,
            treasuryFee: previewData.treasuryFee
        });

        vm.expectEmit(true, true, true, true);
        emit ILeverageManager.Mint(leverageToken, address(this), expectedMintData);
        ActionData memory actualMintData =
            leverageManager.mint(leverageToken, equityToAddInCollateralAsset, previewData.shares);

        assertEq(actualMintData.shares, expectedMintData.shares, "Shares received mismatch with preview");
        assertEq(
            leverageToken.balanceOf(address(this)), actualMintData.shares, "Shares received mismatch with returned data"
        );
        assertEq(
            leverageToken.totalSupply(),
            beforeSharesFeeAdjustedTotalSupply + expectedMintData.shares + expectedMintData.treasuryFee,
            "Shares total supply mismatch, should include accrued management fee, treasury action fee, and shares minted for the mint"
        );
        assertEq(
            leverageToken.balanceOf(treasury),
            beforeSharesFeeAdjustedTotalSupply - beforeSharesTotalSupply + expectedMintData.treasuryFee,
            "Treasury should have received the accrued management fee shares and the treasury action fee shares"
        );
        assertEq(actualMintData.tokenFee, expectedMintData.tokenFee, "LeverageToken fee mismatch");
        assertEq(actualMintData.treasuryFee, expectedMintData.treasuryFee, "Treasury fee mismatch");

        LeverageTokenState memory afterState = leverageManager.getLeverageTokenState(leverageToken);
        assertEq(
            afterState.collateralInDebtAsset,
            beforeState.collateralInDebtAsset + lendingAdapter.convertCollateralToDebtAsset(expectedMintData.collateral),
            "Collateral in leverage token after mint mismatch"
        );
        assertEq(actualMintData.collateral, expectedMintData.collateral, "Collateral added mismatch");
        assertEq(
            afterState.debt, beforeState.debt + expectedMintData.debt, "Debt in leverage token after mint mismatch"
        );
        assertEq(actualMintData.debt, expectedMintData.debt, "Debt borrowed mismatch");
        assertEq(debtToken.balanceOf(address(this)), expectedMintData.debt, "Debt tokens received mismatch");

        assertLe(expectedMintData.tokenFee + expectedMintData.treasuryFee, equityToAddInCollateralAsset);

        if (beforeState.collateralRatio == type(uint256).max) {
            assertLe(afterState.collateralRatio, beforeState.collateralRatio);
        } else {
            assertApproxEqRel(
                afterState.collateralRatio,
                beforeState.collateralRatio,
                collateralRatioDeltaRelative,
                "Collateral ratio after mint mismatch"
            );
            assertGe(
                afterState.collateralRatio,
                beforeState.collateralRatio,
                "Collateral ratio after mint should be greater than or equal to before"
            );
        }
    }
}
