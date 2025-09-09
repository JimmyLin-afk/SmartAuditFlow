// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.26;

// Dependency imports
import {Math} from "@openzeppelin/contracts/utils/math/Math.sol";

// Dependency imports
import {IOracle} from "@morpho-blue/interfaces/IOracle.sol";

// Internal imports
import {ILendingAdapter} from "src/interfaces/ILendingAdapter.sol";
import {ILeverageManager} from "src/interfaces/ILeverageManager.sol";
import {MorphoLendingAdapter} from "src/lending/MorphoLendingAdapter.sol";
import {LeverageManagerTest} from "./LeverageManager.t.sol";
import {ActionData, LeverageTokenState, ExternalAction} from "src/types/DataTypes.sol";

contract LeverageManagerMintTest is LeverageManagerTest {
    /// @dev In this block price on oracle 3392.292471591441746049801068
    function testFork_mint_NoFee() public {
        uint256 equityInCollateralAsset = 10 ether;
        uint256 collateralToAdd = 2 * equityInCollateralAsset;
        uint256 debtToBorrow = 33922_924715; // 33922.924715

        deal(address(WETH), user, collateralToAdd);

        vm.startPrank(user);
        WETH.approve(address(leverageManager), collateralToAdd);
        leverageManager.mint(leverageToken, equityInCollateralAsset, 0);
        vm.stopPrank();

        assertEq(leverageToken.balanceOf(user), equityInCollateralAsset);
        assertEq(WETH.balanceOf(user), 0);
        assertEq(USDC.balanceOf(user), debtToBorrow);

        assertEq(morphoLendingAdapter.getCollateral(), collateralToAdd);
        assertGe(morphoLendingAdapter.getDebt(), debtToBorrow);
        assertLe(morphoLendingAdapter.getDebt(), debtToBorrow + 1);

        // Validate that user never gets more equity than they minted
        uint256 equityAfterMint = _convertToAssets(equityInCollateralAsset);
        assertGe(equityInCollateralAsset, equityAfterMint);
    }

    function testFork_mint_WithFees() public {
        uint256 treasuryActionFee = 10_00; // 10%
        leverageManager.setTreasuryActionFee(ExternalAction.Mint, treasuryActionFee);

        uint256 tokenActionFee = 10_00; // 10%
        leverageToken = _createNewLeverageToken(BASE_RATIO, 2 * BASE_RATIO, 3 * BASE_RATIO, tokenActionFee, 0);

        uint256 managementFee = 10_00; // 10%
        leverageManager.setManagementFee(leverageToken, managementFee);

        morphoLendingAdapter =
            MorphoLendingAdapter(address(leverageManager.getLeverageTokenLendingAdapter(leverageToken)));

        uint256 equityInCollateralAsset = 10 ether;
        uint256 collateralToAdd = 2 * equityInCollateralAsset;
        _mint(user, equityInCollateralAsset, collateralToAdd);

        // 8.1 ether because 10% of equity is for diluting leverage token shares, and 10% of the remaining shares
        // after subtracting the dilution is for the treasury fee (10 * 0.9) * 0.9 = 8.1
        assertEq(leverageToken.balanceOf(user), 8.1 ether);
        assertEq(leverageToken.balanceOf(treasury), 0.9 ether);
        assertEq(leverageToken.balanceOf(user) + leverageToken.balanceOf(treasury), leverageToken.totalSupply());
        // Some slight deviation from 10 ether is expected due to interest accrual in morpho and rounding errors
        assertEq(morphoLendingAdapter.getEquityInCollateralAsset(), 9999999999974771473);

        uint256 collateralRatio = leverageManager.getLeverageTokenState(leverageToken).collateralRatio;
        assertEq(collateralRatio, 1999999999970521409);

        // One year passes, same mint amount occurs
        skip(SECONDS_ONE_YEAR);

        // CR goes down due to morpho borrow interest
        collateralRatio = leverageManager.getLeverageTokenState(leverageToken).collateralRatio;
        assertEq(collateralRatio, 1974502635802161566);

        ActionData memory previewData = leverageManager.previewMint(leverageToken, equityInCollateralAsset);
        // collateralToAdd is higher than before due to higher leverage from CR going down
        assertEq(previewData.collateral, 20.261644896959132013 ether);
        // more shares are minted to the user due to share dilution from management fee and morpho borrow interest
        assertEq(previewData.shares, 8.123906521435763979 ether);
        // treasury fee is 10% of the total shares minted for the mint
        assertEq(previewData.treasuryFee, 0.902656280159529332 ether);

        // Preview data is the same after charging management fee but treasury balance of LT increases by 0.9 ether (10% of total supply)
        leverageManager.chargeManagementFee(leverageToken);
        previewData = leverageManager.previewMint(leverageToken, equityInCollateralAsset);
        assertEq(previewData.collateral, 20.261644896959132013 ether);
        assertEq(previewData.shares, 8.123906521435763979 ether); // 90% of total shares minted
        assertEq(previewData.treasuryFee, 0.902656280159529332 ether); // 10% of total shares minted
        assertEq(leverageToken.balanceOf(treasury), 1.8 ether);

        // Mint again
        _mint(user, equityInCollateralAsset, previewData.collateral);

        assertEq(leverageToken.balanceOf(user), 8.1 ether + previewData.shares);
        assertEq(leverageToken.balanceOf(treasury), 1.8 ether + previewData.treasuryFee);
        assertEq(leverageToken.totalSupply(), leverageToken.balanceOf(user) + leverageToken.balanceOf(treasury));
    }

    /// @dev In this block price on oracle 3392.292471591441746049801068
    function testFork_mint_PriceChangedBetweenMints_CollateralRatioDoesNotChange() public {
        leverageToken = _createNewLeverageToken(
            BASE_RATIO,
            2 * BASE_RATIO,
            3 * BASE_RATIO,
            1, // 0.01% leverage token fee
            0
        );
        morphoLendingAdapter =
            MorphoLendingAdapter(address(leverageManager.getLeverageTokenLendingAdapter(leverageToken)));

        // Mint again like in previous test
        uint256 equityInCollateralAsset = 10 ether;
        uint256 collateralToAdd = 2 * equityInCollateralAsset;
        _mint(user, equityInCollateralAsset, collateralToAdd);

        LeverageTokenState memory stateAfterFirstMint = getLeverageTokenState();
        assertEq(stateAfterFirstMint.collateralRatio, 1999999999970521409); // ~2x CR

        // Price doubles
        (,, address oracle,,) = morphoLendingAdapter.marketParams();
        uint256 currentPrice = IOracle(oracle).price();
        uint256 newPrice = currentPrice * 2;
        vm.mockCall(address(oracle), abi.encodeWithSelector(IOracle.price.selector), abi.encode(newPrice));

        // Since price of ETH doubled current collateral ratio should be 4x and not 2x
        LeverageTokenState memory stateBefore = getLeverageTokenState();
        assertEq(stateBefore.collateralRatio, 3999999999970521409); // ~4x CR
        assertGe(stateBefore.collateralRatio, 2 * stateAfterFirstMint.collateralRatio);

        // Mint based on what preview function says
        uint256 collateral = leverageManager.previewMint(leverageToken, equityInCollateralAsset).collateral;
        uint256 shares = _mint(user, equityInCollateralAsset, collateral);

        // Validate that user never gets more equity than they minted
        uint256 equityAfterMint = _convertToAssets(shares);
        assertGe(equityInCollateralAsset, equityAfterMint);

        // Validate that user has no WETH left
        assertEq(WETH.balanceOf(user), 0);

        // Validate that collateral ratio did not change (minus some rounding error) which means that new mint follows
        // current collateral ratio and not target. It is important that there can be rounding error but it should bring
        // collateral ratio up not down
        LeverageTokenState memory stateAfter = getLeverageTokenState();
        assertGe(stateAfter.collateralRatio, stateBefore.collateralRatio);
        assertEq(stateAfter.collateralRatio, 3999999999982312845);

        // // Price goes down 3x
        newPrice /= 3;
        vm.mockCall(address(oracle), abi.encodeWithSelector(IOracle.price.selector), abi.encode(newPrice));

        stateBefore = getLeverageTokenState();
        assertEq(stateBefore.collateralRatio, 1333333333321541897);

        collateral = leverageManager.previewMint(leverageToken, equityInCollateralAsset).collateral;
        shares = _mint(user, equityInCollateralAsset, collateral);

        // Validate that user never gets more equity than they minted
        equityAfterMint = _convertToAssets(shares);
        assertGe(equityInCollateralAsset, equityAfterMint);

        // Validate that collateral ratio did not change (minus some rounding error) which means that new mint follows
        // current collateral ratio and not target. It is important that there can be rounding error but it should bring
        // collateral ratio up not down
        stateAfter = getLeverageTokenState();
        assertGe(stateAfter.collateralRatio, stateBefore.collateralRatio);
        assertEq(stateAfter.collateralRatio, 1333333333327973589);
    }
}
