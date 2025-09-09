// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.26;

// Dependency imports
import {UnsafeUpgrades} from "@foundry-upgrades/Upgrades.sol";

// Internal imports
import {ILeverageToken} from "src/interfaces/ILeverageToken.sol";
import {LeverageManager} from "src/LeverageManager.sol";
import {BeaconProxyFactory} from "src/BeaconProxyFactory.sol";
import {LeverageToken} from "src/LeverageToken.sol";
import {LeverageManagerHarness} from "test/unit/harness/LeverageManagerHarness.t.sol";
import {MorphoLendingAdapterTest} from "../MorphoLendingAdapter.t.sol";
import {ILeverageManager} from "src/interfaces/ILeverageManager.sol";
import {ILendingAdapter} from "src/interfaces/ILendingAdapter.sol";
import {IntegrationTestBase} from "../IntegrationTestBase.t.sol";
import {LeverageTokenState, ActionData, ActionDataV2} from "src/types/DataTypes.sol";

contract LeverageManagerTest is IntegrationTestBase {
    function testFork_setUp() public view virtual override {
        assertEq(address(leverageManager.getLeverageTokenCollateralAsset(leverageToken)), address(WETH));
        assertEq(address(leverageManager.getLeverageTokenDebtAsset(leverageToken)), address(USDC));
    }

    function _deposit(address caller, uint256 collateralToDeposit, uint256 minShares)
        internal
        returns (ActionDataV2 memory)
    {
        deal(address(WETH), caller, collateralToDeposit);
        vm.startPrank(caller);
        WETH.approve(address(leverageManager), collateralToDeposit);
        ActionDataV2 memory depositData = leverageManager.deposit(leverageToken, collateralToDeposit, minShares);
        vm.stopPrank();

        return depositData;
    }

    function _mint(address caller, uint256 equityInCollateralAsset, uint256 collateralToAdd)
        internal
        returns (uint256)
    {
        deal(address(WETH), caller, collateralToAdd);
        vm.startPrank(caller);
        WETH.approve(address(leverageManager), collateralToAdd);
        uint256 shares = leverageManager.mint(leverageToken, equityInCollateralAsset, 0).shares;
        vm.stopPrank();

        return shares;
    }

    function _mintV2(address caller, uint256 sharesToMint, uint256 maxCollateral)
        internal
        returns (ActionDataV2 memory)
    {
        deal(address(WETH), caller, maxCollateral);
        vm.startPrank(caller);
        WETH.approve(address(leverageManager), maxCollateral);
        ActionDataV2 memory mintData = leverageManager.mintV2(leverageToken, sharesToMint, maxCollateral);
        vm.stopPrank();

        return mintData;
    }

    function _redeem(address caller, uint256 equityInCollateralAsset, uint256 debtToRepay) internal returns (uint256) {
        deal(address(USDC), caller, debtToRepay);
        vm.startPrank(caller);
        USDC.approve(address(leverageManager), debtToRepay);
        uint256 shares = leverageManager.redeem(leverageToken, equityInCollateralAsset, type(uint256).max).shares;
        vm.stopPrank();

        return shares;
    }

    function getLeverageTokenState() internal view returns (LeverageTokenState memory) {
        return LeverageManagerHarness(address(leverageManager)).getLeverageTokenState(leverageToken);
    }
}
