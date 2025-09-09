// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.26;

// External imports
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";

// Internal imports
import {ILeverageRouter} from "src/interfaces/periphery/ILeverageRouter.sol";
import {ISwapAdapter} from "src/interfaces/periphery/ISwapAdapter.sol";
import {LeverageRouter} from "src/periphery/LeverageRouter.sol";
import {LeverageRouterTest} from "./LeverageRouter.t.sol";

contract RedeemTest is LeverageRouterTest {
    function testFuzz_redeem_CollateralSwapWithinMaxCostForFlashLoanRepaymentDebt(
        uint128 requiredCollateral,
        uint128 requiredDebt,
        uint128 equityInCollateralAsset,
        uint256 requiredCollateralForSwap,
        uint128 maxSwapCostInCollateralAsset
    ) public {
        vm.assume(requiredDebt < requiredCollateral);

        uint256 mintShares = 10 ether; // Doesn't matter for this test as the shares received and previewed are mocked
        uint256 redeemShares = 5 ether; // Doesn't matter for this test as the shares received and previewed are mocked

        equityInCollateralAsset = requiredCollateral - requiredDebt;
        maxSwapCostInCollateralAsset = uint128(bound(maxSwapCostInCollateralAsset, 0, equityInCollateralAsset - 1));

        // Bound the required collateral for the swap to repay the debt flash loan to be within the max swap cost
        requiredCollateralForSwap = uint256(
            bound(
                requiredCollateralForSwap,
                0,
                uint256(requiredCollateral) - equityInCollateralAsset + maxSwapCostInCollateralAsset
            )
        );

        _mockLeverageManagerRedeem(
            requiredCollateral,
            equityInCollateralAsset,
            requiredDebt,
            requiredCollateralForSwap,
            redeemShares,
            redeemShares
        );

        _deposit(
            equityInCollateralAsset,
            requiredCollateral,
            requiredDebt,
            requiredCollateral - equityInCollateralAsset,
            mintShares
        );

        // Execute the redeem
        deal(address(debtToken), address(this), requiredDebt);
        debtToken.approve(address(leverageRouter), requiredDebt);
        leverageToken.approve(address(leverageRouter), redeemShares);
        leverageRouter.redeem(
            leverageToken,
            equityInCollateralAsset,
            redeemShares,
            maxSwapCostInCollateralAsset,
            // Mock the swap context (doesn't matter for this test as the swap is mocked)
            ISwapAdapter.SwapContext({
                path: new address[](0),
                encodedPath: new bytes(0),
                fees: new uint24[](0),
                tickSpacing: new int24[](0),
                exchange: ISwapAdapter.Exchange.AERODROME,
                exchangeAddresses: ISwapAdapter.ExchangeAddresses({
                    aerodromeRouter: address(0),
                    aerodromePoolFactory: address(0),
                    aerodromeSlipstreamRouter: address(0),
                    uniswapSwapRouter02: address(0),
                    uniswapV2Router02: address(0)
                }),
                additionalData: new bytes(0)
            })
        );

        // Senders shares are burned
        assertEq(leverageToken.balanceOf(address(this)), mintShares - redeemShares);

        // The LeverageRouter has the required debt to repay the flash loan and Morpho is approved to spend it
        assertEq(debtToken.balanceOf(address(leverageRouter)), requiredDebt);
        assertEq(debtToken.allowance(address(leverageRouter), address(morpho)), requiredDebt);

        // Sender receives the remaining collateral (equity)
        assertEq(collateralToken.balanceOf(address(this)), requiredCollateral - requiredCollateralForSwap);
        assertGe(collateralToken.balanceOf(address(this)), equityInCollateralAsset - maxSwapCostInCollateralAsset);
    }

    function testFuzz_redeem_CollateralSwapMoreThanMaxCostForFlashLoanRepaymentDebt(
        uint128 requiredCollateral,
        uint128 requiredDebt,
        uint128 equityInCollateralAsset,
        uint256 requiredCollateralForSwap,
        uint128 maxSwapCostInCollateralAsset
    ) public {
        vm.assume(requiredDebt < requiredCollateral);

        uint256 shares = 10 ether; // Doesn't matter for this test as the shares received and previewed are mocked

        equityInCollateralAsset = requiredCollateral - requiredDebt;
        maxSwapCostInCollateralAsset = uint128(bound(maxSwapCostInCollateralAsset, 0, equityInCollateralAsset - 1));

        // Bound the required collateral for the swap to repay the debt flash loan to dip deeper into the equity than
        // allowed, per the max swap cost parameter
        requiredCollateralForSwap = uint256(
            bound(
                requiredCollateralForSwap,
                uint256(requiredCollateral) - equityInCollateralAsset + maxSwapCostInCollateralAsset + 1,
                requiredCollateral
            )
        );

        _mockLeverageManagerRedeem(
            requiredCollateral, equityInCollateralAsset, requiredDebt, requiredCollateralForSwap, shares, shares
        );

        _deposit(
            equityInCollateralAsset,
            requiredCollateral,
            requiredDebt,
            requiredCollateral - equityInCollateralAsset,
            shares
        );

        // Execute the redeem
        deal(address(debtToken), address(this), requiredDebt);
        debtToken.approve(address(leverageRouter), requiredDebt);
        leverageToken.approve(address(leverageRouter), shares);

        vm.expectRevert(
            abi.encodeWithSelector(
                ILeverageRouter.MaxSwapCostExceeded.selector,
                equityInCollateralAsset - (requiredCollateral - requiredCollateralForSwap),
                maxSwapCostInCollateralAsset
            )
        );
        leverageRouter.redeem(
            leverageToken,
            equityInCollateralAsset,
            shares,
            maxSwapCostInCollateralAsset,
            // Mock the swap context (doesn't matter for this test as the swap is mocked)
            ISwapAdapter.SwapContext({
                path: new address[](0),
                encodedPath: new bytes(0),
                fees: new uint24[](0),
                tickSpacing: new int24[](0),
                exchange: ISwapAdapter.Exchange.AERODROME,
                exchangeAddresses: ISwapAdapter.ExchangeAddresses({
                    aerodromeRouter: address(0),
                    aerodromePoolFactory: address(0),
                    aerodromeSlipstreamRouter: address(0),
                    uniswapSwapRouter02: address(0),
                    uniswapV2Router02: address(0)
                }),
                additionalData: new bytes(0)
            })
        );
    }

    function test_redeem_TransfersPreviewSharesNotMaxShares() public {
        uint256 totalShares = 60 ether;
        uint256 redeemShares = totalShares / 2;
        uint256 redeemEquityInCollateralAsset = 15 ether;

        // Mock the redeem to burn half of the user's shares (redeemShares). Other values are mocked and don't matter
        // for this test
        _mockLeverageManagerRedeem(
            30 ether, redeemEquityInCollateralAsset, 15 ether, 15 ether, redeemShares, totalShares
        );

        _deposit(30 ether, 60 ether, 30 ether, 30 ether, totalShares);
        leverageToken.approve(address(leverageRouter), totalShares);

        // Expect the shares to be redeemed to be transferred to the LeverageRouter, not the maxShares parameter (totalShares)
        vm.expectEmit(true, true, true, true);
        emit IERC20.Transfer(address(this), address(leverageRouter), redeemShares);

        leverageRouter.redeem(
            leverageToken,
            redeemEquityInCollateralAsset,
            totalShares, // The total shares are passed as the maxShares parameter
            redeemEquityInCollateralAsset,
            ISwapAdapter.SwapContext({
                path: new address[](0),
                encodedPath: new bytes(0),
                fees: new uint24[](0),
                tickSpacing: new int24[](0),
                exchange: ISwapAdapter.Exchange.AERODROME,
                exchangeAddresses: ISwapAdapter.ExchangeAddresses({
                    aerodromeRouter: address(0),
                    aerodromePoolFactory: address(0),
                    aerodromeSlipstreamRouter: address(0),
                    uniswapSwapRouter02: address(0),
                    uniswapV2Router02: address(0)
                }),
                additionalData: new bytes(0)
            })
        );

        // Half of the user's shares were burned
        assertEq(leverageToken.balanceOf(address(this)), redeemShares);
    }
}
