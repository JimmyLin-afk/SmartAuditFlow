// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.26;

// Dependency imports
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {IMorpho, MarketParams} from "@morpho-blue/interfaces/IMorpho.sol";
import {Math} from "@openzeppelin/contracts/utils/math/Math.sol";

// Internal imports
import {LeverageRouter} from "src/periphery/LeverageRouter.sol";
import {MorphoLendingAdapter} from "src/lending/MorphoLendingAdapter.sol";
import {RebalanceAdapter} from "src/rebalance/RebalanceAdapter.sol";
import {ILendingAdapter} from "src/interfaces/ILendingAdapter.sol";
import {ILeverageManager} from "src/interfaces/ILeverageManager.sol";
import {ILeverageRouter} from "src/interfaces/periphery/ILeverageRouter.sol";
import {ILeverageToken} from "src/interfaces/ILeverageToken.sol";
import {IRebalanceAdapter} from "src/interfaces/IRebalanceAdapter.sol";
import {ISwapAdapter} from "src/interfaces/periphery/ISwapAdapter.sol";
import {ActionDataV2, LeverageTokenConfig} from "src/types/DataTypes.sol";
import {LeverageRouterTest} from "./LeverageRouter.t.sol";
import {SwapPathLib} from "../../utils/SwapPathLib.sol";
import {MockSwapper} from "../../unit/mock/MockSwapper.sol";

contract LeverageRouterDepositTest is LeverageRouterTest {
    struct DepositWithMockedSwapParams {
        ILeverageToken leverageToken;
        IERC20 collateralAsset;
        IERC20 debtAsset;
        uint256 userBalanceOfCollateralAsset;
        uint256 collateralFromSender;
        uint256 flashLoanAmount;
        uint256 minShares;
        uint256 collateralRequired;
        uint256 collateralReceivedFromDebtSwap;
    }

    ILeverageRouter public leverageRouterWithMockSwapAdapter;

    MockSwapper public mockSwapper;

    MorphoLendingAdapter ethShortLendingAdapter;
    RebalanceAdapter ethShortRebalanceAdapter;

    ILeverageToken ethShortLeverageToken;

    function setUp() public override {
        super.setUp();

        mockSwapper = new MockSwapper();

        ethShortRebalanceAdapter =
            _deployRebalanceAdapter(1.3e18, 1.5e18, 2e18, 7 minutes, 1.2e18, 0.9e18, 1.3e18, 45_66);

        ethShortLendingAdapter = MorphoLendingAdapter(
            address(morphoLendingAdapterFactory.deployAdapter(USDC_WETH_MARKET_ID, address(this), bytes32(uint256(1))))
        );

        leverageRouterWithMockSwapAdapter =
            new LeverageRouter(leverageManager, MORPHO, ISwapAdapter(address(mockSwapper)));

        ethShortLeverageToken = leverageManager.createNewLeverageToken(
            LeverageTokenConfig({
                lendingAdapter: ILendingAdapter(address(ethShortLendingAdapter)),
                rebalanceAdapter: IRebalanceAdapter(address(ethShortRebalanceAdapter)),
                mintTokenFee: 0,
                redeemTokenFee: 0
            }),
            "Seamless USDC/ETH 2x leverage token",
            "ltUSDC/ETH-2x"
        );
    }

    /// @dev In this block price on oracle 3392.292471591441746049801068
    function testFork_deposit_UniswapV2_FirstDeposit() public {
        uint256 collateralFromSender = 1 ether;
        uint256 collateralToAdd = 2 * collateralFromSender;
        uint256 userBalanceOfCollateralAsset = 4 ether; // User has more than enough assets for the mint of equity
        uint256 flashLoanAmount = 3392.292471e6; // 3392.292471 USDC
        uint256 minShares = 1 ether * 0.99e18 / 1e18; // 1% slippage
        uint256 collateralReceivedFromDebtSwap = 0.997140594716559346 ether; // Swap of 3392.292471 USDC

        {
            // Sanity check that LR preview deposit matches test params
            ActionDataV2 memory leverageRouterPreview =
                leverageRouter.previewDeposit(leverageToken, collateralFromSender);
            assertEq(leverageRouterPreview.debt, flashLoanAmount);
            assertEq(leverageRouterPreview.shares, 1 ether);
            assertEq(leverageRouterPreview.collateral, collateralToAdd);
            assertEq(leverageRouterPreview.tokenFee, 0);
            assertEq(leverageRouterPreview.treasuryFee, 0);
        }

        // The swap results in less collateral than required to get the flash loaned debt amount from a LM deposit, so the debt amount flash loaned
        // needs to be reduced. We reduce it by the percentage delta between the required collateral and the collateral received from the swap
        uint256 deltaPercentage = collateralReceivedFromDebtSwap * 1e18 / (collateralToAdd - collateralFromSender);
        assertEq(deltaPercentage, 0.997140594716559346e18);
        uint256 flashLoanAmountReduced = flashLoanAmount * deltaPercentage / 1e18;
        assertEq(flashLoanAmountReduced, 3382.592531e6);

        // Updated collateral received from the debt swap for lower debt amount
        collateralReceivedFromDebtSwap = 0.994290732650270211 ether;

        // Preview again using the total collateral. This is used by the LM deposit logic
        uint256 totalCollateral = collateralFromSender + collateralReceivedFromDebtSwap;
        assertEq(totalCollateral, 1.994290732650270211 ether);
        ActionDataV2 memory previewData = leverageManager.previewDeposit(leverageToken, totalCollateral);
        assertGe(previewData.debt, flashLoanAmountReduced);
        assertEq(previewData.debt, 3382.608719e6);

        // More than minShares (1% slippage) will be minted
        assertGe(previewData.shares, minShares);
        assertEq(previewData.shares, 0.997145366325135105 ether);

        address[] memory path = new address[](2);
        path[0] = address(USDC);
        path[1] = address(WETH);

        ISwapAdapter.SwapContext memory swapContext = ISwapAdapter.SwapContext({
            exchange: ISwapAdapter.Exchange.UNISWAP_V2,
            encodedPath: new bytes(0),
            path: path,
            fees: new uint24[](0),
            tickSpacing: new int24[](0),
            exchangeAddresses: ISwapAdapter.ExchangeAddresses({
                aerodromeRouter: address(0),
                aerodromePoolFactory: address(0),
                aerodromeSlipstreamRouter: address(0),
                uniswapSwapRouter02: address(0),
                uniswapV2Router02: UNISWAP_V2_ROUTER02
            }),
            additionalData: new bytes(0)
        });

        _dealAndDeposit(
            WETH,
            USDC,
            userBalanceOfCollateralAsset,
            collateralFromSender,
            flashLoanAmountReduced,
            minShares,
            swapContext
        );

        // Collateral is taken from the user for the deposit. All of the collateral should be used
        assertEq(WETH.balanceOf(user), userBalanceOfCollateralAsset - collateralFromSender);

        // Any additional debt that is not used to repay the flash loan is given to the user
        uint256 excessDebt = previewData.debt - flashLoanAmountReduced;
        assertEq(USDC.balanceOf(user), excessDebt);
        assertEq(USDC.balanceOf(user), 0.016188e6);

        assertGe(leverageToken.balanceOf(user), minShares);

        assertEq(morphoLendingAdapter.getCollateral(), totalCollateral);
        assertEq(morphoLendingAdapter.getDebt(), previewData.debt + 1); // + 1 because of rounding up by MorphoBalancesLib.expectedBorrowAssets
    }

    /// @dev In this block price on oracle 3392.292471591441746049801068
    function testFork_deposit_UniswapV2_MultipleDeposits() public {
        // Params from testFork_deposit_UniswapV2_FirstDeposit
        uint256 userBalanceOfCollateralAsset = 4 ether;
        uint256 collateralFromSender = 1 ether;
        uint256 flashLoanAmount = 3382.592531e6;
        uint256 minShares = 0.99 ether;

        address[] memory path = new address[](2);
        path[0] = address(USDC);
        path[1] = address(WETH);

        ISwapAdapter.SwapContext memory swapContext = ISwapAdapter.SwapContext({
            exchange: ISwapAdapter.Exchange.UNISWAP_V2,
            encodedPath: new bytes(0),
            path: path,
            fees: new uint24[](0),
            tickSpacing: new int24[](0),
            exchangeAddresses: ISwapAdapter.ExchangeAddresses({
                aerodromeRouter: address(0),
                aerodromePoolFactory: address(0),
                aerodromeSlipstreamRouter: address(0),
                uniswapSwapRouter02: address(0),
                uniswapV2Router02: UNISWAP_V2_ROUTER02
            }),
            additionalData: new bytes(0)
        });

        _dealAndDeposit(
            WETH, USDC, userBalanceOfCollateralAsset, collateralFromSender, flashLoanAmount, minShares, swapContext
        );

        uint256 expectedUserDebtBalance = 0.016188e6;
        assertEq(USDC.balanceOf(user), expectedUserDebtBalance);

        {
            ActionDataV2 memory previewDataFullDeposit =
                leverageRouter.previewDeposit(leverageToken, collateralFromSender);
            uint256 collateralReceivedFromDebtSwap = 0.993336131989824069 ether;

            // The swap results in less collateral than required to get the flash loaned debt amount from a LM deposit, so the debt amount flash loaned
            // needs to be reduced. We reduce it by the percentage delta between the required collateral and the collateral received from the swap
            uint256 deltaPercentage =
                collateralReceivedFromDebtSwap * 1e18 / (previewDataFullDeposit.collateral - collateralFromSender);
            assertEq(deltaPercentage, 0.993336131402504508e18);
            uint256 flashLoanAmountReduced = previewDataFullDeposit.debt * deltaPercentage / 1e18;
            assertEq(flashLoanAmountReduced, 3369.686681e6);

            // Update for debtReduced
            collateralReceivedFromDebtSwap = 0.989547994451029601 ether;

            // Preview again using the total collateral. This is used by the LR deposit logic
            uint256 totalCollateral = collateralFromSender + collateralReceivedFromDebtSwap;
            assertEq(totalCollateral, 1.989547994451029601 ether);
            ActionDataV2 memory previewDataReducedDeposit =
                leverageManager.previewDeposit(leverageToken, totalCollateral);
            assertGe(previewDataReducedDeposit.debt, flashLoanAmountReduced);
            assertEq(previewDataReducedDeposit.debt, 3374.564342e6);

            // More than minShares (1% slippage) will be minted
            assertGe(previewDataReducedDeposit.shares, minShares);
            assertEq(previewDataReducedDeposit.shares, 0.9947739972255148 ether);

            // Reverts due to 1 debt asset left over in the LR.
            _dealAndDeposit(
                WETH,
                USDC,
                userBalanceOfCollateralAsset,
                collateralFromSender,
                flashLoanAmountReduced,
                previewDataReducedDeposit.shares,
                swapContext
            );

            // Any additional debt that is not used to repay the flash loan is given to the user
            uint256 surplusDebtFromDeposit = previewDataReducedDeposit.debt - flashLoanAmountReduced;
            assertEq(surplusDebtFromDeposit, 4.877661e6);
            expectedUserDebtBalance += surplusDebtFromDeposit;
            assertEq(USDC.balanceOf(user), expectedUserDebtBalance);
        }

        {
            ActionDataV2 memory previewDataFullDeposit =
                leverageRouter.previewDeposit(leverageToken, collateralFromSender);
            uint256 collateralReceivedFromDebtSwap = 0.995230946229750636 ether;

            // The swap results in less collateral than required to get the flash loaned debt amount from a LM deposit, so the debt amount flash loaned
            // needs to be reduced. We reduce it by the percentage delta between the required collateral and the collateral received from the swap
            uint256 deltaPercentage =
                collateralReceivedFromDebtSwap * 1e18 / (previewDataFullDeposit.collateral - collateralFromSender);
            assertEq(deltaPercentage, 0.995230945787895318e18);
            uint256 flashLoanAmountReduced = previewDataFullDeposit.debt * deltaPercentage / 1e18;
            assertEq(flashLoanAmountReduced, 3376.114445e6);

            // Update for debtReduced
            collateralReceivedFromDebtSwap = 0.9904869053653832 ether;

            // Preview again using the total collateral. This is used by the LR deposit logic
            uint256 totalCollateral = collateralFromSender + collateralReceivedFromDebtSwap;
            assertEq(totalCollateral, 1.9904869053653832 ether);
            ActionDataV2 memory previewDataReducedDeposit =
                leverageManager.previewDeposit(leverageToken, totalCollateral);
            assertGe(previewDataReducedDeposit.debt, flashLoanAmountReduced);
            assertEq(previewDataReducedDeposit.debt, 3376.156872e6);

            // More than minShares (1% slippage) will be minted
            assertGe(previewDataReducedDeposit.shares, minShares);
            assertEq(previewDataReducedDeposit.shares, 0.995243452682691599 ether);

            _dealAndDeposit(
                WETH,
                USDC,
                userBalanceOfCollateralAsset,
                collateralFromSender,
                flashLoanAmountReduced,
                previewDataReducedDeposit.shares,
                swapContext
            );

            // Any additional debt that is not used to repay the flash loan is given to the user
            uint256 surplusDebtFromDeposit = previewDataReducedDeposit.debt - flashLoanAmountReduced;
            assertEq(surplusDebtFromDeposit, 0.042427e6);
            expectedUserDebtBalance += surplusDebtFromDeposit;
            assertEq(USDC.balanceOf(user), expectedUserDebtBalance);
        }

        {
            ActionDataV2 memory previewDataFullDeposit =
                leverageRouter.previewDeposit(leverageToken, collateralFromSender);
            uint256 collateralReceivedFromDebtSwap = 0.9942781864904543 ether;

            // The swap results in less collateral than required to get the flash loaned debt amount from a LM deposit, so the debt amount flash loaned
            // needs to be reduced. We reduce it by the percentage delta between the required collateral and the collateral received from the swap
            uint256 deltaPercentage =
                collateralReceivedFromDebtSwap * 1e18 / (previewDataFullDeposit.collateral - collateralFromSender);
            assertEq(deltaPercentage, 0.994278186196095526e18);
            uint256 flashLoanAmountReduced = previewDataFullDeposit.debt * deltaPercentage / 1e18;
            assertEq(flashLoanAmountReduced, 3372.882406e6);

            // Update for debtReduced
            collateralReceivedFromDebtSwap = 0.988591828264731799 ether;

            // Preview again using the total collateral. This is used by the LR deposit logic
            uint256 totalCollateral = collateralFromSender + collateralReceivedFromDebtSwap;
            assertEq(totalCollateral, 1.988591828264731799 ether);
            ActionDataV2 memory previewDataReducedDeposit =
                leverageManager.previewDeposit(leverageToken, totalCollateral);
            assertGe(previewDataReducedDeposit.debt, flashLoanAmountReduced);
            assertEq(previewDataReducedDeposit.debt, 3372.942544e6);

            // More than minShares (1% slippage) will be minted
            assertGe(previewDataReducedDeposit.shares, minShares);
            assertEq(previewDataReducedDeposit.shares, 0.994295914132365898 ether);

            _dealAndDeposit(
                WETH,
                USDC,
                userBalanceOfCollateralAsset,
                collateralFromSender,
                flashLoanAmountReduced,
                previewDataReducedDeposit.shares,
                swapContext
            );

            // Any additional debt that is not used to repay the flash loan is given to the user
            uint256 surplusDebtFromDeposit = previewDataReducedDeposit.debt - flashLoanAmountReduced;
            assertEq(surplusDebtFromDeposit, 0.060138e6);
            expectedUserDebtBalance += surplusDebtFromDeposit;
            assertEq(USDC.balanceOf(user), expectedUserDebtBalance);
        }
    }

    /// @dev In this block price on oracle 3392.292471591441746049801068
    function testForkFuzz_deposit_MultipleDeposits_MockedSwap_CollateralDecimalsGtDebtDecimals(
        uint256 collateralFromSenderA,
        uint256 collateralReceivedFromDebtSwapA,
        uint256 collateralFromSenderB,
        uint256 collateralReceivedFromDebtSwapB,
        uint256 collateralFromSenderC,
        uint256 collateralReceivedFromDebtSwapC
    ) public {
        collateralFromSenderA = bound(collateralFromSenderA, 1, 1000 ether);
        collateralFromSenderB = bound(collateralFromSenderB, 1, 1000 ether);
        collateralFromSenderC = bound(collateralFromSenderC, 1, 1000 ether);

        _supplyUSDCForETHLongLeverageToken(15000000e6);

        ActionDataV2 memory previewData = leverageRouter.previewDeposit(leverageToken, collateralFromSenderA);

        collateralReceivedFromDebtSwapA =
            bound(collateralReceivedFromDebtSwapA, 1, previewData.collateral - collateralFromSenderA);

        _depositWithMockedSwap(
            DepositWithMockedSwapParams({
                leverageToken: leverageToken,
                collateralAsset: WETH,
                debtAsset: USDC,
                userBalanceOfCollateralAsset: collateralFromSenderA,
                collateralFromSender: collateralFromSenderA,
                flashLoanAmount: previewData.debt,
                minShares: 0,
                collateralRequired: previewData.collateral,
                collateralReceivedFromDebtSwap: collateralReceivedFromDebtSwapA
            })
        );

        previewData = leverageRouter.previewDeposit(leverageToken, collateralFromSenderB);

        collateralReceivedFromDebtSwapB =
            bound(collateralReceivedFromDebtSwapB, 1, previewData.collateral - collateralFromSenderB);

        _depositWithMockedSwap(
            DepositWithMockedSwapParams({
                leverageToken: leverageToken,
                collateralAsset: WETH,
                debtAsset: USDC,
                userBalanceOfCollateralAsset: collateralFromSenderB,
                collateralFromSender: collateralFromSenderB,
                flashLoanAmount: previewData.debt,
                minShares: 0,
                collateralRequired: previewData.collateral,
                collateralReceivedFromDebtSwap: collateralReceivedFromDebtSwapB
            })
        );

        previewData = leverageRouter.previewDeposit(leverageToken, collateralFromSenderC);

        collateralReceivedFromDebtSwapC =
            bound(collateralReceivedFromDebtSwapC, 1, previewData.collateral - collateralFromSenderC);

        _depositWithMockedSwap(
            DepositWithMockedSwapParams({
                leverageToken: leverageToken,
                collateralAsset: WETH,
                debtAsset: USDC,
                userBalanceOfCollateralAsset: collateralFromSenderC,
                collateralFromSender: collateralFromSenderC,
                flashLoanAmount: previewData.debt,
                minShares: 0,
                collateralRequired: previewData.collateral,
                collateralReceivedFromDebtSwap: collateralReceivedFromDebtSwapC
            })
        );
    }

    /// @dev In this block price on oracle 3392.292471591441746049801068
    function testForkFuzz_deposit_MultipleDeposits_MockedSwap_DebtDecimalsGtCollateralDecimals(
        uint256 collateralFromSenderA,
        uint256 collateralReceivedFromDebtSwapA,
        uint256 collateralFromSenderB,
        uint256 collateralReceivedFromDebtSwapB,
        uint256 collateralFromSenderC,
        uint256 collateralReceivedFromDebtSwapC
    ) public {
        collateralFromSenderA = bound(collateralFromSenderA, 1, 3000000e6);
        collateralFromSenderB = bound(collateralFromSenderB, 1, 3000000e6);
        collateralFromSenderC = bound(collateralFromSenderC, 1, 3000000e6);

        _supplyWETHForETHShortLeverageToken(6000 ether);

        ActionDataV2 memory previewData = leverageRouter.previewDeposit(ethShortLeverageToken, collateralFromSenderA);

        collateralReceivedFromDebtSwapA =
            bound(collateralReceivedFromDebtSwapA, 1, previewData.collateral - collateralFromSenderA);

        _depositWithMockedSwap(
            DepositWithMockedSwapParams({
                leverageToken: ethShortLeverageToken,
                collateralAsset: USDC,
                debtAsset: WETH,
                userBalanceOfCollateralAsset: collateralFromSenderA,
                collateralFromSender: collateralFromSenderA,
                flashLoanAmount: previewData.debt,
                minShares: 0,
                collateralRequired: previewData.collateral,
                collateralReceivedFromDebtSwap: collateralReceivedFromDebtSwapA
            })
        );

        previewData = leverageRouter.previewDeposit(ethShortLeverageToken, collateralFromSenderB);

        collateralReceivedFromDebtSwapB =
            bound(collateralReceivedFromDebtSwapB, 1, previewData.collateral - collateralFromSenderB);

        _depositWithMockedSwap(
            DepositWithMockedSwapParams({
                leverageToken: ethShortLeverageToken,
                collateralAsset: USDC,
                debtAsset: WETH,
                userBalanceOfCollateralAsset: collateralFromSenderB,
                collateralFromSender: collateralFromSenderB,
                flashLoanAmount: previewData.debt,
                minShares: 0,
                collateralRequired: previewData.collateral,
                collateralReceivedFromDebtSwap: collateralReceivedFromDebtSwapB
            })
        );

        previewData = leverageRouter.previewDeposit(ethShortLeverageToken, collateralFromSenderC);

        collateralReceivedFromDebtSwapC =
            bound(collateralReceivedFromDebtSwapC, 1, previewData.collateral - collateralFromSenderC);

        _depositWithMockedSwap(
            DepositWithMockedSwapParams({
                leverageToken: ethShortLeverageToken,
                collateralAsset: USDC,
                debtAsset: WETH,
                userBalanceOfCollateralAsset: collateralFromSenderC,
                collateralFromSender: collateralFromSenderC,
                flashLoanAmount: previewData.debt,
                minShares: 0,
                collateralRequired: previewData.collateral,
                collateralReceivedFromDebtSwap: collateralReceivedFromDebtSwapC
            })
        );
    }

    /// @dev In this block price on oracle 3392.292471591441746049801068
    function testFork_deposit_ExceedsSlippage() public {
        uint256 collateralFromSender = 1 ether;
        uint256 collateralToAdd = 2 * collateralFromSender;
        uint256 userBalanceOfCollateralAsset = 4 ether; // User has more than enough assets for the mint of equity
        uint256 flashLoanAmount = 3392.292471e6; // 3392.292471 USDC
        uint256 sharesFromDeposit = 1 ether;
        uint256 minShares = sharesFromDeposit * 0.99715e18 / 1e18; // 0.285% slippage
        uint256 collateralReceivedFromDebtSwap = 0.997140594716559346 ether; // Swap of 3392.292471 USDC

        {
            // Sanity check that LR preview deposit matches test params
            ActionDataV2 memory previewDataFullDeposit =
                leverageRouter.previewDeposit(leverageToken, collateralFromSender);
            assertEq(previewDataFullDeposit.debt, flashLoanAmount);
            assertEq(previewDataFullDeposit.shares, sharesFromDeposit);
            assertEq(previewDataFullDeposit.collateral, collateralToAdd);
            assertEq(previewDataFullDeposit.tokenFee, 0);
            assertEq(previewDataFullDeposit.treasuryFee, 0);
        }

        // The swap results in less collateral than required to get the flash loaned debt amount from a LM deposit, so the debt amount flash loaned
        // needs to be reduced. We reduce it by the percentage delta between the required collateral and the collateral received from the swap
        uint256 deltaPercentage = collateralReceivedFromDebtSwap * 1e18 / (collateralToAdd - collateralFromSender);
        assertEq(deltaPercentage, 0.997140594716559346e18);
        uint256 flashLoanAmountReduced = flashLoanAmount * deltaPercentage / 1e18;
        assertEq(flashLoanAmountReduced, 3382.592531e6);

        // Updated collateral received from the debt swap for lower debt amount
        collateralReceivedFromDebtSwap = 0.994290732650270211 ether;

        // Preview again using the total collateral. This is used by the LM deposit logic
        uint256 totalCollateral = collateralFromSender + collateralReceivedFromDebtSwap;
        assertEq(totalCollateral, 1.994290732650270211 ether);
        ActionDataV2 memory previewData = leverageManager.previewDeposit(leverageToken, totalCollateral);
        assertGe(previewData.debt, flashLoanAmountReduced);
        assertEq(previewData.debt, 3382.608719e6);

        // More than minShares (0.285% slippage) will be minted
        assertLt(previewData.shares, minShares);
        assertEq(previewData.shares, 0.997145366325135105 ether);

        address[] memory path = new address[](2);
        path[0] = address(USDC);
        path[1] = address(WETH);

        ISwapAdapter.SwapContext memory swapContext = ISwapAdapter.SwapContext({
            exchange: ISwapAdapter.Exchange.UNISWAP_V2,
            encodedPath: new bytes(0),
            path: path,
            fees: new uint24[](0),
            tickSpacing: new int24[](0),
            exchangeAddresses: ISwapAdapter.ExchangeAddresses({
                aerodromeRouter: address(0),
                aerodromePoolFactory: address(0),
                aerodromeSlipstreamRouter: address(0),
                uniswapSwapRouter02: address(0),
                uniswapV2Router02: UNISWAP_V2_ROUTER02
            }),
            additionalData: new bytes(0)
        });

        deal(address(WETH), user, userBalanceOfCollateralAsset);
        vm.startPrank(user);
        WETH.approve(address(leverageRouter), collateralFromSender);

        vm.expectRevert(
            abi.encodeWithSelector(ILeverageManager.SlippageTooHigh.selector, 0.997145366325135105 ether, 0.99715 ether)
        );
        leverageRouter.deposit(leverageToken, collateralFromSender, flashLoanAmountReduced, minShares, swapContext);
        vm.stopPrank();
    }

    /// @dev In this block price on oracle 3392.292471591441746049801068
    function testFork_deposit_InsufficientDebtFromDepositToRepayFlashLoan() public {
        uint256 collateralFromSender = 0.01 ether;

        // 2x collateral ratio
        ActionDataV2 memory previewData = leverageRouter.previewDeposit(leverageToken, collateralFromSender);
        assertEq(previewData.collateral, collateralFromSender * 2);
        assertEq(previewData.debt, 33.922924e6);

        address[] memory path = new address[](2);
        path[0] = address(USDC);
        path[1] = address(WETH);

        ISwapAdapter.SwapContext memory swapContext = ISwapAdapter.SwapContext({
            exchange: ISwapAdapter.Exchange.UNISWAP_V2,
            encodedPath: new bytes(0),
            path: path,
            fees: new uint24[](0),
            tickSpacing: new int24[](0),
            exchangeAddresses: ISwapAdapter.ExchangeAddresses({
                aerodromeRouter: address(0),
                aerodromePoolFactory: address(0),
                aerodromeSlipstreamRouter: address(0),
                uniswapSwapRouter02: address(0),
                uniswapV2Router02: UNISWAP_V2_ROUTER02
            }),
            additionalData: new bytes(0)
        });

        // The collateral received from swapping 33.922924e6 USDC is 0.009976155542446272 WETH in this block using Uniswap V2
        uint256 collateralReceivedFromDebtSwap = 0.009976155542446272 ether;

        // The collateral from the swap + the collateral from the sender is less than the collateral required
        uint256 totalCollateral = collateralReceivedFromDebtSwap + collateralFromSender;
        assertLt(totalCollateral, previewData.collateral);

        deal(address(WETH), user, collateralFromSender);
        vm.startPrank(user);
        WETH.approve(address(leverageRouter), collateralFromSender);

        // Reverts when morpho attempts to pull assets to repay the flash loan. The debt amount returned from the deposit is too
        // low because the collateral from the swap + the collateral from the sender is less than the collateral required.
        vm.expectRevert("transferFrom reverted"); // Thrown by morpho
        leverageRouter.deposit(leverageToken, collateralFromSender, previewData.debt, 0, swapContext);
        vm.stopPrank();
    }

    /// @dev In this block price on oracle 3392.292471591441746049801068
    function testFork_deposit_UniswapV3() public {
        uint256 collateralFromSender = 1 ether;
        uint256 collateralToAdd = 2 * collateralFromSender;
        uint256 flashLoanAmount = 3392.292471e6; // 3392.292471 USDC
        uint256 minShares = 1 ether * 0.99e18 / 1e18; // 1% slippage
        uint256 collateralReceivedFromDebtSwap = 0.999899417781964728 ether; // Swap of 3392.292471 USDC

        {
            // Sanity check that LR preview deposit matches test params
            ActionDataV2 memory previewDataFullDeposit =
                leverageRouter.previewDeposit(leverageToken, collateralFromSender);
            assertEq(previewDataFullDeposit.debt, flashLoanAmount);
            assertEq(previewDataFullDeposit.shares, 1 ether);
            assertEq(previewDataFullDeposit.collateral, collateralToAdd);
            assertEq(previewDataFullDeposit.tokenFee, 0);
            assertEq(previewDataFullDeposit.treasuryFee, 0);
        }

        // The swap results in less collateral than required to get the flash loaned debt amount from a LM deposit, so the debt amount flash loaned
        // needs to be reduced. We reduce it by the percentage delta between the required collateral and the collateral received from the swap
        uint256 deltaPercentage = collateralReceivedFromDebtSwap * 1e18 / (collateralToAdd - collateralFromSender);
        assertEq(deltaPercentage, 0.999899417781964728e18);
        uint256 flashLoanAmountReduced = flashLoanAmount * deltaPercentage / 1e18;
        assertEq(flashLoanAmountReduced, 3391.951266e6);

        // Updated collateral received from the debt swap for lower debt amount
        collateralReceivedFromDebtSwap = 0.999798847238411671 ether;

        // Preview again using the total collateral. This is used by the LM deposit logic
        uint256 totalCollateral = collateralFromSender + collateralReceivedFromDebtSwap;
        assertEq(totalCollateral, 1.999798847238411671 ether);
        ActionDataV2 memory previewData = leverageManager.previewDeposit(leverageToken, totalCollateral);
        assertGe(previewData.debt, flashLoanAmountReduced);
        assertEq(previewData.debt, 3391.951287e6);

        // More than minShares (1% slippage) will be minted
        assertGe(previewData.shares, minShares);
        assertEq(previewData.shares, 0.999899423619205835 ether);

        address[] memory path = new address[](2);
        path[0] = address(USDC);
        path[1] = address(WETH);

        uint24[] memory fees = new uint24[](1);
        fees[0] = 500;

        ISwapAdapter.SwapContext memory swapContext = ISwapAdapter.SwapContext({
            exchange: ISwapAdapter.Exchange.UNISWAP_V3,
            encodedPath: SwapPathLib._encodeUniswapV3Path(path, fees, false),
            path: path,
            fees: fees,
            tickSpacing: new int24[](0),
            exchangeAddresses: ISwapAdapter.ExchangeAddresses({
                aerodromeRouter: address(0),
                aerodromePoolFactory: address(0),
                aerodromeSlipstreamRouter: address(0),
                uniswapSwapRouter02: UNISWAP_SWAP_ROUTER02,
                uniswapV2Router02: address(0)
            }),
            additionalData: new bytes(0)
        });

        _dealAndDeposit(
            WETH, USDC, collateralFromSender, collateralFromSender, flashLoanAmountReduced, minShares, swapContext
        );

        // Collateral is taken from the user for the deposit. All of the collateral should be used
        assertEq(WETH.balanceOf(user), 0);

        // Any additional debt that is not used to repay the flash loan is given to the user
        uint256 excessDebt = previewData.debt - flashLoanAmountReduced;
        assertEq(USDC.balanceOf(user), excessDebt);
        assertEq(USDC.balanceOf(user), 0.000021e6);

        assertGe(leverageToken.balanceOf(user), minShares);

        assertEq(morphoLendingAdapter.getCollateral(), totalCollateral);
        assertEq(morphoLendingAdapter.getDebt(), previewData.debt + 1); // + 1 because of rounding up by MorphoBalancesLib.expectedBorrowAssets
    }

    /// @dev In this block price on oracle 3392.292471591441746049801068
    function testFork_deposit_Aerodrome() public {
        uint256 collateralFromSender = 1 ether;
        uint256 collateralToAdd = 2 * collateralFromSender;
        uint256 flashLoanAmount = 3392.292471e6; // 3392.292471 USDC
        uint256 minShares = 1 ether * 0.99e18 / 1e18; // 1% slippage
        uint256 collateralReceivedFromDebtSwap = 0.99780113268167845 ether; // Swap of 3392.292471 USDC

        {
            // Sanity check that LR preview deposit matches test params
            ActionDataV2 memory previewDataFullDeposit =
                leverageRouter.previewDeposit(leverageToken, collateralFromSender);
            assertEq(previewDataFullDeposit.debt, flashLoanAmount);
            assertEq(previewDataFullDeposit.shares, 1 ether);
            assertEq(previewDataFullDeposit.collateral, collateralToAdd);
            assertEq(previewDataFullDeposit.tokenFee, 0);
            assertEq(previewDataFullDeposit.treasuryFee, 0);
        }

        // The swap results in less collateral than required to get the flash loaned debt amount from a LM deposit, so the debt amount flash loaned
        // needs to be reduced. We reduce it by the percentage delta between the required collateral and the collateral received from the swap
        uint256 deltaPercentage = collateralReceivedFromDebtSwap * 1e18 / (collateralToAdd - collateralFromSender);
        assertEq(deltaPercentage, 0.99780113268167845e18);
        uint256 flashLoanAmountReduced = flashLoanAmount * deltaPercentage / 1e18;
        assertEq(flashLoanAmountReduced, 3384.833269e6);

        // Updated collateral received from the debt swap for lower debt amount
        collateralReceivedFromDebtSwap = 0.995607717905650985 ether;

        // Preview again using the total collateral. This is used by the LM deposit logic
        uint256 totalCollateral = collateralFromSender + collateralReceivedFromDebtSwap;
        assertEq(totalCollateral, 1.995607717905650985 ether);
        ActionDataV2 memory previewData = leverageManager.previewDeposit(leverageToken, totalCollateral);
        assertGe(previewData.debt, flashLoanAmountReduced);
        assertEq(previewData.debt, 3384.842518e6);

        // More than minShares (1% slippage) will be minted
        assertGe(previewData.shares, minShares);
        assertEq(previewData.shares, 0.997803858952825492 ether);

        address[] memory path = new address[](2);
        path[0] = address(USDC);
        path[1] = address(WETH);

        ISwapAdapter.SwapContext memory swapContext = ISwapAdapter.SwapContext({
            exchange: ISwapAdapter.Exchange.AERODROME,
            encodedPath: new bytes(0),
            path: path,
            fees: new uint24[](0),
            tickSpacing: new int24[](0),
            exchangeAddresses: ISwapAdapter.ExchangeAddresses({
                aerodromeRouter: AERODROME_ROUTER,
                aerodromePoolFactory: AERODROME_POOL_FACTORY,
                aerodromeSlipstreamRouter: address(0),
                uniswapSwapRouter02: address(0),
                uniswapV2Router02: address(0)
            }),
            additionalData: new bytes(0)
        });

        _dealAndDeposit(
            WETH, USDC, collateralFromSender, collateralFromSender, flashLoanAmountReduced, minShares, swapContext
        );

        // Collateral is taken from the user for the deposit. All of the collateral should be used
        assertEq(WETH.balanceOf(user), 0);

        // Any additional debt that is not used to repay the flash loan is given to the user
        uint256 excessDebt = previewData.debt - flashLoanAmountReduced;
        assertEq(USDC.balanceOf(user), excessDebt);
        assertEq(USDC.balanceOf(user), 0.009249e6);

        assertGe(leverageToken.balanceOf(user), minShares);

        assertEq(morphoLendingAdapter.getCollateral(), totalCollateral);
        assertEq(morphoLendingAdapter.getDebt(), previewData.debt + 1); // + 1 because of rounding up by MorphoBalancesLib.expectedBorrowAssets
    }

    /// @dev In this block price on oracle 3392.292471591441746049801068
    function testFork_deposit_AerodromeSlipstream() public {
        uint256 collateralFromSender = 1 ether;
        uint256 collateralToAdd = 2 * collateralFromSender;
        uint256 flashLoanAmount = 3392.292471e6; // 3392.292471 USDC
        uint256 minShares = 1 ether * 0.99e18 / 1e18; // 1% slippage

        // Swap is favorable with slipstream, results in more than required collateral
        uint256 collateralReceivedFromDebtSwap = 1.00009355883189593 ether; // Swap of 3392.292471 USDC

        uint256 totalCollateral = collateralFromSender + collateralReceivedFromDebtSwap;

        {
            // Sanity check that LR preview deposit matches test params
            ActionDataV2 memory previewDataFullDeposit =
                leverageRouter.previewDeposit(leverageToken, collateralFromSender);
            assertEq(previewDataFullDeposit.debt, flashLoanAmount);
            assertEq(previewDataFullDeposit.shares, 1 ether);
            assertEq(previewDataFullDeposit.collateral, collateralToAdd);
            assertEq(previewDataFullDeposit.tokenFee, 0);
            assertEq(previewDataFullDeposit.treasuryFee, 0);
        }

        // Preview again using the total collateral. This is used by the LM deposit logic
        assertEq(totalCollateral, 2.00009355883189593 ether);
        ActionDataV2 memory previewData = leverageManager.previewDeposit(leverageToken, totalCollateral);
        assertGe(previewData.debt, flashLoanAmount);
        assertEq(previewData.debt, 3392.451161e6);

        address[] memory path = new address[](2);
        path[0] = address(USDC);
        path[1] = address(WETH);

        int24[] memory tickSpacing = new int24[](1);
        tickSpacing[0] = 100;

        ISwapAdapter.SwapContext memory swapContext = ISwapAdapter.SwapContext({
            exchange: ISwapAdapter.Exchange.AERODROME_SLIPSTREAM,
            encodedPath: SwapPathLib._encodeAerodromeSlipstreamPath(path, tickSpacing, false),
            path: path,
            fees: new uint24[](0),
            tickSpacing: tickSpacing,
            exchangeAddresses: ISwapAdapter.ExchangeAddresses({
                aerodromeRouter: address(0),
                aerodromePoolFactory: address(0),
                aerodromeSlipstreamRouter: AERODROME_SLIPSTREAM_ROUTER,
                uniswapSwapRouter02: address(0),
                uniswapV2Router02: address(0)
            }),
            additionalData: new bytes(0)
        });

        _dealAndDeposit(WETH, USDC, collateralFromSender, collateralFromSender, flashLoanAmount, minShares, swapContext);

        // Collateral is taken from the user for the deposit. All of the collateral should be used
        assertEq(WETH.balanceOf(user), 0);

        // Any additional debt that is not used to repay the flash loan is given to the user
        uint256 excessDebt = previewData.debt - flashLoanAmount;
        assertEq(USDC.balanceOf(user), excessDebt);
        assertEq(USDC.balanceOf(user), 0.15869e6);

        assertGe(leverageToken.balanceOf(user), minShares);

        assertEq(morphoLendingAdapter.getCollateral(), totalCollateral);
        assertEq(morphoLendingAdapter.getDebt(), previewData.debt + 1); // + 1 because of rounding up by MorphoBalancesLib.expectedBorrowAssets
    }

    /// @dev In this block price on oracle 3392.292471591441746049801068
    function testFork_deposit_UniswapV2_MultiHop() public {
        uint256 collateralFromSender = 1 ether;
        uint256 collateralToAdd = 2 * collateralFromSender;
        uint256 userBalanceOfCollateralAsset = 4 ether; // User has more than enough assets for the mint of equity
        uint256 flashLoanAmount = 3392.292471e6; // 3392.292471 USDC
        uint256 minShares = 1 ether * 0.99e18 / 1e18; // 1% slippage
        uint256 collateralReceivedFromDebtSwap = 0.003436017464761568 ether; // Swap of 3392.292471 USDC

        {
            // Sanity check that LR preview deposit matches test params
            ActionDataV2 memory previewDataFullDeposit =
                leverageRouter.previewDeposit(leverageToken, collateralFromSender);
            assertEq(previewDataFullDeposit.debt, flashLoanAmount);
            assertEq(previewDataFullDeposit.shares, 1 ether);
            assertEq(previewDataFullDeposit.collateral, collateralToAdd);
            assertEq(previewDataFullDeposit.tokenFee, 0);
            assertEq(previewDataFullDeposit.treasuryFee, 0);
        }

        // The swap results in less collateral than required to get the flash loaned debt amount from a LM deposit, so the debt amount flash loaned
        // needs to be reduced. We reduce it by the percentage delta between the required collateral and the collateral received from the swap
        uint256 deltaPercentage = collateralReceivedFromDebtSwap * 1e18 / (collateralToAdd - collateralFromSender);
        assertEq(deltaPercentage, 0.003436017464761568e18);
        uint256 flashLoanAmountReduced = flashLoanAmount * deltaPercentage / 1e18;
        assertEq(flashLoanAmountReduced, 11.655976e6);

        // Updated collateral received from the debt swap for lower debt amount
        collateralReceivedFromDebtSwap = 0.001720627030031886 ether;

        // Preview again using the total collateral. This is used by the LM deposit logic
        uint256 totalCollateral = collateralFromSender + collateralReceivedFromDebtSwap;
        assertEq(totalCollateral, 1.001720627030031886 ether);
        ActionDataV2 memory previewData = leverageManager.previewDeposit(leverageToken, totalCollateral);
        assertGe(previewData.debt, flashLoanAmountReduced);
        assertEq(previewData.debt, 1699.06467e6);

        // Less than minShares (1% slippage) will be minted
        assertLt(previewData.shares, minShares);
        assertEq(previewData.shares, 0.500860313515015943 ether);

        address[] memory path = new address[](3);
        path[0] = address(USDC);
        path[1] = address(DAI);
        path[2] = address(WETH);

        ISwapAdapter.SwapContext memory swapContext = ISwapAdapter.SwapContext({
            exchange: ISwapAdapter.Exchange.UNISWAP_V2,
            encodedPath: new bytes(0),
            path: path,
            fees: new uint24[](0),
            tickSpacing: new int24[](0),
            exchangeAddresses: ISwapAdapter.ExchangeAddresses({
                aerodromeRouter: address(0),
                aerodromePoolFactory: address(0),
                aerodromeSlipstreamRouter: address(0),
                uniswapSwapRouter02: address(0),
                uniswapV2Router02: UNISWAP_V2_ROUTER02
            }),
            additionalData: new bytes(0)
        });

        deal(address(WETH), user, userBalanceOfCollateralAsset);
        vm.startPrank(user);
        WETH.approve(address(leverageRouter), collateralFromSender);

        vm.expectRevert(
            abi.encodeWithSelector(ILeverageManager.SlippageTooHigh.selector, 0.500860313515015943 ether, minShares)
        );
        leverageRouter.deposit(leverageToken, collateralFromSender, flashLoanAmountReduced, minShares, swapContext);

        // If we update minShares, successful
        leverageRouter.deposit(
            leverageToken, collateralFromSender, flashLoanAmountReduced, previewData.shares, swapContext
        );
        vm.stopPrank();

        // Collateral is taken from the user for the deposit. All of the collateral should be used
        assertEq(WETH.balanceOf(user), userBalanceOfCollateralAsset - collateralFromSender);

        // Any additional debt that is not used to repay the flash loan is given to the user
        uint256 excessDebt = previewData.debt - flashLoanAmountReduced;
        assertEq(USDC.balanceOf(user), excessDebt);
        assertEq(USDC.balanceOf(user), 1687.408694e6);

        assertEq(leverageToken.balanceOf(user), previewData.shares);

        assertEq(morphoLendingAdapter.getCollateral(), totalCollateral);
        assertEq(morphoLendingAdapter.getDebt(), previewData.debt + 1); // + 1 because of rounding up by MorphoBalancesLib.expectedBorrowAssets
    }

    /// @dev In this block price on oracle 3392.292471591441746049801068
    function testFork_deposit_UniswapV3_MultiHop() public {
        uint256 collateralFromSender = 1 ether;
        uint256 collateralToAdd = 2 * collateralFromSender;
        uint256 flashLoanAmount = 3392.292471e6; // 3392.292471 USDC
        uint256 minShares = 1 ether * 0.85e18 / 1e18; // 15% slippage
        uint256 collateralReceivedFromDebtSwap = 0.730785046551638276 ether; // Swap of 3392.292471 USDC

        {
            // Sanity check that LR preview deposit matches test params
            ActionDataV2 memory previewDataFullDeposit =
                leverageRouter.previewDeposit(leverageToken, collateralFromSender);
            assertEq(previewDataFullDeposit.debt, flashLoanAmount);
            assertEq(previewDataFullDeposit.shares, 1 ether);
            assertEq(previewDataFullDeposit.collateral, collateralToAdd);
            assertEq(previewDataFullDeposit.tokenFee, 0);
            assertEq(previewDataFullDeposit.treasuryFee, 0);
        }

        // The swap results in less collateral than required to get the flash loaned debt amount from a LM deposit, so the debt amount flash loaned
        // needs to be reduced. We reduce it by the percentage delta between the required collateral and the collateral received from the swap
        uint256 deltaPercentage = collateralReceivedFromDebtSwap * 1e18 / (collateralToAdd - collateralFromSender);
        assertEq(deltaPercentage, 0.730785046551638276e18);
        uint256 flashLoanAmountReduced = flashLoanAmount * deltaPercentage / 1e18;
        assertEq(flashLoanAmountReduced, 2479.036611e6);

        // Updated collateral received from the debt swap for lower debt amount
        collateralReceivedFromDebtSwap = 0.719360769453766291 ether;

        // Preview again using the total collateral. This is used by the LM deposit logic
        uint256 totalCollateral = collateralFromSender + collateralReceivedFromDebtSwap;
        assertEq(totalCollateral, 1.719360769453766291 ether);
        ActionDataV2 memory previewData = leverageManager.previewDeposit(leverageToken, totalCollateral);
        assertGe(previewData.debt, flashLoanAmountReduced);
        assertEq(previewData.debt, 2916.287297e6);

        // Greater than minShares (15% slippage) will be minted
        assertGe(previewData.shares, minShares);
        assertEq(previewData.shares, 0.859680384726883145 ether);

        address[] memory path = new address[](3);
        path[0] = address(USDC);
        path[1] = address(DAI);
        path[2] = address(WETH);

        uint24[] memory fees = new uint24[](2);
        fees[0] = 500;
        fees[1] = 500;

        ISwapAdapter.SwapContext memory swapContext = ISwapAdapter.SwapContext({
            exchange: ISwapAdapter.Exchange.UNISWAP_V3,
            encodedPath: SwapPathLib._encodeUniswapV3Path(path, fees, false),
            path: path,
            fees: fees,
            tickSpacing: new int24[](0),
            exchangeAddresses: ISwapAdapter.ExchangeAddresses({
                aerodromeRouter: address(0),
                aerodromePoolFactory: address(0),
                aerodromeSlipstreamRouter: address(0),
                uniswapSwapRouter02: UNISWAP_SWAP_ROUTER02,
                uniswapV2Router02: address(0)
            }),
            additionalData: new bytes(0)
        });

        _dealAndDeposit(
            WETH, USDC, collateralFromSender, collateralFromSender, flashLoanAmountReduced, minShares, swapContext
        );

        // Collateral is taken from the user for the deposit. All of the collateral should be used
        assertEq(WETH.balanceOf(user), 0);

        // Any additional debt that is not used to repay the flash loan is given to the user
        uint256 excessDebt = previewData.debt - flashLoanAmountReduced;
        assertEq(USDC.balanceOf(user), excessDebt);
        assertEq(USDC.balanceOf(user), 437.250686e6);

        assertEq(leverageToken.balanceOf(user), previewData.shares);

        assertEq(morphoLendingAdapter.getCollateral(), totalCollateral);
        assertEq(morphoLendingAdapter.getDebt(), previewData.debt + 1); // + 1 because of rounding up by MorphoBalancesLib.expectedBorrowAssets
    }

    /// @dev In this block price on oracle 3392.292471591441746049801068
    function testFork_deposit_Aerodrome_MultiHop() public {
        uint256 collateralFromSender = 1 ether;
        uint256 collateralToAdd = 2 * collateralFromSender;
        uint256 flashLoanAmount = 3392.292471e6; // 3392.292471 USDC
        uint256 minShares = 1 ether * 0.99e18 / 1e18; // 1% slippage
        uint256 collateralReceivedFromDebtSwap = 0.001479490022113963 ether; // Swap of 3392.292471 USDC

        {
            // Sanity check that LR preview deposit matches test params
            ActionDataV2 memory previewDataFullDeposit =
                leverageRouter.previewDeposit(leverageToken, collateralFromSender);
            assertEq(previewDataFullDeposit.debt, flashLoanAmount);
            assertEq(previewDataFullDeposit.shares, 1 ether);
            assertEq(previewDataFullDeposit.collateral, collateralToAdd);
            assertEq(previewDataFullDeposit.tokenFee, 0);
            assertEq(previewDataFullDeposit.treasuryFee, 0);
        }

        // The swap results in less collateral than required to get the flash loaned debt amount from a LM deposit, so the debt amount flash loaned
        // needs to be reduced. We reduce it by the percentage delta between the required collateral and the collateral received from the swap
        uint256 deltaPercentage = collateralReceivedFromDebtSwap * 1e18 / (collateralToAdd - collateralFromSender);
        assertEq(deltaPercentage, 0.001479490022113963e18);
        uint256 flashLoanAmountReduced = flashLoanAmount * deltaPercentage / 1e18;
        assertEq(flashLoanAmountReduced, 5.018862e6);

        // Updated collateral received from the debt swap for lower debt amount
        collateralReceivedFromDebtSwap = 0.000737563974906262 ether;

        // Preview again using the total collateral. This is used by the LM deposit logic
        uint256 totalCollateral = collateralFromSender + collateralReceivedFromDebtSwap;
        assertEq(totalCollateral, 1.000737563974906262 ether);
        ActionDataV2 memory previewData = leverageManager.previewDeposit(leverageToken, totalCollateral);
        assertGe(previewData.debt, flashLoanAmountReduced);
        assertEq(previewData.debt, 1697.397252e6);

        // Less than minShares (1% slippage) will be minted
        assertLt(previewData.shares, minShares);
        assertEq(previewData.shares, 0.500368781987453131 ether);

        address[] memory path = new address[](3);
        path[0] = address(USDC);
        path[1] = address(DAI);
        path[2] = address(WETH);

        ISwapAdapter.SwapContext memory swapContext = ISwapAdapter.SwapContext({
            exchange: ISwapAdapter.Exchange.AERODROME,
            encodedPath: new bytes(0),
            path: path,
            fees: new uint24[](0),
            tickSpacing: new int24[](0),
            exchangeAddresses: ISwapAdapter.ExchangeAddresses({
                aerodromeRouter: AERODROME_ROUTER,
                aerodromePoolFactory: AERODROME_POOL_FACTORY,
                aerodromeSlipstreamRouter: address(0),
                uniswapSwapRouter02: address(0),
                uniswapV2Router02: address(0)
            }),
            additionalData: new bytes(0)
        });

        deal(address(WETH), user, collateralFromSender);
        vm.startPrank(user);
        WETH.approve(address(leverageRouter), collateralFromSender);

        vm.expectRevert(
            abi.encodeWithSelector(ILeverageManager.SlippageTooHigh.selector, 0.500368781987453131 ether, minShares)
        );
        leverageRouter.deposit(leverageToken, collateralFromSender, flashLoanAmountReduced, minShares, swapContext);

        // If we update minShares, successful
        leverageRouter.deposit(
            leverageToken, collateralFromSender, flashLoanAmountReduced, previewData.shares, swapContext
        );
        vm.stopPrank();

        // Collateral is taken from the user for the deposit. All of the collateral should be used
        assertEq(WETH.balanceOf(user), 0);

        // Any additional debt that is not used to repay the flash loan is given to the user
        uint256 excessDebt = previewData.debt - flashLoanAmountReduced;
        assertEq(USDC.balanceOf(user), excessDebt);
        assertEq(USDC.balanceOf(user), 1692.37839e6);

        assertEq(leverageToken.balanceOf(user), previewData.shares);

        assertEq(morphoLendingAdapter.getCollateral(), totalCollateral);
        assertEq(morphoLendingAdapter.getDebt(), previewData.debt + 1); // + 1 because of rounding up by MorphoBalancesLib.expectedBorrowAssets
    }

    /// @dev In this block price on oracle 3392.292471591441746049801068
    function testFork_deposit_AerodromeSlipstream_MultiHop() public {
        uint256 collateralFromSender = 1 ether;
        uint256 collateralToAdd = 2 * collateralFromSender;
        uint256 flashLoanAmount = 3392.292471e6; // 3392.292471 USDC
        uint256 minShares = 1 ether * 0.99e18 / 1e18; // 1% slippage
        uint256 collateralReceivedFromDebtSwap = 0.999075127525769712 ether; // Swap of 3392.292471 USDC

        {
            // Sanity check that LR preview deposit matches test params
            ActionDataV2 memory previewDataFullDeposit =
                leverageRouter.previewDeposit(leverageToken, collateralFromSender);
            assertEq(previewDataFullDeposit.debt, flashLoanAmount);
            assertEq(previewDataFullDeposit.shares, 1 ether);
            assertEq(previewDataFullDeposit.collateral, collateralToAdd);
            assertEq(previewDataFullDeposit.tokenFee, 0);
            assertEq(previewDataFullDeposit.treasuryFee, 0);
        }

        // The swap results in less collateral than required to get the flash loaned debt amount from a LM deposit, so the debt amount flash loaned
        // needs to be reduced. We reduce it by the percentage delta between the required collateral and the collateral received from the swap
        uint256 deltaPercentage = collateralReceivedFromDebtSwap * 1e18 / (collateralToAdd - collateralFromSender);
        assertEq(deltaPercentage, 0.999075127525769712e18);
        uint256 flashLoanAmountReduced = flashLoanAmount * deltaPercentage / 1e18;
        assertEq(flashLoanAmountReduced, 3389.155033e6);

        // Updated collateral received from the debt swap for lower debt amount
        collateralReceivedFromDebtSwap = 0.998151321850066641 ether;

        // Preview again using the total collateral. This is used by the LM deposit logic
        uint256 totalCollateral = collateralFromSender + collateralReceivedFromDebtSwap;
        assertEq(totalCollateral, 1.998151321850066641 ether);
        ActionDataV2 memory previewData = leverageManager.previewDeposit(leverageToken, totalCollateral);
        assertGe(previewData.debt, flashLoanAmountReduced);
        assertEq(previewData.debt, 3389.156843e6);

        // Greater than minShares (1% slippage) will be minted
        assertGe(previewData.shares, minShares);
        assertEq(previewData.shares, 0.99907566092503332 ether);

        address[] memory path = new address[](3);
        path[0] = address(USDC);
        path[1] = address(cbBTC);
        path[2] = address(WETH);

        int24[] memory tickSpacing = new int24[](2);
        tickSpacing[0] = 100;
        tickSpacing[1] = 100;

        ISwapAdapter.SwapContext memory swapContext = ISwapAdapter.SwapContext({
            exchange: ISwapAdapter.Exchange.AERODROME_SLIPSTREAM,
            encodedPath: SwapPathLib._encodeAerodromeSlipstreamPath(path, tickSpacing, false),
            path: path,
            fees: new uint24[](0),
            tickSpacing: tickSpacing,
            exchangeAddresses: ISwapAdapter.ExchangeAddresses({
                aerodromeRouter: address(0),
                aerodromePoolFactory: address(0),
                aerodromeSlipstreamRouter: AERODROME_SLIPSTREAM_ROUTER,
                uniswapSwapRouter02: address(0),
                uniswapV2Router02: address(0)
            }),
            additionalData: new bytes(0)
        });

        _dealAndDeposit(
            WETH, USDC, collateralFromSender, collateralFromSender, flashLoanAmountReduced, minShares, swapContext
        );

        // Collateral is taken from the user for the deposit. All of the collateral should be used
        assertEq(WETH.balanceOf(user), 0);

        // Any additional debt that is not used to repay the flash loan is given to the user
        uint256 excessDebt = previewData.debt - flashLoanAmountReduced;
        assertEq(USDC.balanceOf(user), excessDebt);
        assertEq(USDC.balanceOf(user), 0.00181e6);

        assertEq(leverageToken.balanceOf(user), previewData.shares);

        assertEq(morphoLendingAdapter.getCollateral(), totalCollateral);
        assertEq(morphoLendingAdapter.getDebt(), previewData.debt + 1); // + 1 because of rounding up by MorphoBalancesLib.expectedBorrowAssets
    }

    function _depositWithMockedSwap(DepositWithMockedSwapParams memory params) internal {
        uint256 collateralToAdd =
            leverageRouter.previewDeposit(params.leverageToken, params.collateralFromSender).collateral;

        uint256 flashLoanAmountReduced = params.flashLoanAmount;
        if (params.collateralReceivedFromDebtSwap < collateralToAdd - params.collateralFromSender) {
            // The swap results in less collateral than required to get the flash loaned debt amount from a LM deposit, so the debt amount flash loaned
            // needs to be reduced. We reduce it by the percentage delta between the required collateral and the collateral received from the swap
            uint256 deltaPercentage =
                params.collateralReceivedFromDebtSwap * 1e18 / (collateralToAdd - params.collateralFromSender);
            flashLoanAmountReduced = params.flashLoanAmount * deltaPercentage / 1e18;
        }

        if (flashLoanAmountReduced == 0) {
            return;
        }

        // Mock the swap of the debt asset to the collateral asset to be the required amount
        uint256 collateralReceivedFromReducedDebtSwap = params.collateralRequired > params.collateralFromSender
            ? params.collateralRequired - params.collateralFromSender
            : 0;

        // The entire amount of collateral is used for the deposit
        uint256 collateralUsedForDeposit = params.collateralFromSender + collateralReceivedFromReducedDebtSwap;
        uint256 debtFromDeposit = leverageManager.previewDeposit(params.leverageToken, collateralUsedForDeposit).debt;

        mockSwapper.mockNextExactInputSwap(
            params.debtAsset, params.collateralAsset, collateralReceivedFromReducedDebtSwap
        );

        {
            address[] memory path = new address[](2);
            path[0] = address(params.debtAsset);
            path[1] = address(params.collateralAsset);

            ISwapAdapter.SwapContext memory swapContext = ISwapAdapter.SwapContext({
                exchange: ISwapAdapter.Exchange.UNISWAP_V2,
                encodedPath: new bytes(0),
                path: path,
                fees: new uint24[](0),
                tickSpacing: new int24[](0),
                exchangeAddresses: ISwapAdapter.ExchangeAddresses({
                    aerodromeRouter: address(0),
                    aerodromePoolFactory: address(0),
                    aerodromeSlipstreamRouter: address(0),
                    uniswapSwapRouter02: address(0),
                    uniswapV2Router02: UNISWAP_V2_ROUTER02
                }),
                additionalData: new bytes(0)
            });

            deal(address(params.collateralAsset), user, params.userBalanceOfCollateralAsset);

            vm.startPrank(user);
            params.collateralAsset.approve(address(leverageRouterWithMockSwapAdapter), params.collateralFromSender);
            leverageRouterWithMockSwapAdapter.deposit(
                params.leverageToken, params.collateralFromSender, flashLoanAmountReduced, params.minShares, swapContext
            );
            vm.stopPrank();
        }

        // No leftover assets in the LR
        assertEq(params.collateralAsset.balanceOf(address(leverageRouterWithMockSwapAdapter)), 0);
        assertEq(params.debtAsset.balanceOf(address(leverageRouterWithMockSwapAdapter)), 0);

        // Collateral is taken from the user for the deposit. All of the collateral should be used
        assertEq(params.collateralAsset.balanceOf(user), 0);

        // Any additional debt that is not used to repay the flash loan is given to the user
        uint256 excessDebt = debtFromDeposit - flashLoanAmountReduced;
        assertEq(params.debtAsset.balanceOf(user), excessDebt);
        // Transfer any excess debt away for multiple uses/iterations of the user debt balance assertion above within a single test
        if (excessDebt > 0) {
            vm.prank(user);
            params.debtAsset.transfer(address(this), excessDebt);
        }
    }

    function _supplyWETHForETHShortLeverageToken(uint256 amount) internal {
        deal(address(WETH), address(this), amount);
        IMorpho morpho = IMorpho(ethShortLendingAdapter.morpho());

        (address loanToken, address collateralToken, address oracle, address irm, uint256 lltv) =
            ethShortLendingAdapter.marketParams();
        MarketParams memory marketParams =
            MarketParams({loanToken: loanToken, collateralToken: collateralToken, oracle: oracle, irm: irm, lltv: lltv});

        WETH.approve(address(morpho), amount);
        morpho.supply(marketParams, amount, 0, address(this), new bytes(0));
    }

    function _supplyUSDCForETHLongLeverageToken(uint256 amount) internal {
        deal(address(USDC), address(this), amount);
        IMorpho morpho = IMorpho(morphoLendingAdapter.morpho());

        (address loanToken, address collateralToken, address oracle, address irm, uint256 lltv) =
            morphoLendingAdapter.marketParams();
        MarketParams memory marketParams =
            MarketParams({loanToken: loanToken, collateralToken: collateralToken, oracle: oracle, irm: irm, lltv: lltv});

        USDC.approve(address(morpho), amount);
        morpho.supply(marketParams, amount, 0, address(morphoLendingAdapter), new bytes(0));
    }
}
