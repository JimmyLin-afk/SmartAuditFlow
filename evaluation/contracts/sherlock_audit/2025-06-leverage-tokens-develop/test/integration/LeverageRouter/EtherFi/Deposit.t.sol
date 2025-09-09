// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.26;

// Dependency imports
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {Id} from "@morpho-blue/interfaces/IMorpho.sol";

// Internal imports
import {MorphoLendingAdapter} from "src/lending/MorphoLendingAdapter.sol";
import {RebalanceAdapter} from "src/rebalance/RebalanceAdapter.sol";
import {ILendingAdapter} from "src/interfaces/ILendingAdapter.sol";
import {IRebalanceAdapter} from "src/interfaces/IRebalanceAdapter.sol";
import {ISwapAdapter} from "src/interfaces/periphery/ISwapAdapter.sol";
import {IEtherFiL2ModeSyncPool} from "src/interfaces/periphery/IEtherFiL2ModeSyncPool.sol";
import {IEtherFiL2ExchangeRateProvider} from "src/interfaces/periphery/IEtherFiL2ExchangeRateProvider.sol";
import {ActionDataV2, LeverageTokenConfig} from "src/types/DataTypes.sol";
import {LeverageRouterTest} from "../LeverageRouter.t.sol";

contract LeverageRouterDepositEtherFiTest is LeverageRouterTest {
    /// @notice The ETH address per the EtherFi L2 Mode Sync Pool contract
    address internal constant ETH_ADDRESS = 0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE;

    IERC20 public constant WEETH = IERC20(0x04C0599Ae5A44757c0af6F9eC3b93da8976c150A);

    IEtherFiL2ModeSyncPool public constant etherFiL2ModeSyncPool =
        IEtherFiL2ModeSyncPool(0xc38e046dFDAdf15f7F56853674242888301208a5);

    IEtherFiL2ExchangeRateProvider public constant etherFiL2ExchangeRateProvider =
        IEtherFiL2ExchangeRateProvider(0xF2c5519c634796B73dE90c7Dc27B4fEd560fC3ca);

    Id public constant WEETH_WETH_MARKET_ID =
        Id.wrap(0x78d11c03944e0dc298398f0545dc8195ad201a18b0388cb8058b1bcb89440971);

    function setUp() public override {
        super.setUp();

        morphoLendingAdapter = MorphoLendingAdapter(
            address(morphoLendingAdapterFactory.deployAdapter(WEETH_WETH_MARKET_ID, address(this), bytes32("1")))
        );

        rebalanceAdapterImplementation = new RebalanceAdapter();
        rebalanceAdapter = _deployRebalanceAdapter(1.5e18, 2e18, 2.5e18, 7 minutes, 1.2e18, 0.9e18, 1.2e18, 40_00);

        leverageToken = leverageManager.createNewLeverageToken(
            LeverageTokenConfig({
                lendingAdapter: ILendingAdapter(address(morphoLendingAdapter)),
                rebalanceAdapter: IRebalanceAdapter(address(rebalanceAdapter)),
                mintTokenFee: 0,
                redeemTokenFee: 0
            }),
            "Seamless WEETH/WETH 2x leverage token",
            "ltWEETH/WETH-2x"
        );
    }

    function testFork_setUp() public view virtual override {
        assertEq(address(leverageManager.getLeverageTokenCollateralAsset(leverageToken)), address(WEETH));
        assertEq(address(leverageManager.getLeverageTokenDebtAsset(leverageToken)), address(WETH));

        assertEq(address(leverageRouter.leverageManager()), address(leverageManager));
        assertEq(address(leverageRouter.morpho()), address(MORPHO));
    }

    function testFork_Deposit() public {
        uint256 collateralFromSender = 1 ether;
        uint256 collateralToAdd = 2 * collateralFromSender;
        uint256 userBalanceOfCollateralAsset = 4 ether; // User has more than enough assets for the mint of equity
        uint256 debt = leverageRouter.previewDeposit(leverageToken, collateralFromSender).debt;

        _dealAndDeposit(WEETH, userBalanceOfCollateralAsset, collateralFromSender, debt);

        // Initial mint results in 1:1 shares to equity
        assertEq(leverageToken.balanceOf(user), collateralFromSender);
        // Collateral is taken from the user for the mint
        assertEq(WEETH.balanceOf(user), userBalanceOfCollateralAsset - collateralFromSender);

        assertEq(morphoLendingAdapter.getCollateral(), collateralToAdd);
        // 1.058332450654038384 WETH (WEETH to WETH is not 1:1)
        assertEq(morphoLendingAdapter.getDebt(), 1_058332450654038384);

        // No leftover assets in the LeverageRouter
        assertEq(WEETH.balanceOf(address(leverageRouter)), 0);
        assertEq(WETH.balanceOf(address(leverageRouter)), 0);
        assertEq(address(leverageRouter).balance, 0);
    }

    function testFuzzFork_Deposit(uint256 collateralFromSender) public {
        collateralFromSender = bound(collateralFromSender, 1 ether, 500 ether);

        ActionDataV2 memory previewData = leverageRouter.previewDeposit(leverageToken, collateralFromSender);

        uint256 expectedWeEthFromDebtSwap =
            etherFiL2ExchangeRateProvider.getConversionAmount(ETH_ADDRESS, previewData.debt);

        if (expectedWeEthFromDebtSwap + collateralFromSender < previewData.collateral) {
            collateralFromSender += 100;
        }

        ActionDataV2 memory previewDataFullDeposit =
            leverageManager.previewDeposit(leverageToken, collateralFromSender + expectedWeEthFromDebtSwap);

        _dealAndDeposit(WEETH, collateralFromSender, collateralFromSender, previewData.debt);

        // All collateral is used for the deposit
        assertEq(WEETH.balanceOf(user), 0);
        assertEq(WEETH.balanceOf(address(leverageRouter)), 0);

        // User receives shares and surplus debt
        assertEq(leverageToken.balanceOf(user), previewDataFullDeposit.shares);
        assertEq(WETH.balanceOf(user), previewDataFullDeposit.debt - previewData.debt);
        assertEq(WETH.balanceOf(address(leverageRouter)), 0);
    }

    function _dealAndDeposit(IERC20 collateralAsset, uint256 dealAmount, uint256 collateralFromSender, uint256 debt)
        internal
    {
        deal(address(collateralAsset), user, dealAmount);

        ISwapAdapter.EtherFiSwapContext memory etherFiSwapContext = ISwapAdapter.EtherFiSwapContext({
            etherFiL2ModeSyncPool: IEtherFiL2ModeSyncPool(address(etherFiL2ModeSyncPool)),
            tokenIn: ETH_ADDRESS,
            weETH: address(WEETH),
            referral: address(0)
        });

        ISwapAdapter.SwapContext memory swapContext = ISwapAdapter.SwapContext({
            path: new address[](0),
            encodedPath: new bytes(0),
            fees: new uint24[](0),
            tickSpacing: new int24[](0),
            exchange: ISwapAdapter.Exchange.ETHERFI,
            exchangeAddresses: ISwapAdapter.ExchangeAddresses({
                aerodromeRouter: address(0),
                aerodromePoolFactory: address(0),
                aerodromeSlipstreamRouter: address(0),
                uniswapSwapRouter02: address(0),
                uniswapV2Router02: address(0)
            }),
            additionalData: abi.encode(etherFiSwapContext)
        });

        vm.startPrank(user);
        collateralAsset.approve(address(leverageRouter), collateralFromSender);
        leverageRouter.deposit(leverageToken, collateralFromSender, debt, 0, swapContext);
        vm.stopPrank();

        // No leftover assets in the LeverageRouter
        assertEq(collateralAsset.balanceOf(address(leverageRouter)), 0);
    }
}
