// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.26;

// Dependency imports
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";

// Internal imports
import {IEtherFiL2ModeSyncPool} from "src/interfaces/periphery/IEtherFiL2ModeSyncPool.sol";
import {ISwapAdapter} from "src/interfaces/periphery/ISwapAdapter.sol";
import {SwapAdapterTest} from "./SwapAdapter.t.sol";
import {MockEtherFiL2ModeSyncPool} from "test/unit/mock/MockEtherFiL2ModeSyncPool.sol";
import {MockWETH} from "test/unit/mock/MockWETH.sol";

contract SwapExactInputEtherFiTest is SwapAdapterTest {
    MockEtherFiL2ModeSyncPool public etherFiL2ModeSyncPool;

    MockWETH public weth;
    IERC20 public weEth;

    function setUp() public override {
        super.setUp();

        weEth = toToken;
        etherFiL2ModeSyncPool = new MockEtherFiL2ModeSyncPool(toToken);
        weth = new MockWETH();
    }

    function test_SwapExactInputEtherFi() public {
        uint256 inputAmount = 100 ether;
        uint256 minOutputAmount = 10 ether;

        ISwapAdapter.EtherFiSwapContext memory etherFiSwapContext = ISwapAdapter.EtherFiSwapContext({
            etherFiL2ModeSyncPool: IEtherFiL2ModeSyncPool(address(etherFiL2ModeSyncPool)),
            tokenIn: etherFiL2ModeSyncPool.ETH_ADDRESS(),
            weETH: address(toToken),
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

        deal(address(weth), address(this), inputAmount);

        etherFiL2ModeSyncPool.mockSetAmountOut(minOutputAmount);

        weth.approve(address(swapAdapter), inputAmount);
        uint256 outputAmount = swapAdapter.swapExactInput(weth, inputAmount, minOutputAmount, swapContext);

        assertEq(outputAmount, minOutputAmount);
        assertEq(weEth.balanceOf(address(this)), minOutputAmount);
    }
}
