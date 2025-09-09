import {
    loadFixture,
    time,
} from "@nomicfoundation/hardhat-toolbox-viem/network-helpers";
import { expect } from "chai";
import { deployPoolsAndRouterMockEnvironment } from "./fixtures";
import { signClosePositionRequest } from "./utils/SigningUtils";
import { takeBalanceSnapshot } from "./utils/StateUtils";
import { getERC20ApproveFunctionCallData, getRouterSwapFunctionCallData, getExactOutSwapperFunctionCallData, getSwapExactlyOutFunctionCallData } from "./utils/SwapUtils";
import { ClosePositionRequest, FunctionCallData, PayoutType } from "./utils/PerpStructUtils";
import { zeroAddress } from "viem";

describe("ExactOutSwapper", function () {
    describe("Exact Out Swaps", function () {
        it("Direct Exact Out Swap", async function () {
            const { user1, exactOutSwapper, weth, usdc, mockSwap, mockSwapRouter, publicClient, totalAmountIn, initialUSDCPrice } = await loadFixture(deployPoolsAndRouterMockEnvironment);

            const expectedAmountOut = 1000n * (10n ** 6n); // 1000 USDC
            const expectedAmountIn = expectedAmountOut * initialUSDCPrice / 10_000n * (10n ** 18n) / (10n ** 6n); // 0.4 ETH

            const wethBalancesBefore = await takeBalanceSnapshot(publicClient, weth.address, user1.account.address, exactOutSwapper.address, mockSwap.address);
            const usdcBalancesBefore = await takeBalanceSnapshot(publicClient, usdc.address, user1.account.address, exactOutSwapper.address, mockSwap.address);

            // Approve WasabiRouter for WETH transfer
            await weth.write.approve([exactOutSwapper.address, totalAmountIn], { account: user1.account });

            // Encode swaps
            const swapCalldata = getRouterSwapFunctionCallData(mockSwapRouter.address, weth.address, usdc.address, totalAmountIn, exactOutSwapper.address);
            const reverseSwapCalldata = getRouterSwapFunctionCallData(mockSwapRouter.address, usdc.address, weth.address, 0n, exactOutSwapper.address);

            await exactOutSwapper.write.swapExactOut(
                [weth.address, usdc.address, expectedAmountOut, totalAmountIn, swapCalldata, reverseSwapCalldata],
                { account: user1.account }
            );

            const wethBalancesAfter = await takeBalanceSnapshot(publicClient, weth.address, user1.account.address, exactOutSwapper.address, mockSwap.address, mockSwapRouter.address);
            const usdcBalancesAfter = await takeBalanceSnapshot(publicClient, usdc.address, user1.account.address, exactOutSwapper.address, mockSwap.address, mockSwapRouter.address);

            // WETH balance checks
            expect(wethBalancesBefore.get(user1.account.address) - wethBalancesAfter.get(user1.account.address))
                .to.be.lt(totalAmountIn, "User should have spent less than amountInMax WETH from their account");
            expect(wethBalancesBefore.get(user1.account.address) - wethBalancesAfter.get(user1.account.address))
                .to.equal(expectedAmountIn, "User should have spent expected WETH from their account");
            expect(wethBalancesAfter.get(exactOutSwapper.address) - wethBalancesBefore.get(exactOutSwapper.address))
                .to.equal(0n, "WasabiRouter should not have gained any WETH");
            expect(wethBalancesAfter.get(mockSwap.address) - wethBalancesBefore.get(mockSwap.address))
                .to.equal(expectedAmountIn, "MockSwap should have received expected WETH");
            expect(wethBalancesAfter.get(mockSwapRouter.address))
                .to.equal(0n, "MockSwapRouter should not have gained any WETH");

            // USDC balance checks
            expect(usdcBalancesAfter.get(user1.account.address) - usdcBalancesBefore.get(user1.account.address))
                .to.equal(expectedAmountOut, "User should have received expected USDC to their account");
            expect(usdcBalancesAfter.get(exactOutSwapper.address) - usdcBalancesBefore.get(exactOutSwapper.address)).
                to.equal(0n, "WasabiRouter should not have gained any USDC");
            expect(usdcBalancesBefore.get(mockSwap.address) - usdcBalancesAfter.get(mockSwap.address))
                .to.equal(expectedAmountOut, "MockSwap should have sent expected USDC to the user");
            expect(usdcBalancesAfter.get(mockSwapRouter.address))
                .to.equal(0n, "MockSwapRouter should not have gained any USDC");
        });

        it("Close Short Position With Exact Out Swap", async function () {
            const { user1, owner, orderSigner, wasabiShortPool, exactOutSwapper, weth, uPPG, mockSwap, mockSwapRouter, initialPPGPrice, sendDefaultShortOpenPositionRequest, computeShortMaxInterest } = await loadFixture(deployPoolsAndRouterMockEnvironment);

            const {position} = await sendDefaultShortOpenPositionRequest();

            await time.increase(86400n * 3n); // 1 day later

            // Price decrease by 20%
            const newPPGPrice = initialPPGPrice * 8n / 10n;
            await mockSwap.write.setPrice([uPPG.address, weth.address, newPPGPrice], { account: owner.account });

            // Prepare to Close Position
            const maxInterest = await computeShortMaxInterest(position);
            const expectedAmountOut = maxInterest + position.principal;
            const expectedAmountIn = expectedAmountOut * newPPGPrice / 10_000n;
            
            // Encode swaps
            const swapCalldata = getRouterSwapFunctionCallData(mockSwapRouter.address, weth.address, uPPG.address, position.collateralAmount, exactOutSwapper.address);
            const reverseSwapCalldata = getRouterSwapFunctionCallData(mockSwapRouter.address, uPPG.address, weth.address, 0n, exactOutSwapper.address);

            // Encode ClosePositionRequest
            const functionCallDataList: FunctionCallData[] = [
                getERC20ApproveFunctionCallData(weth.address, exactOutSwapper.address, position.collateralAmount),
                getExactOutSwapperFunctionCallData(exactOutSwapper.address, weth.address, uPPG.address, expectedAmountOut, position.collateralAmount, swapCalldata, reverseSwapCalldata),
            ];
            const request: ClosePositionRequest = {
                expiration: (BigInt(await time.latest()) + 300n),
                amount: 0n,
                interest: maxInterest,
                position,
                functionCallDataList,
                referrer: zeroAddress
            };
            const signature = await signClosePositionRequest(orderSigner, "WasabiShortPool", wasabiShortPool.address, request)

            // Close position
            const hash = await wasabiShortPool.write.closePosition([PayoutType.WRAPPED, request, signature], { account: user1.account });

            // PositionClosed event checks
            const closeEvents = await wasabiShortPool.getEvents.PositionClosed();
            expect(closeEvents).to.have.lengthOf(1);
            const closeEvent = closeEvents[0];

            expect(closeEvent.args.principalRepaid).to.equal(position.principal, "Principal should be repaid");
            expect(closeEvent.args.interestPaid).to.equal(maxInterest, "Interest should be paid");
            expect(closeEvent.args.payout).to.be.gt(0n, "Payout should be greater than 0");

            // Swap event checks
            const swapEvents = await mockSwap.getEvents.Swap();
            expect(swapEvents).to.have.lengthOf(2);
            const swapEvent = swapEvents[0];
            const reverseSwapEvent = swapEvents[1];

            expect(swapEvent.args.currencyIn).to.equal(reverseSwapEvent.args.currencyOut, "Swap currencies should be reversed");
            expect(swapEvent.args.currencyOut).to.equal(reverseSwapEvent.args.currencyIn, "Swap currencies should be reversed");
            expect(swapEvent.args.amountIn).to.equal(position.collateralAmount, "Swap amount in should be equal to collateral amount");
            expect(swapEvent.args.amountIn! - reverseSwapEvent.args.amountOut!).to.be.approximately(expectedAmountIn, 1n, "Swap amount in should be equal to expected amount in, +/- 1 wei");
            expect(swapEvent.args.amountOut! - reverseSwapEvent.args.amountIn!).to.equal(expectedAmountOut, "Swap amount out should be equal to expected amount out");
        });
    });

    describe("Validations", function () {
        it("NotWhitelistedSwapRouter", async function () {
            const { user1, exactOutSwapper, wasabiRouter, weth, usdc, mockSwapRouter, totalAmountIn } = await loadFixture(deployPoolsAndRouterMockEnvironment);

            const expectedAmountOut = 1000n * (10n ** 6n); // 1000 USDC

            // Encode swaps
            let swapCalldata = getRouterSwapFunctionCallData(wasabiRouter.address, weth.address, usdc.address, totalAmountIn, exactOutSwapper.address);
            let reverseSwapCalldata = getRouterSwapFunctionCallData(wasabiRouter.address, usdc.address, weth.address, 0n, exactOutSwapper.address);

            await expect(exactOutSwapper.write.swapExactOut(
                [weth.address, usdc.address, expectedAmountOut, totalAmountIn, swapCalldata, reverseSwapCalldata],
                { account: user1.account }
            )).to.be.rejectedWith("NotWhitelistedSwapRouter");

            swapCalldata = getRouterSwapFunctionCallData(mockSwapRouter.address, weth.address, usdc.address, totalAmountIn, exactOutSwapper.address);

            await expect(exactOutSwapper.write.swapExactOut(
                [weth.address, usdc.address, expectedAmountOut, totalAmountIn, swapCalldata, reverseSwapCalldata],
                { account: user1.account }
            )).to.be.rejectedWith("NotWhitelistedSwapRouter");
        });

        it("NotWhitelistedFunctionSelector", async function () {
            const { user1, exactOutSwapper, weth, usdc, mockSwapRouter, initialUSDCPrice, totalAmountIn } = await loadFixture(deployPoolsAndRouterMockEnvironment);

            const expectedAmountOut = 1000n * (10n ** 6n); // 1000 USDC

            // Approve WasabiRouter for WETH transfer
            await weth.write.approve([exactOutSwapper.address, totalAmountIn], { account: user1.account });

            // Encode swaps
            let swapCalldata = getSwapExactlyOutFunctionCallData(mockSwapRouter.address, weth.address, usdc.address, totalAmountIn, expectedAmountOut);
            let reverseSwapCalldata = getSwapExactlyOutFunctionCallData(mockSwapRouter.address, usdc.address, weth.address, 0n, expectedAmountOut);

            await expect(exactOutSwapper.write.swapExactOut(
                [weth.address, usdc.address, expectedAmountOut, totalAmountIn, swapCalldata, reverseSwapCalldata],
                { account: user1.account }
            )).to.be.rejectedWith("NotWhitelistedFunctionSelector");

            swapCalldata = getRouterSwapFunctionCallData(mockSwapRouter.address, weth.address, usdc.address, totalAmountIn, exactOutSwapper.address);

            await expect(exactOutSwapper.write.swapExactOut(
                [weth.address, usdc.address, expectedAmountOut, totalAmountIn, swapCalldata, reverseSwapCalldata],
                { account: user1.account }
            )).to.be.rejectedWith("NotWhitelistedFunctionSelector");
        });

        it("InsufficientAmountOutReceived", async function () {
            const { user1, exactOutSwapper, weth, usdc, mockSwapRouter, initialUSDCPrice } = await loadFixture(deployPoolsAndRouterMockEnvironment);

            const expectedAmountOut = 1000n * (10n ** 6n); // 1000 USDC
            const expectedAmountIn = expectedAmountOut * initialUSDCPrice / 10_000n * (10n ** 18n) / (10n ** 6n); // 0.4 ETH
            const amountInMax = expectedAmountIn - 10n;

            // Approve WasabiRouter for WETH transfer
            await weth.write.approve([exactOutSwapper.address, amountInMax], { account: user1.account });

            // Encode swaps
            const swapCalldata = getRouterSwapFunctionCallData(mockSwapRouter.address, weth.address, usdc.address, amountInMax, exactOutSwapper.address);
            const reverseSwapCalldata = getRouterSwapFunctionCallData(mockSwapRouter.address, usdc.address, weth.address, 0n, exactOutSwapper.address);

            await expect(exactOutSwapper.write.swapExactOut(
                [weth.address, usdc.address, expectedAmountOut, amountInMax, swapCalldata, reverseSwapCalldata],
                { account: user1.account }
            )).to.be.rejectedWith("InsufficientAmountOutReceived");
        });
    });
});