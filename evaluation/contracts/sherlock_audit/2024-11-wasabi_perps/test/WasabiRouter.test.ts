import {
    loadFixture,
    time,
} from "@nomicfoundation/hardhat-toolbox-viem/network-helpers";
import { expect } from "chai";
import { getAddress, parseEther, zeroAddress } from "viem";
import { deployPoolsAndRouterMockEnvironment } from "./fixtures";
import { signAddCollateralRequest, signOpenPositionRequest } from "./utils/SigningUtils";
import { getBalance, takeBalanceSnapshot } from "./utils/StateUtils";
import { AddCollateralRequest, FunctionCallData, OpenPositionRequest } from "./utils/PerpStructUtils";
import { getApproveAndSwapFunctionCallData } from "./utils/SwapUtils";
import { int } from "hardhat/internal/core/params/argumentTypes";

describe("WasabiRouter", function () {
    describe("Deployment", function () {
        it("Should set the right owner", async function () {
            const { wasabiRouter, manager } = await loadFixture(deployPoolsAndRouterMockEnvironment);
            expect(await wasabiRouter.read.owner()).to.equal(getAddress(manager.address));
        });

        it("Should set the right pool addresses", async function () {
            const { wasabiRouter, wasabiLongPool, wasabiShortPool } = await loadFixture(deployPoolsAndRouterMockEnvironment);
            expect(await wasabiRouter.read.longPool()).to.equal(getAddress(wasabiLongPool.address));
            expect(await wasabiRouter.read.shortPool()).to.equal(getAddress(wasabiShortPool.address));
        });

        it("Should set the right EIP712 domain", async function () {
            const { wasabiRouter } = await loadFixture(deployPoolsAndRouterMockEnvironment);
            const [, name, version, , verifyingContract] = await wasabiRouter.read.eip712Domain();
            expect(name).to.equal("WasabiRouter");
            expect(version).to.equal("1");
            expect(getAddress(verifyingContract)).to.equal(getAddress(wasabiRouter.address));
        });
    });

    describe("Open Position w/ Vault Deposits", function () {
        it("Long Position", async function () {
            const { sendRouterLongOpenPositionRequest, user1, orderExecutor, wethVault, wethAddress, uPPG, wasabiLongPool, publicClient, executionFee, totalAmountIn } = await loadFixture(deployPoolsAndRouterMockEnvironment);

            // Deposit into WETH Vault
            await wethVault.write.depositEth(
                [user1.account.address], 
                { value: parseEther("50"), account: user1.account }
            );

            const wethBalancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wethVault.address, orderExecutor.account.address);
            const poolPPGBalanceBefore = await getBalance(publicClient, uPPG.address, wasabiLongPool.address);
            const userBalanceBefore = await publicClient.getBalance({ address: user1.account.address });
            const orderExecutorBalanceBefore = await publicClient.getBalance({ address: orderExecutor.account.address });
            const userVaultSharesBefore = await wethVault.read.balanceOf([user1.account.address]);
            const expectedSharesSpent = await wethVault.read.convertToShares([totalAmountIn + executionFee]);

            const {position, gasUsed} = await sendRouterLongOpenPositionRequest();

            const wethBalancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wethVault.address, orderExecutor.account.address);
            const poolPPGBalanceAfter = await getBalance(publicClient, uPPG.address, wasabiLongPool.address);
            const userBalanceAfter = await publicClient.getBalance({ address: user1.account.address });
            const orderExecutorBalanceAfter = await publicClient.getBalance({ address: orderExecutor.account.address });
            const userVaultSharesAfter = await wethVault.read.balanceOf([user1.account.address]);
            
            expect(orderExecutorBalanceAfter).to.equal(orderExecutorBalanceBefore - gasUsed, "Order signer should have spent gas");
            expect(userBalanceAfter).to.equal(userBalanceBefore, "User should not have spent gas");
            expect(wethBalancesAfter.get(user1.account.address)).to.equal(wethBalancesBefore.get(user1.account.address), "User should not have spent WETH from their account");
            expect(wethBalancesAfter.get(wethVault.address)).to.equal(
                wethBalancesBefore.get(wethVault.address) - position.principal - position.downPayment - position.feesToBePaid - executionFee, 
                "Principal, down payment and fees should have been transferred from WETH vault"
            );
            expect(poolPPGBalanceAfter).to.equal(poolPPGBalanceBefore + position.collateralAmount, "Pool should have received uPPG collateral");
            expect(userVaultSharesAfter).to.equal(userVaultSharesBefore - expectedSharesSpent, "User's vault shares should have been burned");
            expect(wethBalancesAfter.get(orderExecutor.account.address)).to.equal(wethBalancesBefore.get(orderExecutor.account.address) + executionFee, "Fee receiver should have received execution fee");
        })

        it("Short Position", async function () {
            const { sendRouterShortOpenPositionRequest, user1, orderExecutor, wethVault, wethAddress, wasabiShortPool, publicClient, executionFee, totalAmountIn } = await loadFixture(deployPoolsAndRouterMockEnvironment);

            // Deposit into WETH Vault
            await wethVault.write.depositEth(
                [user1.account.address], 
                { value: parseEther("50"), account: user1.account }
            );

            const wethBalancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, wethVault.address, orderExecutor.account.address);
            const userBalanceBefore = await publicClient.getBalance({ address: user1.account.address });
            const orderExecutorBalanceBefore = await publicClient.getBalance({ address: orderExecutor.account.address });
            const userVaultSharesBefore = await wethVault.read.balanceOf([user1.account.address]);
            const expectedSharesSpent = await wethVault.read.convertToShares([totalAmountIn + executionFee]);

            const {position, gasUsed} = await sendRouterShortOpenPositionRequest();

            const wethBalancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, wethVault.address, orderExecutor.account.address);
            const userBalanceAfter = await publicClient.getBalance({ address: user1.account.address });
            const orderExecutorBalanceAfter = await publicClient.getBalance({ address: orderExecutor.account.address });
            const userVaultSharesAfter = await wethVault.read.balanceOf([user1.account.address]);
            
            expect(orderExecutorBalanceAfter).to.equal(orderExecutorBalanceBefore - gasUsed, "Order signer should have spent gas");
            expect(userBalanceAfter).to.equal(userBalanceBefore, "User should not have spent gas");
            expect(wethBalancesAfter.get(user1.account.address)).to.equal(wethBalancesBefore.get(user1.account.address), "User should not have spent WETH from their account");
            expect(wethBalancesAfter.get(wethVault.address)).to.equal(wethBalancesBefore.get(wethVault.address) - totalAmountIn - executionFee, "WETH down payment + fees should have been transferred from WETH vault");
            expect(wethBalancesAfter.get(wasabiShortPool.address)).to.equal(wethBalancesBefore.get(wasabiShortPool.address) + position.collateralAmount, "WETH collateral should have been transferred to short pool");
            expect(userVaultSharesAfter).to.equal(userVaultSharesBefore - expectedSharesSpent, "User's vault shares should have been burned");
            expect(wethBalancesAfter.get(orderExecutor.account.address)).to.equal(wethBalancesBefore.get(orderExecutor.account.address) + executionFee, "Fee receiver should have received execution fee");
        });
    });

    describe("Edit Position w/ Vault Deposits", function () {
        it("Long Position - Increase Size", async function () {
            const { sendRouterLongOpenPositionRequest, user1, wasabiRouter, wethVault, wethAddress, wasabiLongPool, mockSwap, uPPG, publicClient, executionFee, totalAmountIn, longTotalSize, initialPPGPrice, priceDenominator, orderSigner, orderExecutor } = await loadFixture(deployPoolsAndRouterMockEnvironment);

            // Deposit into WETH Vault
            await wethVault.write.depositEth(
                [user1.account.address], 
                { value: parseEther("50"), account: user1.account }
            );

            // Open position
            const {position} = await sendRouterLongOpenPositionRequest();
            
            await time.increase(86400n); // 1 day later

            // Edit position
            const functionCallDataList: FunctionCallData[] =
                getApproveAndSwapFunctionCallData(mockSwap.address, wethAddress, uPPG.address, longTotalSize);
            const openPositionRequest: OpenPositionRequest = {
                id: position.id,
                currency: position.currency,
                targetCurrency: position.collateralCurrency,
                downPayment: position.downPayment,
                principal: position.principal,
                minTargetAmount: longTotalSize * initialPPGPrice / priceDenominator,
                expiration: BigInt(await time.latest()) + 86400n,
                fee: position.feesToBePaid,
                functionCallDataList,
                existingPosition: position,
                referrer: zeroAddress
            };
            const traderRequest = { ...openPositionRequest, functionCallDataList: [], interestToPay: 0n };
            const signature = await signOpenPositionRequest(orderSigner, "WasabiLongPool", wasabiLongPool.address, openPositionRequest);
            const traderSignature = await signOpenPositionRequest(user1, "WasabiRouter", wasabiRouter.address, traderRequest);

            const wethBalancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wethVault.address, orderExecutor.account.address);
            const poolPPGBalanceBefore = await getBalance(publicClient, uPPG.address, wasabiLongPool.address);
            const userBalanceBefore = await publicClient.getBalance({ address: user1.account.address });
            const orderExecutorBalanceBefore = await publicClient.getBalance({ address: orderExecutor.account.address });
            const userVaultSharesBefore = await wethVault.read.balanceOf([user1.account.address]);
            const expectedSharesSpent = await wethVault.read.convertToShares([totalAmountIn + executionFee]);
            const totalAssetValueBefore = await wethVault.read.totalAssetValue();

            const hash = await wasabiRouter.write.openPosition(
                [wasabiLongPool.address, openPositionRequest, signature, traderSignature, executionFee], 
                { account: orderExecutor.account }
            );
            const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);

            const wethBalancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wethVault.address, orderExecutor.account.address);
            const poolPPGBalanceAfter = await getBalance(publicClient, uPPG.address, wasabiLongPool.address);
            const userBalanceAfter = await publicClient.getBalance({ address: user1.account.address });
            const orderExecutorBalanceAfter = await publicClient.getBalance({ address: orderExecutor.account.address });
            const userVaultSharesAfter = await wethVault.read.balanceOf([user1.account.address]);
            const totalAssetValueAfter = await wethVault.read.totalAssetValue();

            // Checks
            const events = await wasabiLongPool.getEvents.PositionIncreased();
            expect(events).to.have.lengthOf(1);
            const eventData = events[0].args;

            expect(eventData.id).to.equal(position.id);
            expect(eventData.downPaymentAdded).to.equal(totalAmountIn - eventData.feesAdded!);
            expect(eventData.principalAdded).to.equal(openPositionRequest.principal);
            expect(eventData.collateralAdded! + position.collateralAmount).to.equal(await uPPG.read.balanceOf([wasabiLongPool.address]));
            expect(eventData.collateralAdded).to.greaterThanOrEqual(openPositionRequest.minTargetAmount);

            expect(orderExecutorBalanceAfter).to.equal(orderExecutorBalanceBefore - gasUsed, "Order signer should have spent gas");
            expect(userBalanceAfter).to.equal(userBalanceBefore, "User should not have spent gas");
            expect(wethBalancesAfter.get(user1.account.address)).to.equal(wethBalancesBefore.get(user1.account.address), "User should not have spent WETH from their account");
            expect(wethBalancesAfter.get(wethVault.address)).to.equal(
                wethBalancesBefore.get(wethVault.address) - position.principal - position.downPayment - position.feesToBePaid - executionFee, 
                "Principal, down payment and fees should have been transferred from WETH vault"
            );
            expect(totalAssetValueAfter).to.equal(
                totalAssetValueBefore - position.downPayment - position.feesToBePaid - executionFee, 
                "Total asset value should reflect WETH withdrawn"
            );
            expect(poolPPGBalanceAfter).to.equal(poolPPGBalanceBefore + position.collateralAmount, "Pool should have received uPPG collateral");
            expect(userVaultSharesAfter).to.equal(userVaultSharesBefore - expectedSharesSpent, "User's vault shares should have been burned");
            expect(wethBalancesAfter.get(orderExecutor.account.address)).to.equal(wethBalancesBefore.get(orderExecutor.account.address) + executionFee, "Fee receiver should have received execution fee");
        });

        it("Long Position - Add Collateral", async function () {
            const { sendRouterLongOpenPositionRequest, computeLongMaxInterest, user1, wasabiRouter, wethVault, wethAddress, wasabiLongPool, uPPG, publicClient, totalAmountIn, orderSigner } = await loadFixture(deployPoolsAndRouterMockEnvironment);

            // Deposit into WETH Vault
            await wethVault.write.depositEth(
                [user1.account.address], 
                { value: parseEther("50"), account: user1.account }
            );

            // Open position
            const {position} = await sendRouterLongOpenPositionRequest();
            
            await time.increase(86400n); // 1 day later

            // Edit position
            const interest = await computeLongMaxInterest(position);
            const request: AddCollateralRequest = {
                amount: totalAmountIn,
                interest,
                expiration: BigInt(await time.latest()) + 86400n,
                position
            }
            const signature = await signAddCollateralRequest(orderSigner, "WasabiLongPool", wasabiLongPool.address, request);

            const wethBalancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wethVault.address);
            const poolPPGBalanceBefore = await getBalance(publicClient, uPPG.address, wasabiLongPool.address);
            const userVaultSharesBefore = await wethVault.read.balanceOf([user1.account.address]);
            const expectedSharesSpent = await wethVault.read.convertToShares([totalAmountIn]);
            const totalAssetValueBefore = await wethVault.read.totalAssetValue();

            const hash = await wasabiRouter.write.addCollateral(
                [wasabiLongPool.address, request, signature], 
                { account: user1.account }
            );

            const wethBalancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wethVault.address);
            const poolPPGBalanceAfter = await getBalance(publicClient, uPPG.address, wasabiLongPool.address);
            const userVaultSharesAfter = await wethVault.read.balanceOf([user1.account.address]);
            const totalAssetValueAfter = await wethVault.read.totalAssetValue();

            // Checks
            const events = await wasabiLongPool.getEvents.CollateralAdded();
            expect(events).to.have.lengthOf(1);
            const eventData = events[0].args;

            expect(eventData.id).to.equal(position.id);
            expect(eventData.downPaymentAdded).to.equal(request.amount - interest);
            expect(eventData.principalReduced).to.equal(request.amount - interest);
            expect(eventData.collateralAdded).to.equal(0);

            expect(wethBalancesAfter.get(user1.account.address)).to.equal(wethBalancesBefore.get(user1.account.address), "User should not have spent WETH from their account");
            expect(wethBalancesAfter.get(wethVault.address)).to.equal(
                wethBalancesBefore.get(wethVault.address), 
                "Amount transferred from WETH vault should be repaid as principal and interest"
            );
            expect(totalAssetValueAfter).to.equal(totalAssetValueBefore - totalAmountIn + interest, "Total asset value should reflect WETH withdrawn and interest paid");
            expect(poolPPGBalanceAfter).to.equal(poolPPGBalanceBefore + eventData.collateralAdded!, "Pool should have received uPPG collateral");
            expect(userVaultSharesAfter).to.equal(userVaultSharesBefore - expectedSharesSpent, "User's vault shares should have been burned");
        });

        it("Short Position - Increase Size", async function () {
            const { sendRouterShortOpenPositionRequest, user1, orderSigner, orderExecutor, wethVault, weth, uPPG, wasabiShortPool, wasabiRouter, mockSwap, publicClient, executionFee, totalAmountIn, shortPrincipal, initialPPGPrice, priceDenominator } = await loadFixture(deployPoolsAndRouterMockEnvironment);

            // Deposit into WETH Vault
            await wethVault.write.depositEth(
                [user1.account.address], 
                { value: parseEther("50"), account: user1.account }
            );

            // Open position
            const {position} = await sendRouterShortOpenPositionRequest();
            
            await time.increase(86400n); // 1 day later

            // Edit position
            const functionCallDataList: FunctionCallData[] =
            getApproveAndSwapFunctionCallData(mockSwap.address, uPPG.address, weth.address, shortPrincipal);
            const openPositionRequest: OpenPositionRequest = {
                id: position.id,
                currency: position.currency,
                targetCurrency: position.collateralCurrency,
                downPayment: position.downPayment,
                principal: position.principal,
                minTargetAmount: shortPrincipal * initialPPGPrice / priceDenominator,
                expiration: BigInt(await time.latest()) + 86400n,
                fee: position.feesToBePaid,
                functionCallDataList,
                existingPosition: position,
                referrer: zeroAddress
            };
            const traderRequest = { ...openPositionRequest, functionCallDataList: [], interestToPay: 0n };
            const signature = await signOpenPositionRequest(orderSigner, "WasabiShortPool", wasabiShortPool.address, openPositionRequest);
            const traderSignature = await signOpenPositionRequest(user1, "WasabiRouter", wasabiRouter.address, traderRequest);

            const wethBalancesBefore = await takeBalanceSnapshot(publicClient, weth.address, user1.account.address, wasabiShortPool.address, wethVault.address, orderExecutor.account.address);
            const userBalanceBefore = await publicClient.getBalance({ address: user1.account.address });
            const orderExecutorBalanceBefore = await publicClient.getBalance({ address: orderExecutor.account.address });
            const userVaultSharesBefore = await wethVault.read.balanceOf([user1.account.address]);
            const expectedSharesSpent = await wethVault.read.convertToShares([totalAmountIn + executionFee]);

            const hash = await wasabiRouter.write.openPosition(
                [wasabiShortPool.address, openPositionRequest, signature, traderSignature, executionFee], 
                { account: orderExecutor.account }
            );
            const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);

            const wethBalancesAfter = await takeBalanceSnapshot(publicClient, weth.address, user1.account.address, wasabiShortPool.address, wethVault.address, orderExecutor.account.address);
            const userBalanceAfter = await publicClient.getBalance({ address: user1.account.address });
            const orderExecutorBalanceAfter = await publicClient.getBalance({ address: orderExecutor.account.address });
            const userVaultSharesAfter = await wethVault.read.balanceOf([user1.account.address]);

            const events = await wasabiShortPool.getEvents.PositionIncreased();
            expect(events).to.have.lengthOf(1);
            const eventData = events[0].args;
            expect(eventData.id).to.equal(position.id);
            expect(eventData.downPaymentAdded).to.equal(totalAmountIn - eventData.feesAdded!);
            expect(eventData.principalAdded).to.equal(openPositionRequest.principal);
            expect(eventData.collateralAdded! + position.collateralAmount).to.equal(await weth.read.balanceOf([wasabiShortPool.address]));
            expect(eventData.collateralAdded).to.greaterThanOrEqual(openPositionRequest.minTargetAmount);

            expect(orderExecutorBalanceAfter).to.equal(orderExecutorBalanceBefore - gasUsed, "Order signer should have spent gas");
            expect(userBalanceAfter).to.equal(userBalanceBefore, "User should not have spent gas");
            expect(wethBalancesAfter.get(user1.account.address)).to.equal(wethBalancesBefore.get(user1.account.address), "User should not have spent WETH from their account");
            expect(wethBalancesAfter.get(wethVault.address)).to.equal(wethBalancesBefore.get(wethVault.address) - totalAmountIn - executionFee, "WETH down payment + fees should have been transferred from WETH vault");
            expect(wethBalancesAfter.get(wasabiShortPool.address)).to.equal(wethBalancesBefore.get(wasabiShortPool.address) + eventData.collateralAdded!, "WETH collateral should have been transferred to short pool");
            expect(userVaultSharesAfter).to.equal(userVaultSharesBefore - expectedSharesSpent, "User's vault shares should have been burned");
            expect(wethBalancesAfter.get(orderExecutor.account.address)).to.equal(wethBalancesBefore.get(orderExecutor.account.address) + executionFee, "Fee receiver should have received execution fee");
        });

        it("Short Position - Add Collateral", async function () {
            const { sendRouterShortOpenPositionRequest, user1, orderSigner, wethVault, weth, wasabiShortPool, wasabiRouter, publicClient, totalAmountIn } = await loadFixture(deployPoolsAndRouterMockEnvironment);

            // Deposit into WETH Vault
            await wethVault.write.depositEth(
                [user1.account.address], 
                { value: parseEther("50"), account: user1.account }
            );

            // Open position
            const {position} = await sendRouterShortOpenPositionRequest();
            
            await time.increase(86400n); // 1 day later

            const request: AddCollateralRequest = {
                amount: totalAmountIn,
                interest: 0n,
                expiration: BigInt(await time.latest()) + 86400n,
                position
            };
            const signature = await signAddCollateralRequest(orderSigner, "WasabiShortPool", wasabiShortPool.address, request);

            const wethBalancesBefore = await takeBalanceSnapshot(publicClient, weth.address, user1.account.address, wasabiShortPool.address, wethVault.address);
            const userVaultSharesBefore = await wethVault.read.balanceOf([user1.account.address]);
            const expectedSharesSpent = await wethVault.read.convertToShares([totalAmountIn]);
            const totalAssetValueBefore = await wethVault.read.totalAssetValue();

            const hash = await wasabiRouter.write.addCollateral(
                [wasabiShortPool.address, request, signature], 
                { account: user1.account }
            );

            const wethBalancesAfter = await takeBalanceSnapshot(publicClient, weth.address, user1.account.address, wasabiShortPool.address, wethVault.address);
            const userVaultSharesAfter = await wethVault.read.balanceOf([user1.account.address]);
            const totalAssetValueAfter = await wethVault.read.totalAssetValue();

            // Checks
            const events = await wasabiShortPool.getEvents.CollateralAdded();
            expect(events).to.have.lengthOf(1);
            const eventData = events[0].args;

            expect(eventData.id).to.equal(position.id);
            expect(eventData.downPaymentAdded).to.equal(request.amount);
            expect(eventData.principalReduced).to.equal(0);
            expect(eventData.collateralAdded).to.equal(totalAmountIn);

            expect(wethBalancesAfter.get(user1.account.address)).to.equal(wethBalancesBefore.get(user1.account.address), "User should not have spent WETH from their account");
            expect(wethBalancesAfter.get(wethVault.address)).to.equal(
                wethBalancesBefore.get(wethVault.address) - totalAmountIn, 
                "Down payment should have been transferred from WETH vault"
            );
            expect(wethBalancesAfter.get(wasabiShortPool.address)).to.equal(wethBalancesBefore.get(wasabiShortPool.address) + totalAmountIn, "WETH collateral should have been transferred to short pool");
            expect(totalAssetValueAfter).to.equal(totalAssetValueBefore - totalAmountIn, "Total asset value should reflect WETH withdrawn");
            expect(userVaultSharesAfter).to.equal(userVaultSharesBefore - expectedSharesSpent, "User's vault shares should have been burned");
        });
    });

    describe("Swaps w/ Vault Deposits", function () {
        describe("Vault -> Vault", function () {
            it("Exact In", async function () {
                const { createExactInRouterSwapData, user1, wasabiRouter, wethVault, ppgVault, wethAddress, uPPG, publicClient, swapFeeBips, feeReceiver, totalAmountIn, initialPPGPrice, priceDenominator } = await loadFixture(deployPoolsAndRouterMockEnvironment);

                // Deposit into WETH Vault
                await wethVault.write.depositEth(
                    [user1.account.address], 
                    { value: parseEther("50"), account: user1.account }
                );

                const wethBalancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wethVault.address);
                const ppgBalancesBefore = await takeBalanceSnapshot(publicClient, uPPG.address, user1.account.address, ppgVault.address, feeReceiver);
                const userWETHVaultSharesBefore = await wethVault.read.balanceOf([user1.account.address]);
                const userPPGVaultSharesBefore = await ppgVault.read.balanceOf([user1.account.address]);

                const swapCalldata = await createExactInRouterSwapData({amount: totalAmountIn, tokenIn: wethAddress, tokenOut: uPPG.address, swapRecipient: wasabiRouter.address, swapFee: swapFeeBips});
                await wasabiRouter.write.swapVaultToVault(
                    [totalAmountIn, wethAddress, uPPG.address, swapCalldata],
                    { account: user1.account }
                );

                const wethBalancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wethVault.address);
                const ppgBalancesAfter = await takeBalanceSnapshot(publicClient, uPPG.address, user1.account.address, ppgVault.address, feeReceiver);
                const userWETHVaultSharesAfter = await wethVault.read.balanceOf([user1.account.address]);
                const userPPGVaultSharesAfter = await ppgVault.read.balanceOf([user1.account.address]);

                expect(wethBalancesAfter.get(user1.account.address)).to.equal(wethBalancesBefore.get(user1.account.address), "User should not have spent WETH from their account");
                expect(wethBalancesAfter.get(wethVault.address)).to.equal(wethBalancesBefore.get(wethVault.address) - totalAmountIn, "User should have spent WETH from their vault deposits");
                expect(ppgBalancesAfter.get(user1.account.address)).to.equal(ppgBalancesBefore.get(user1.account.address), "User should not have received uPPG to their account");
                expect(ppgBalancesAfter.get(ppgVault.address)).to.equal(
                    ppgBalancesBefore.get(ppgVault.address) + (totalAmountIn * initialPPGPrice / priceDenominator * (10_000n - swapFeeBips) / 10_000n), 
                    "uPPG should have been deposited into the vault, minus the swap fee"
                );
                expect(ppgBalancesAfter.get(feeReceiver)).to.equal(totalAmountIn * initialPPGPrice / priceDenominator * swapFeeBips / 10_000n, "Fee receiver should have received fee in uPPG");
                expect(userWETHVaultSharesAfter).to.be.lt(userWETHVaultSharesBefore, "User should have fewer WETH Vault shares after the swap");
                expect(userPPGVaultSharesAfter).to.be.gt(userPPGVaultSharesBefore, "User should have more PPG Vault shares after the swap");
            });

            it("Exact Out", async function () {
                const { createExactOutRouterSwapData, user1, wasabiRouter, wethVault, ppgVault, wethAddress, uPPG, publicClient, swapFeeBips, feeReceiver, totalAmountIn, initialPPGPrice, priceDenominator } = await loadFixture(deployPoolsAndRouterMockEnvironment);

                // Deposit into WETH Vault
                await wethVault.write.depositEth(
                    [user1.account.address], 
                    { value: parseEther("50"), account: user1.account }
                );

                const wethBalancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wethVault.address);
                const ppgBalancesBefore = await takeBalanceSnapshot(publicClient, uPPG.address, user1.account.address, ppgVault.address, feeReceiver);
                const userWETHVaultSharesBefore = await wethVault.read.balanceOf([user1.account.address]);
                const userPPGVaultSharesBefore = await ppgVault.read.balanceOf([user1.account.address]);

                // Price is 1:1 uPPG:WETH, so amountOut == totalAmountIn
                // Pass in amountInMax > totalAmountIn to WasabiRouter call to make sure it returns remaining amount to vault
                const amountInMax = totalAmountIn + parseEther("0.1");
                const swapCalldata = await createExactOutRouterSwapData({amountInMax, amountOut: totalAmountIn, tokenIn: wethAddress, tokenOut: uPPG.address, swapRecipient: wasabiRouter.address, swapFee: swapFeeBips});
                await wasabiRouter.write.swapVaultToVault(
                    [amountInMax, wethAddress, uPPG.address, swapCalldata],
                    { account: user1.account }
                );

                const wethBalancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wethVault.address);
                const ppgBalancesAfter = await takeBalanceSnapshot(publicClient, uPPG.address, user1.account.address, ppgVault.address, feeReceiver);
                const userWETHVaultSharesAfter = await wethVault.read.balanceOf([user1.account.address]);
                const userPPGVaultSharesAfter = await ppgVault.read.balanceOf([user1.account.address]);

                expect(wethBalancesAfter.get(user1.account.address)).to.equal(wethBalancesBefore.get(user1.account.address), "User should not have spent WETH from their account");
                expect(wethBalancesAfter.get(wethVault.address)).to.equal(wethBalancesBefore.get(wethVault.address) - totalAmountIn, "User should have spent WETH from their vault deposits");
                expect(ppgBalancesAfter.get(user1.account.address)).to.equal(ppgBalancesBefore.get(user1.account.address), "User should not have received uPPG to their account");
                expect(ppgBalancesAfter.get(ppgVault.address)).to.equal(
                    ppgBalancesBefore.get(ppgVault.address) + (totalAmountIn * initialPPGPrice / priceDenominator * (10_000n - swapFeeBips) / 10_000n), 
                    "uPPG should have been deposited into the vault, minus the swap fee"
                );
                expect(ppgBalancesAfter.get(feeReceiver)).to.equal(totalAmountIn * initialPPGPrice / priceDenominator * swapFeeBips / 10_000n, "Fee receiver should have received fee in uPPG");
                expect(userWETHVaultSharesAfter).to.be.lt(userWETHVaultSharesBefore, "User should have fewer WETH Vault shares after the swap");
                expect(userPPGVaultSharesAfter).to.be.gt(userPPGVaultSharesBefore, "User should have more PPG Vault shares after the swap");
            });
        });
        
        describe("Vault -> Token", function () {
            it("Exact In (ERC20 out)", async function () {
                const { createExactInRouterSwapData, user1, wasabiRouter, wethVault, ppgVault, wethAddress, uPPG, publicClient, swapFeeBips, feeReceiver, totalAmountIn, initialPPGPrice, priceDenominator } = await loadFixture(deployPoolsAndRouterMockEnvironment);

                // Deposit into WETH Vault
                await wethVault.write.depositEth(
                    [user1.account.address], 
                    { value: parseEther("50"), account: user1.account }
                );

                const wethBalancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wethVault.address);
                const ppgBalancesBefore = await takeBalanceSnapshot(publicClient, uPPG.address, user1.account.address, ppgVault.address, feeReceiver);
                const userWETHVaultSharesBefore = await wethVault.read.balanceOf([user1.account.address]);
                const userPPGVaultSharesBefore = await ppgVault.read.balanceOf([user1.account.address]);

                const swapCalldata = await createExactInRouterSwapData({amount: totalAmountIn, tokenIn: wethAddress, tokenOut: uPPG.address, swapRecipient: user1.account.address, swapFee: swapFeeBips});
                await wasabiRouter.write.swapVaultToToken(
                    [totalAmountIn, wethAddress, uPPG.address, swapCalldata],
                    { account: user1.account }
                );

                const wethBalancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wethVault.address);
                const ppgBalancesAfter = await takeBalanceSnapshot(publicClient, uPPG.address, user1.account.address, ppgVault.address, feeReceiver);
                const userWETHVaultSharesAfter = await wethVault.read.balanceOf([user1.account.address]);
                const userPPGVaultSharesAfter = await ppgVault.read.balanceOf([user1.account.address]);

                expect(wethBalancesAfter.get(user1.account.address)).to.equal(wethBalancesBefore.get(user1.account.address), "User should not have spent WETH from their account");
                expect(wethBalancesAfter.get(wethVault.address)).to.equal(wethBalancesBefore.get(wethVault.address) - totalAmountIn, "User should have spent WETH from their vault deposits");
                expect(ppgBalancesAfter.get(user1.account.address)).to.equal(
                    ppgBalancesBefore.get(user1.account.address) + (totalAmountIn * initialPPGPrice / priceDenominator * (10_000n - swapFeeBips) / 10_000n),
                    "User should have received uPPG to their account, minus the swap fee"
                );
                expect(ppgBalancesAfter.get(ppgVault.address)).to.equal(ppgBalancesBefore.get(ppgVault.address), "uPPG should not have been deposited into the vault");
                expect(ppgBalancesAfter.get(feeReceiver)).to.equal(totalAmountIn * initialPPGPrice / priceDenominator * swapFeeBips / 10_000n, "Fee receiver should have received fee in uPPG");
                expect(userWETHVaultSharesAfter).to.be.lt(userWETHVaultSharesBefore, "User should have fewer WETH Vault shares after the swap");
                expect(userPPGVaultSharesAfter).to.equal(userPPGVaultSharesBefore, "User should not have more PPG Vault shares after the swap");
            });

            it("Exact In (ETH out)", async function () {
                const { createExactInRouterSwapData, user1, wasabiRouter, wethVault, ppgVault, wethAddress, uPPG, publicClient, swapFeeBips, feeReceiver, totalAmountIn, initialPPGPrice, priceDenominator } = await loadFixture(deployPoolsAndRouterMockEnvironment);

                // Approve and deposit into uPPG Vault
                await uPPG.write.approve(
                    [ppgVault.address, parseEther("10")],
                    { account: user1.account }
                );
                await ppgVault.write.deposit(
                    [parseEther("10"), user1.account.address],
                    { account: user1.account }
                );

                const wethBalancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wethVault.address);
                const ppgBalancesBefore = await takeBalanceSnapshot(publicClient, uPPG.address, user1.account.address, ppgVault.address);
                const userWETHVaultSharesBefore = await wethVault.read.balanceOf([user1.account.address]);
                const userPPGVaultSharesBefore = await ppgVault.read.balanceOf([user1.account.address]);
                const userETHBalanceBefore = await publicClient.getBalance({ address: user1.account.address });
                const feeReceiverBalanceBefore = await publicClient.getBalance({ address: feeReceiver });

                const swapCalldata = await createExactInRouterSwapData({amount: totalAmountIn, tokenIn: uPPG.address, tokenOut: wethAddress, swapRecipient: user1.account.address, swapFee: swapFeeBips, unwrapEth: true});
                const hash = await wasabiRouter.write.swapVaultToToken(
                    [totalAmountIn, uPPG.address, wethAddress, swapCalldata],
                    { account: user1.account }
                );
                const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);

                const wethBalancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wethVault.address);
                const ppgBalancesAfter = await takeBalanceSnapshot(publicClient, uPPG.address, user1.account.address, ppgVault.address);
                const userWETHVaultSharesAfter = await wethVault.read.balanceOf([user1.account.address]);
                const userPPGVaultSharesAfter = await ppgVault.read.balanceOf([user1.account.address]);
                const userETHBalanceAfter = await publicClient.getBalance({ address: user1.account.address });
                const feeReceiverBalanceAfter = await publicClient.getBalance({ address: feeReceiver });

                expect(ppgBalancesAfter.get(user1.account.address)).to.equal(ppgBalancesBefore.get(user1.account.address), "uPPG should not have been transferred from the user's account");
                expect(ppgBalancesAfter.get(ppgVault.address)).to.equal(ppgBalancesBefore.get(ppgVault.address) - totalAmountIn, "uPPG should have been withdrawn from the vault");
                expect(wethBalancesBefore.get(user1.account.address)).to.equal(wethBalancesAfter.get(user1.account.address), "User should not have received WETH out");
                expect(userETHBalanceAfter).to.equal(
                    userETHBalanceBefore + (totalAmountIn * priceDenominator / initialPPGPrice * (10_000n - swapFeeBips) / 10_000n) - gasUsed,
                    "User should have received ETH to their account, minus the swap fees"
                );
                expect(feeReceiverBalanceAfter).to.equal(feeReceiverBalanceBefore + totalAmountIn * priceDenominator / initialPPGPrice * swapFeeBips / 10_000n, "Fee receiver should have received fee in ETH");
                expect(userWETHVaultSharesAfter).to.equal(userWETHVaultSharesBefore, "User should not have more WETH Vault shares after the swap");
                expect(userPPGVaultSharesAfter).to.be.lt(userPPGVaultSharesBefore, "User should have fewer PPG Vault shares after the swap");
            });

            it("Exact Out (ERC20 out)", async function () {
                const { createExactOutRouterSwapData, user1, wasabiRouter, wethVault, ppgVault, wethAddress, uPPG, publicClient, swapFeeBips, feeReceiver, totalAmountIn, initialPPGPrice, priceDenominator } = await loadFixture(deployPoolsAndRouterMockEnvironment);

                // Deposit into WETH Vault
                await wethVault.write.depositEth(
                    [user1.account.address], 
                    { value: parseEther("50"), account: user1.account }
                );

                const wethBalancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wethVault.address);
                const ppgBalancesBefore = await takeBalanceSnapshot(publicClient, uPPG.address, user1.account.address, ppgVault.address, feeReceiver);
                const userWETHVaultSharesBefore = await wethVault.read.balanceOf([user1.account.address]);
                const userPPGVaultSharesBefore = await ppgVault.read.balanceOf([user1.account.address]);

                // Price is 1:1 uPPG:WETH, so amountOut == totalAmountIn
                // Pass in amountInMax > totalAmountIn to WasabiRouter call to make sure it returns remaining amount to vault
                const amountInMax = totalAmountIn + parseEther("0.1");
                const swapCalldata = await createExactOutRouterSwapData({amountInMax, amountOut: totalAmountIn, tokenIn: wethAddress, tokenOut: uPPG.address, swapRecipient: user1.account.address, swapFee: swapFeeBips});
                await wasabiRouter.write.swapVaultToToken(
                    [amountInMax, wethAddress, uPPG.address, swapCalldata],
                    { account: user1.account }
                );

                const wethBalancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wethVault.address);
                const ppgBalancesAfter = await takeBalanceSnapshot(publicClient, uPPG.address, user1.account.address, ppgVault.address, feeReceiver);
                const userWETHVaultSharesAfter = await wethVault.read.balanceOf([user1.account.address]);
                const userPPGVaultSharesAfter = await ppgVault.read.balanceOf([user1.account.address]);

                expect(wethBalancesAfter.get(user1.account.address)).to.equal(wethBalancesBefore.get(user1.account.address), "User should not have spent WETH from their account");
                expect(wethBalancesAfter.get(wethVault.address)).to.equal(wethBalancesBefore.get(wethVault.address) - totalAmountIn, "User should have spent WETH from their vault deposits");
                expect(ppgBalancesAfter.get(user1.account.address)).to.equal(
                    ppgBalancesBefore.get(user1.account.address) + (totalAmountIn * initialPPGPrice / priceDenominator * (10_000n - swapFeeBips) / 10_000n),
                    "User should have received uPPG to their account, minus the swap fee"
                );
                expect(ppgBalancesAfter.get(ppgVault.address)).to.equal(ppgBalancesBefore.get(ppgVault.address), "uPPG should not have been deposited into the vault");
                expect(ppgBalancesAfter.get(feeReceiver)).to.equal(totalAmountIn * initialPPGPrice / priceDenominator * swapFeeBips / 10_000n, "Fee receiver should have received fee in uPPG");
                expect(userWETHVaultSharesAfter).to.be.lt(userWETHVaultSharesBefore, "User should have fewer WETH Vault shares after the swap");
                expect(userPPGVaultSharesAfter).to.equal(userPPGVaultSharesBefore, "User should not have more PPG Vault shares after the swap");
            });

            it("Exact Out (ETH out)", async function () {
                const { createExactOutRouterSwapData, user1, wasabiRouter, wethVault, ppgVault, wethAddress, uPPG, publicClient, swapFeeBips, feeReceiver, totalAmountIn, initialPPGPrice, priceDenominator } = await loadFixture(deployPoolsAndRouterMockEnvironment);

                // Approve and deposit into uPPG Vault
                await uPPG.write.approve(
                    [ppgVault.address, parseEther("10")],
                    { account: user1.account }
                );
                await ppgVault.write.deposit(
                    [parseEther("10"), user1.account.address],
                    { account: user1.account }
                );

                const wethBalancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wethVault.address);
                const ppgBalancesBefore = await takeBalanceSnapshot(publicClient, uPPG.address, user1.account.address, ppgVault.address);
                const userWETHVaultSharesBefore = await wethVault.read.balanceOf([user1.account.address]);
                const userPPGVaultSharesBefore = await ppgVault.read.balanceOf([user1.account.address]);
                const userETHBalanceBefore = await publicClient.getBalance({ address: user1.account.address });
                const feeReceiverBalanceBefore = await publicClient.getBalance({ address: feeReceiver });

                // Price is 1:1 uPPG:WETH, so amountOut == totalAmountIn
                // Pass in amountInMax > totalAmountIn to WasabiRouter call to make sure it returns remaining amount to vault
                const amountInMax = totalAmountIn + parseEther("0.1");
                const swapCalldata = await createExactOutRouterSwapData({amountInMax, amountOut: totalAmountIn, tokenIn: uPPG.address, tokenOut: wethAddress, swapRecipient: user1.account.address, swapFee: swapFeeBips, unwrapEth: true});
                const hash = await wasabiRouter.write.swapVaultToToken(
                    [amountInMax, uPPG.address, wethAddress, swapCalldata],
                    { account: user1.account }
                );
                const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);

                const wethBalancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wethVault.address);
                const ppgBalancesAfter = await takeBalanceSnapshot(publicClient, uPPG.address, user1.account.address, ppgVault.address);
                const userWETHVaultSharesAfter = await wethVault.read.balanceOf([user1.account.address]);
                const userPPGVaultSharesAfter = await ppgVault.read.balanceOf([user1.account.address]);
                const userETHBalanceAfter = await publicClient.getBalance({ address: user1.account.address });
                const feeReceiverBalanceAfter = await publicClient.getBalance({ address: feeReceiver });

                expect(ppgBalancesAfter.get(user1.account.address)).to.equal(ppgBalancesBefore.get(user1.account.address), "uPPG should not have been transferred from the user's account");
                expect(ppgBalancesAfter.get(ppgVault.address)).to.equal(ppgBalancesBefore.get(ppgVault.address) - totalAmountIn, "uPPG should have been withdrawn from the vault");
                expect(wethBalancesBefore.get(user1.account.address)).to.equal(wethBalancesAfter.get(user1.account.address), "User should not have received WETH out");
                expect(userETHBalanceAfter).to.equal(
                    userETHBalanceBefore + (totalAmountIn * priceDenominator / initialPPGPrice * (10_000n - swapFeeBips) / 10_000n) - gasUsed,
                    "User should have received ETH to their account, minus the swap fees"
                );
                expect(feeReceiverBalanceAfter).to.equal(feeReceiverBalanceBefore + totalAmountIn * priceDenominator / initialPPGPrice * swapFeeBips / 10_000n, "Fee receiver should have received fee in ETH");
                expect(userWETHVaultSharesAfter).to.equal(userWETHVaultSharesBefore, "User should not have more WETH Vault shares after the swap");
                expect(userPPGVaultSharesAfter).to.be.lt(userPPGVaultSharesBefore, "User should have fewer PPG Vault shares after the swap");
            });

            it("Withdraw from Vault (ERC20)", async function () {
                const { user1, wasabiRouter, wethVault, wethAddress, publicClient, swapFeeBips, feeReceiver, totalAmountIn } = await loadFixture(deployPoolsAndRouterMockEnvironment);

                // Deposit into WETH Vault
                await wethVault.write.depositEth(
                    [user1.account.address], 
                    { value: parseEther("50"), account: user1.account }
                );

                const wethBalancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wethVault.address, feeReceiver);
                const userWETHVaultSharesBefore = await wethVault.read.balanceOf([user1.account.address]);

                // Call swapVaultToToken with tokenIn == tokenOut and empty calldata
                await wasabiRouter.write.swapVaultToToken(
                    [totalAmountIn, wethAddress, wethAddress, "0x"],
                    { account: user1.account }
                );

                const wethBalancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wethVault.address, feeReceiver);
                const userWETHVaultSharesAfter = await wethVault.read.balanceOf([user1.account.address]);

                expect(wethBalancesAfter.get(wethVault.address)).to.equal(wethBalancesBefore.get(wethVault.address) - totalAmountIn, "User should have withdrawn WETH from their vault deposits");
                expect(wethBalancesAfter.get(user1.account.address)).to.equal(
                    wethBalancesBefore.get(user1.account.address) + totalAmountIn * (10000n - swapFeeBips) / 10000n, 
                    "User should have received WETH to their account, minus the withdraw fee"
                );
                expect(wethBalancesAfter.get(feeReceiver)).to.equal(
                    wethBalancesBefore.get(feeReceiver) + totalAmountIn * swapFeeBips / 10000n, 
                    "Fee receiver should have received fee in WETH"
                );
                expect(userWETHVaultSharesAfter).to.be.lt(userWETHVaultSharesBefore, "User should have fewer WETH Vault shares after the swap");
            });

            it("Withdraw from Vault (ETH)", async function () {
                const { user1, wasabiRouter, wethVault, wethAddress, publicClient, swapFeeBips, feeReceiver, totalAmountIn } = await loadFixture(deployPoolsAndRouterMockEnvironment);

                // Deposit into WETH Vault
                await wethVault.write.depositEth(
                    [user1.account.address], 
                    { value: parseEther("50"), account: user1.account }
                );

                const wethBalancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wethVault.address, feeReceiver);
                const userWETHVaultSharesBefore = await wethVault.read.balanceOf([user1.account.address]);
                const userETHBalanceBefore = await publicClient.getBalance({ address: user1.account.address });
                const feeReceiverBalanceBefore = await publicClient.getBalance({ address: feeReceiver });

                // Call swapVaultToToken with tokenIn == WETH, tokenOut == address(0) and empty calldata
                const hash = await wasabiRouter.write.swapVaultToToken(
                    [totalAmountIn, wethAddress, zeroAddress, "0x"],
                    { account: user1.account }
                );
                const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);

                const wethBalancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wethVault.address, feeReceiver);
                const userWETHVaultSharesAfter = await wethVault.read.balanceOf([user1.account.address]);
                const userETHBalanceAfter = await publicClient.getBalance({ address: user1.account.address });
                const feeReceiverBalanceAfter = await publicClient.getBalance({ address: feeReceiver });

                expect(wethBalancesAfter.get(wethVault.address)).to.equal(
                    wethBalancesBefore.get(wethVault.address) - totalAmountIn, 
                    "User should have withdrawn WETH from their vault deposits"
                );
                expect(wethBalancesAfter.get(user1.account.address)).to.equal(
                    wethBalancesBefore.get(user1.account.address), 
                    "User should not have received WETH to their account"
                );
                expect(wethBalancesAfter.get(feeReceiver)).to.equal(wethBalancesBefore.get(feeReceiver), "Fee receiver should not have received fee in WETH");
                expect(userWETHVaultSharesAfter).to.be.lt(userWETHVaultSharesBefore, "User should have fewer WETH Vault shares after the swap");
                expect(userETHBalanceAfter).to.equal(
                    userETHBalanceBefore - gasUsed + totalAmountIn * (10000n - swapFeeBips) / 10000n, 
                    "User should have received ETH to their account, minus the withdraw fee"
                ); 
                expect(feeReceiverBalanceAfter).to.equal(
                    feeReceiverBalanceBefore + totalAmountIn * swapFeeBips / 10000n,
                    "Fee receiver should have received fee in ETH"
                );
            });
        });

        describe("Token -> Vault", function () {
            it("Exact In (ERC20)", async function () {
                const { createExactInRouterSwapData, user1, wasabiRouter, wethVault, ppgVault, weth, uPPG, publicClient, swapFeeBips, feeReceiver, totalAmountIn, initialPPGPrice, priceDenominator } = await loadFixture(deployPoolsAndRouterMockEnvironment);

                // Deposit into WETH Vault
                await wethVault.write.depositEth(
                    [user1.account.address], 
                    { value: parseEther("50"), account: user1.account }
                );

                const wethBalancesBefore = await takeBalanceSnapshot(publicClient, weth.address, user1.account.address, wethVault.address);
                const ppgBalancesBefore = await takeBalanceSnapshot(publicClient, uPPG.address, user1.account.address, ppgVault.address, feeReceiver);
                const userWETHVaultSharesBefore = await wethVault.read.balanceOf([user1.account.address]);
                const userPPGVaultSharesBefore = await ppgVault.read.balanceOf([user1.account.address]);

                // Approve WasabiRouter for WETH transfer
                await weth.write.approve(
                    [wasabiRouter.address, totalAmountIn],
                    { account: user1.account }
                );
                const swapCalldata = await createExactInRouterSwapData({amount: totalAmountIn, tokenIn: weth.address, tokenOut: uPPG.address, swapRecipient: wasabiRouter.address, swapFee: swapFeeBips});
                await wasabiRouter.write.swapTokenToVault(
                    [totalAmountIn, weth.address, uPPG.address, swapCalldata],
                    { account: user1.account }
                );

                const wethBalancesAfter = await takeBalanceSnapshot(publicClient, weth.address, user1.account.address, wethVault.address);
                const ppgBalancesAfter = await takeBalanceSnapshot(publicClient, uPPG.address, user1.account.address, ppgVault.address, feeReceiver);
                const userWETHVaultSharesAfter = await wethVault.read.balanceOf([user1.account.address]);
                const userPPGVaultSharesAfter = await ppgVault.read.balanceOf([user1.account.address]);

                expect(wethBalancesAfter.get(user1.account.address)).to.equal(wethBalancesBefore.get(user1.account.address) - totalAmountIn, "User should have spent WETH from their account");
                expect(wethBalancesAfter.get(wethVault.address)).to.equal(wethBalancesBefore.get(wethVault.address), "User should not have spent WETH from their vault deposits");
                expect(ppgBalancesAfter.get(user1.account.address)).to.equal(ppgBalancesBefore.get(user1.account.address), "User should not have received uPPG to their account");
                expect(ppgBalancesAfter.get(ppgVault.address)).to.equal(
                    ppgBalancesBefore.get(ppgVault.address) + (totalAmountIn * initialPPGPrice / priceDenominator * (10_000n - swapFeeBips) / 10_000n), 
                    "uPPG should have been deposited into the vault, minus the swap fee"
                );
                expect(ppgBalancesAfter.get(feeReceiver)).to.equal(totalAmountIn * initialPPGPrice / priceDenominator * swapFeeBips / 10_000n, "Fee receiver should have received fee in uPPG");
                expect(userWETHVaultSharesAfter).to.equal(userWETHVaultSharesBefore, "User should not have fewer WETH Vault shares after the swap");
                expect(userPPGVaultSharesAfter).to.be.gt(userPPGVaultSharesBefore, "User should have more PPG Vault shares after the swap");
            });

            it("Exact In (ETH)", async function () {
                const { createExactInRouterSwapData, user1, wasabiRouter, wethVault, ppgVault, weth, uPPG, publicClient, swapFeeBips, feeReceiver, totalAmountIn, initialPPGPrice, priceDenominator } = await loadFixture(deployPoolsAndRouterMockEnvironment);

                // Deposit into WETH Vault
                await wethVault.write.depositEth(
                    [user1.account.address], 
                    { value: parseEther("50"), account: user1.account }
                );

                const wethBalancesBefore = await takeBalanceSnapshot(publicClient, weth.address, user1.account.address, wethVault.address);
                const ppgBalancesBefore = await takeBalanceSnapshot(publicClient, uPPG.address, user1.account.address, ppgVault.address, feeReceiver);
                const userWETHVaultSharesBefore = await wethVault.read.balanceOf([user1.account.address]);
                const userPPGVaultSharesBefore = await ppgVault.read.balanceOf([user1.account.address]);
                const userETHBalanceBefore = await publicClient.getBalance({ address: user1.account.address });

                const swapCalldata = await createExactInRouterSwapData({amount: totalAmountIn, tokenIn: weth.address, tokenOut: uPPG.address, swapRecipient: wasabiRouter.address, swapFee: swapFeeBips});
                const hash = await wasabiRouter.write.swapTokenToVault(
                    [totalAmountIn, weth.address, uPPG.address, swapCalldata],
                    { account: user1.account, value: totalAmountIn }
                );
                const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);

                const wethBalancesAfter = await takeBalanceSnapshot(publicClient, weth.address, user1.account.address, wethVault.address);
                const ppgBalancesAfter = await takeBalanceSnapshot(publicClient, uPPG.address, user1.account.address, ppgVault.address, feeReceiver);
                const userWETHVaultSharesAfter = await wethVault.read.balanceOf([user1.account.address]);
                const userPPGVaultSharesAfter = await ppgVault.read.balanceOf([user1.account.address]);
                const userETHBalanceAfter = await publicClient.getBalance({ address: user1.account.address });

                expect(wethBalancesAfter.get(user1.account.address)).to.equal(wethBalancesBefore.get(user1.account.address), "User should not have spent WETH from their account");
                expect(wethBalancesAfter.get(wethVault.address)).to.equal(wethBalancesBefore.get(wethVault.address), "User should not have spent WETH from their vault deposits");
                expect(userETHBalanceAfter).to.equal(userETHBalanceBefore - totalAmountIn - gasUsed, "User should have spent ETH from their account")
                expect(ppgBalancesAfter.get(user1.account.address)).to.equal(ppgBalancesBefore.get(user1.account.address), "User should not have received uPPG to their account");
                expect(ppgBalancesAfter.get(ppgVault.address)).to.equal(
                    ppgBalancesBefore.get(ppgVault.address) + (totalAmountIn * initialPPGPrice / priceDenominator * (10_000n - swapFeeBips) / 10_000n), 
                    "uPPG should have been deposited into the vault, minus the swap fee"
                );
                expect(ppgBalancesAfter.get(feeReceiver)).to.equal(totalAmountIn * initialPPGPrice / priceDenominator * swapFeeBips / 10_000n, "Fee receiver should have received fee in uPPG");
                expect(userWETHVaultSharesAfter).to.equal(userWETHVaultSharesBefore, "User should not have fewer WETH Vault shares after the swap");
                expect(userPPGVaultSharesAfter).to.be.gt(userPPGVaultSharesBefore, "User should have more PPG Vault shares after the swap");
            });

            it("Exact Out (ERC20)", async function () {
                const { createExactOutRouterSwapData, user1, wasabiRouter, wethVault, ppgVault, weth, uPPG, publicClient, swapFeeBips, feeReceiver, totalAmountIn, initialPPGPrice, priceDenominator } = await loadFixture(deployPoolsAndRouterMockEnvironment);

                // Deposit into WETH Vault
                await wethVault.write.depositEth(
                    [user1.account.address], 
                    { value: parseEther("50"), account: user1.account }
                );

                const wethBalancesBefore = await takeBalanceSnapshot(publicClient, weth.address, user1.account.address, wethVault.address);
                const ppgBalancesBefore = await takeBalanceSnapshot(publicClient, uPPG.address, user1.account.address, ppgVault.address, feeReceiver);
                const userWETHVaultSharesBefore = await wethVault.read.balanceOf([user1.account.address]);
                const userPPGVaultSharesBefore = await ppgVault.read.balanceOf([user1.account.address]);

                // Price is 1:1 uPPG:WETH, so amountOut == totalAmountIn
                // Pass in amountInMax > totalAmountIn to WasabiRouter call to make sure it returns remaining amount to vault
                const amountInMax = totalAmountIn + parseEther("0.1");
                // Approve WasabiRouter for WETH transfer
                await weth.write.approve(
                    [wasabiRouter.address, amountInMax],
                    { account: user1.account }
                );
                const swapCalldata = await createExactOutRouterSwapData({amountInMax, amountOut: totalAmountIn, tokenIn: weth.address, tokenOut: uPPG.address, swapRecipient: wasabiRouter.address, swapFee: swapFeeBips});
                await wasabiRouter.write.swapTokenToVault(
                    [amountInMax, weth.address, uPPG.address, swapCalldata],
                    { account: user1.account }
                );

                const wethBalancesAfter = await takeBalanceSnapshot(publicClient, weth.address, user1.account.address, wethVault.address);
                const ppgBalancesAfter = await takeBalanceSnapshot(publicClient, uPPG.address, user1.account.address, ppgVault.address, feeReceiver);
                const userWETHVaultSharesAfter = await wethVault.read.balanceOf([user1.account.address]);
                const userPPGVaultSharesAfter = await ppgVault.read.balanceOf([user1.account.address]);

                expect(wethBalancesAfter.get(user1.account.address)).to.equal(wethBalancesBefore.get(user1.account.address) - totalAmountIn, "User should have spent WETH from their account");
                expect(wethBalancesAfter.get(wethVault.address)).to.equal(wethBalancesBefore.get(wethVault.address), "User should not have spent WETH from their vault deposits");
                expect(ppgBalancesAfter.get(user1.account.address)).to.equal(ppgBalancesBefore.get(user1.account.address), "User should not have received uPPG to their account");
                expect(ppgBalancesAfter.get(ppgVault.address)).to.equal(
                    ppgBalancesBefore.get(ppgVault.address) + (totalAmountIn * initialPPGPrice / priceDenominator * (10_000n - swapFeeBips) / 10_000n), 
                    "uPPG should have been deposited into the vault, minus the swap fee"
                );
                expect(ppgBalancesAfter.get(feeReceiver)).to.equal(totalAmountIn * initialPPGPrice / priceDenominator * swapFeeBips / 10_000n, "Fee receiver should have received fee in uPPG");
                expect(userWETHVaultSharesAfter).to.equal(userWETHVaultSharesBefore, "User should not have fewer WETH Vault shares after the swap");
                expect(userPPGVaultSharesAfter).to.be.gt(userPPGVaultSharesBefore, "User should have more PPG Vault shares after the swap");
            });

            it("Exact Out (ETH)", async function () {
                const { createExactOutRouterSwapData, user1, wasabiRouter, wethVault, ppgVault, weth, uPPG, publicClient, swapFeeBips, feeReceiver, totalAmountIn, initialPPGPrice, priceDenominator } = await loadFixture(deployPoolsAndRouterMockEnvironment);

                // Deposit into WETH Vault
                await wethVault.write.depositEth(
                    [user1.account.address], 
                    { value: parseEther("50"), account: user1.account }
                );

                const wethBalancesBefore = await takeBalanceSnapshot(publicClient, weth.address, user1.account.address, wethVault.address);
                const ppgBalancesBefore = await takeBalanceSnapshot(publicClient, uPPG.address, user1.account.address, ppgVault.address, feeReceiver);
                const userWETHVaultSharesBefore = await wethVault.read.balanceOf([user1.account.address]);
                const userPPGVaultSharesBefore = await ppgVault.read.balanceOf([user1.account.address]);
                const userETHBalanceBefore = await publicClient.getBalance({ address: user1.account.address });

                // Price is 1:1 uPPG:WETH, so amountOut == totalAmountIn
                // Pass in amountInMax > totalAmountIn to WasabiRouter call to make sure it returns remaining amount to vault
                const amountInMax = totalAmountIn + parseEther("0.1");
                const swapCalldata = await createExactOutRouterSwapData({amountInMax, amountOut: totalAmountIn, tokenIn: weth.address, tokenOut: uPPG.address, swapRecipient: wasabiRouter.address, swapFee: swapFeeBips});
                const hash = await wasabiRouter.write.swapTokenToVault(
                    [amountInMax, weth.address, uPPG.address, swapCalldata],
                    { account: user1.account, value: amountInMax }
                );
                const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);

                const wethBalancesAfter = await takeBalanceSnapshot(publicClient, weth.address, user1.account.address, wethVault.address);
                const ppgBalancesAfter = await takeBalanceSnapshot(publicClient, uPPG.address, user1.account.address, ppgVault.address, feeReceiver);
                const userWETHVaultSharesAfter = await wethVault.read.balanceOf([user1.account.address]);
                const userPPGVaultSharesAfter = await ppgVault.read.balanceOf([user1.account.address]);
                const userETHBalanceAfter = await publicClient.getBalance({ address: user1.account.address });

                expect(wethBalancesAfter.get(user1.account.address)).to.equal(wethBalancesBefore.get(user1.account.address), "User should not have spent WETH from their account");
                expect(wethBalancesAfter.get(wethVault.address)).to.equal(wethBalancesBefore.get(wethVault.address), "User should not have spent WETH from their vault deposits");
                expect(userETHBalanceAfter).to.equal(userETHBalanceBefore - totalAmountIn - gasUsed, "User should have spent ETH from their account")
                expect(ppgBalancesAfter.get(user1.account.address)).to.equal(ppgBalancesBefore.get(user1.account.address), "User should not have received uPPG to their account");
                expect(ppgBalancesAfter.get(ppgVault.address)).to.equal(
                    ppgBalancesBefore.get(ppgVault.address) + (totalAmountIn * initialPPGPrice / priceDenominator * (10_000n - swapFeeBips) / 10_000n), 
                    "uPPG should have been deposited into the vault, minus the swap fee"
                );
                expect(ppgBalancesAfter.get(feeReceiver)).to.equal(totalAmountIn * initialPPGPrice / priceDenominator * swapFeeBips / 10_000n, "Fee receiver should have received fee in uPPG");
                expect(userWETHVaultSharesAfter).to.equal(userWETHVaultSharesBefore, "User should not have fewer WETH Vault shares after the swap");
                expect(userPPGVaultSharesAfter).to.be.gt(userPPGVaultSharesBefore, "User should have more PPG Vault shares after the swap");
            });

            it("Deposit to Vault (ERC20)", async function () {
                const { user1, wasabiRouter, wethVault, weth, publicClient, feeReceiver, totalAmountIn } = await loadFixture(deployPoolsAndRouterMockEnvironment);

                const wethBalancesBefore = await takeBalanceSnapshot(publicClient, weth.address, user1.account.address, wethVault.address, feeReceiver);
                const userWETHVaultSharesBefore = await wethVault.read.balanceOf([user1.account.address]);

                // Approve WasabiRouter for WETH transfer
                await weth.write.approve(
                    [wasabiRouter.address, totalAmountIn],
                    { account: user1.account }
                );
                // Call swapTokenToVault with tokenIn == tokenOut and empty calldata
                await wasabiRouter.write.swapTokenToVault(
                    [totalAmountIn, weth.address, weth.address, "0x"],
                    { account: user1.account }
                );

                const wethBalancesAfter = await takeBalanceSnapshot(publicClient, weth.address, user1.account.address, wethVault.address, feeReceiver);
                const userWETHVaultSharesAfter = await wethVault.read.balanceOf([user1.account.address]);

                expect(wethBalancesAfter.get(user1.account.address)).to.equal(wethBalancesBefore.get(user1.account.address) - totalAmountIn, "User should have spent WETH from their account");
                expect(wethBalancesAfter.get(wethVault.address)).to.equal(wethBalancesBefore.get(wethVault.address) + totalAmountIn, "User should have deposited WETH into their vault");
                expect(wethBalancesAfter.get(feeReceiver)).to.equal(wethBalancesBefore.get(feeReceiver), "Fee receiver should not have received fee for deposit");
                expect(userWETHVaultSharesAfter).to.be.gt(userWETHVaultSharesBefore, "User should have more WETH Vault shares after the deposit");
            });

            it("Deposit to Vault (ETH)", async function () {
                const { user1, wasabiRouter, wethVault, weth, publicClient, feeReceiver, totalAmountIn } = await loadFixture(deployPoolsAndRouterMockEnvironment);

                const wethBalancesBefore = await takeBalanceSnapshot(publicClient, weth.address, user1.account.address, wethVault.address, feeReceiver);
                const userWETHVaultSharesBefore = await wethVault.read.balanceOf([user1.account.address]);
                const userETHBalanceBefore = await publicClient.getBalance({ address: user1.account.address });

                // Call swapTokenToVault with tokenIn == tokenOut and empty calldata
                const hash = await wasabiRouter.write.swapTokenToVault(
                    [totalAmountIn, weth.address, weth.address, "0x"],
                    { account: user1.account, value: totalAmountIn }
                );
                const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);

                const wethBalancesAfter = await takeBalanceSnapshot(publicClient, weth.address, user1.account.address, wethVault.address, feeReceiver);
                const userWETHVaultSharesAfter = await wethVault.read.balanceOf([user1.account.address]);
                const userETHBalanceAfter = await publicClient.getBalance({ address: user1.account.address });

                expect(wethBalancesAfter.get(user1.account.address)).to.equal(wethBalancesBefore.get(user1.account.address), "User should not have spent WETH from their account");
                expect(userETHBalanceAfter).to.equal(userETHBalanceBefore - totalAmountIn - gasUsed, "User should have spent ETH from their account");
                expect(wethBalancesAfter.get(wethVault.address)).to.equal(wethBalancesBefore.get(wethVault.address) + totalAmountIn, "User should have deposited WETH into their vault");
                expect(wethBalancesAfter.get(feeReceiver)).to.equal(wethBalancesBefore.get(feeReceiver), "Fee receiver should not have received fee for deposit");
                expect(userWETHVaultSharesAfter).to.be.gt(userWETHVaultSharesBefore, "User should have more WETH Vault shares after the deposit");
            });
        });
    });

    describe("Validations", function () {
        describe("Initialization Validations", function () {
            it("InvalidInitialization", async function () {
                const { wasabiRouter, user1, wasabiLongPool, wasabiShortPool, weth, manager, mockSwapRouter, feeReceiver, swapFeeBips } = await loadFixture(deployPoolsAndRouterMockEnvironment);
                await expect(wasabiRouter.write.initialize([wasabiLongPool.address, wasabiShortPool.address, weth.address, manager.address, mockSwapRouter.address, feeReceiver, swapFeeBips], { account: user1.account })).to.be.rejectedWith("InvalidInitialization");
            });

            it("NotInitializing", async function () {
                const { wasabiRouter, user1, wasabiLongPool, wasabiShortPool, weth, manager, mockSwapRouter, feeReceiver, swapFeeBips } = await loadFixture(deployPoolsAndRouterMockEnvironment);
                await expect(wasabiRouter.write.__WasabiRouter_init([wasabiLongPool.address, wasabiShortPool.address, weth.address, manager.address, mockSwapRouter.address, feeReceiver, swapFeeBips], { account: user1.account })).to.be.rejectedWith("NotInitializing");
            });
        });

        describe("Open Position Validations", function () {
            it("InvalidPool", async function () {
                const { wasabiRouter, user1, longOpenPositionRequest, longOpenSignature, uPPG } = await loadFixture(deployPoolsAndRouterMockEnvironment);
                await expect(wasabiRouter.write.openPosition([uPPG.address, longOpenPositionRequest, longOpenSignature], { account: user1.account })).to.be.rejectedWith("InvalidPool");
            });

            it("InvalidSignature", async function () {
                const { user1, orderExecutor, wasabiRouter, wasabiShortPool, wethVault, shortOpenPositionRequest, shortOpenSignature } = await loadFixture(deployPoolsAndRouterMockEnvironment);

                // Deposit into WETH Vault
                await wethVault.write.depositEth(
                    [user1.account.address],
                    { value: parseEther("50"), account: user1.account }
                );

                const routerRequest = { ...shortOpenPositionRequest, functionCallDataList: [] };
                const traderSignature = await signOpenPositionRequest(user1, "WasabiRouter", wasabiRouter.address, routerRequest);
                const badSignature = { ...traderSignature, v: traderSignature.v + 2 };

                await expect(wasabiRouter.write.openPosition([wasabiShortPool.address, shortOpenPositionRequest, shortOpenSignature, badSignature, 0n], { account: orderExecutor.account })).to.be.rejectedWith("InvalidSignature");
            });

            it("AccessManagerUnauthorizedAccount", async function () {
                const { wasabiRouter, wasabiLongPool, user1, longOpenPositionRequest, longOpenSignature, wethVault } = await loadFixture(deployPoolsAndRouterMockEnvironment);

                // Deposit into WETH Vault
                await wethVault.write.depositEth(
                    [user1.account.address],
                    { value: parseEther("50"), account: user1.account }
                );

                const routerRequest = { ...longOpenPositionRequest, functionCallDataList: [] };
                const traderSignature = await signOpenPositionRequest(user1, "WasabiRouter", wasabiRouter.address, routerRequest);

                await expect(wasabiRouter.write.openPosition([wasabiLongPool.address, longOpenPositionRequest, longOpenSignature, traderSignature, 0n], { account: user1.account })).to.be.rejectedWith("AccessManagerUnauthorizedAccount");
            });

            it("ERC4626ExceededMaxWithdraw", async function () {
                const { wasabiRouter, wasabiLongPool, user1, orderExecutor, longOpenPositionRequest, longOpenSignature } = await loadFixture(deployPoolsAndRouterMockEnvironment);

                // Do not deposit into WETH Vault

                const routerRequest = { ...longOpenPositionRequest, functionCallDataList: [] };
                const traderSignature = await signOpenPositionRequest(user1, "WasabiRouter", wasabiRouter.address, routerRequest);

                await expect(wasabiRouter.write.openPosition([wasabiLongPool.address, longOpenPositionRequest, longOpenSignature, traderSignature, 0n], { account: orderExecutor.account })).to.be.rejectedWith("ERC4626ExceededMaxWithdraw");
            });
        });

        describe("Swap Validations", function () {
            it("InvalidETHReceived", async function () {
                const { createExactInRouterSwapData, user1, wasabiRouter, ppgVault, weth, uPPG, swapFeeBips, totalAmountIn } = await loadFixture(deployPoolsAndRouterMockEnvironment);

                const amount = parseEther("10");
                await uPPG.write.mint([user1.account.address, amount]);
                await uPPG.write.approve([ppgVault.address, amount], { account: user1.account });
                await ppgVault.write.deposit([amount, user1.account.address], { account: user1.account });

                // ETH sent with tokenIn != WETH
                await expect(wasabiRouter.write.swapTokenToVault(
                    [totalAmountIn, uPPG.address, weth.address, "0x"],
                    { account: user1.account, value: totalAmountIn }
                )).to.be.rejectedWith("InvalidETHReceived");

                // Transfer ETH to WasabiRouter directly
                await expect(user1.sendTransaction({
                    to: wasabiRouter.address,
                    value: totalAmountIn
                })).to.be.rejectedWith("InvalidETHReceived");
            });

            it("InvalidVault", async function () {
                const { createExactInRouterSwapData, user1, wasabiRouter, wethVault, wethAddress, usdc, swapFeeBips, totalAmountIn } = await loadFixture(deployPoolsAndRouterMockEnvironment);

                // Deposit into WETH Vault
                await wethVault.write.depositEth(
                    [user1.account.address], 
                    { value: parseEther("50"), account: user1.account }
                );

                let swapCalldata = await createExactInRouterSwapData({amount: totalAmountIn, tokenIn: wethAddress, tokenOut: usdc.address, swapRecipient: wasabiRouter.address, swapFee: swapFeeBips});
                await expect(wasabiRouter.write.swapVaultToVault(
                    [totalAmountIn, wethAddress, usdc.address, swapCalldata],
                    { account: user1.account }
                )).to.be.rejectedWith("InvalidVault");

                swapCalldata = await createExactInRouterSwapData({amount: totalAmountIn, tokenIn: usdc.address, tokenOut: wethAddress, swapRecipient: wasabiRouter.address, swapFee: swapFeeBips});
                await expect(wasabiRouter.write.swapVaultToVault(
                    [totalAmountIn, usdc.address, wethAddress, swapCalldata],
                    { account: user1.account }
                )).to.be.rejectedWith("InvalidVault");
            });

            it("InvalidFeeBips", async function () {
                const { wasabiRouter } = await loadFixture(deployPoolsAndRouterMockEnvironment);

                await expect(wasabiRouter.write.setWithdrawFeeBips([10001n])).to.be.rejectedWith("InvalidFeeBips");
            });
        });
    });
});