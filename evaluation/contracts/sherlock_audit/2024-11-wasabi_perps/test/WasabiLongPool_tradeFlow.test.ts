import {
    time,
    loadFixture,
} from "@nomicfoundation/hardhat-toolbox-viem/network-helpers";
import {encodeFunctionData, zeroAddress, parseEther, parseUnits} from "viem";
import { expect } from "chai";
import { Position, formatEthValue, getEventPosition, PayoutType, OpenPositionRequest, FunctionCallData, getFee, getEmptyPosition, AddCollateralRequest, RemoveCollateralRequest } from "./utils/PerpStructUtils";
import { getApproveAndSwapFunctionCallData } from "./utils/SwapUtils";
import { deployLongPoolMockEnvironment, deployPoolsAndRouterMockEnvironment } from "./fixtures";
import { getBalance, takeBalanceSnapshot } from "./utils/StateUtils";
import { signAddCollateralRequest, signOpenPositionRequest, signRemoveCollateralRequest } from "./utils/SigningUtils";


describe("WasabiLongPool - Trade Flow Test", function () {
    describe("Open Position", function () {
        it("Open Position", async function () {
            const { wasabiLongPool, tradeFeeValue, uPPG, weth, user1, openPositionRequest, totalAmountIn, signature, publicClient, feeReceiver } = await loadFixture(deployLongPoolMockEnvironment);

            const feeReceiverBalanceBefore = await weth.read.balanceOf([feeReceiver]);

            const hash = await wasabiLongPool.write.openPosition([openPositionRequest, signature], { value: totalAmountIn, account: user1.account });

            const feeReceiverBalanceAfter = await weth.read.balanceOf([feeReceiver]);

            const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed);
            console.log('gas used to open', gasUsed);

            const events = await wasabiLongPool.getEvents.PositionOpened();
            expect(events).to.have.lengthOf(1);
            const eventData = events[0].args;
            expect(eventData.positionId).to.equal(openPositionRequest.id);
            expect(eventData.downPayment).to.equal(totalAmountIn - eventData.feesToBePaid!);
            expect(eventData.principal).to.equal(openPositionRequest.principal);
            expect(eventData.collateralAmount).to.equal(await uPPG.read.balanceOf([wasabiLongPool.address]));
            expect(eventData.collateralAmount).to.greaterThanOrEqual(openPositionRequest.minTargetAmount);
            expect(eventData.feesToBePaid!).to.equal(feeReceiverBalanceAfter - feeReceiverBalanceBefore);
        });

        it("Open and Increase Position", async function () {
            const { wasabiLongPool, mockSwap, wethAddress, uPPG, user1, totalAmountIn, totalSize, initialPrice, priceDenominator, orderSigner, contractName, sendDefaultOpenPositionRequest } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();

            await time.increase(86400n); // 1 day later

            const functionCallDataList: FunctionCallData[] =
                    getApproveAndSwapFunctionCallData(mockSwap.address, wethAddress, uPPG.address, totalSize);
            const openPositionRequest: OpenPositionRequest = {
                id: position.id,
                currency: position.currency,
                targetCurrency: position.collateralCurrency,
                downPayment: position.downPayment,
                principal: position.principal,
                minTargetAmount: totalSize * initialPrice / priceDenominator,
                expiration: BigInt(await time.latest()) + 86400n,
                fee: position.feesToBePaid,
                functionCallDataList,
                existingPosition: position,
                referrer: zeroAddress
            };
            const signature = await signOpenPositionRequest(orderSigner, contractName, wasabiLongPool.address, openPositionRequest);

            // Increase Position
            await wasabiLongPool.write.openPosition([openPositionRequest, signature], { value: totalAmountIn, account: user1.account });

            const events = await wasabiLongPool.getEvents.PositionIncreased();
            expect(events).to.have.lengthOf(1);
            const eventData = events[0].args;
            expect(eventData.id).to.equal(position.id);
            expect(eventData.downPaymentAdded).to.equal(totalAmountIn - eventData.feesAdded!);
            expect(eventData.principalAdded).to.equal(openPositionRequest.principal);
            expect(eventData.collateralAdded! + position.collateralAmount).to.equal(await uPPG.read.balanceOf([wasabiLongPool.address]));
            expect(eventData.collateralAdded).to.greaterThanOrEqual(openPositionRequest.minTargetAmount);
        });

        it("Open Position and Add Collateral", async function () {
            const { wasabiLongPool, weth, vault, user1, downPayment, orderSigner, contractName, sendDefaultOpenPositionRequest, computeMaxInterest } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();
            const totalAssetValueBefore = await vault.read.totalAssetValue();

            await time.increase(86400n); // 1 day later

            const interest = await computeMaxInterest(position);
            const addCollateralRequest: AddCollateralRequest = {
                amount: downPayment,
                interest,
                expiration: BigInt(await time.latest()) + 86400n,
                position,
            };
            const signature = await signAddCollateralRequest(orderSigner, contractName, wasabiLongPool.address, addCollateralRequest);

            // Add Collateral
            const vaultBalanceBefore = await weth.read.balanceOf([vault.address]);
            await wasabiLongPool.write.addCollateral([addCollateralRequest, signature], { value: position.downPayment, account: user1.account });
            const vaultBalanceAfter = await weth.read.balanceOf([vault.address]);

            expect(vaultBalanceAfter).to.equal(vaultBalanceBefore + downPayment, "Vault should have received principal and interest");

            const events = await wasabiLongPool.getEvents.CollateralAdded();
            expect(events).to.have.lengthOf(1);
            const eventData = events[0].args;
            expect(eventData.id).to.equal(position.id);
            expect(eventData.downPaymentAdded).to.equal(downPayment - interest);
            expect(eventData.collateralAdded).to.equal(0n);
            expect(eventData.principalReduced).to.equal(downPayment - interest);
            expect(eventData.interestPaid).to.equal(interest);
            const totalAssetValueAfter = await vault.read.totalAssetValue();
            expect(totalAssetValueAfter).to.equal(totalAssetValueBefore + interest);
        });
    });

    describe("Close Position", function () {
        it("Price Not Changed", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, computeMaxInterest, publicClient, wasabiLongPool, user1, uPPG, feeReceiver, wethAddress, vault } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();

            await time.increase(86400n); // 1 day later
            // await time.setNextBlockTimestamp(await time.latest() + 100);

            // Close Position
            const { request, signature } = await createSignedClosePositionRequest({ position, interest: 0n });

            const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceBefore = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });
            const feeReceiverSharesBefore = await vault.read.balanceOf([feeReceiver]);

            const maxInterest = await computeMaxInterest(position);
            
            const hash = await wasabiLongPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account });

            const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceAfter = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });
            const feeReceiverSharesAfter = await vault.read.balanceOf([feeReceiver]);

            // Checks
            const events = await wasabiLongPool.getEvents.PositionClosed();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;
            const totalFeesPaid = closePositionEvent.feeAmount!;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal);
            expect(closePositionEvent.interestPaid!).to.equal(maxInterest, "If given interest value is 0, should use max interest");
            expect(await uPPG.read.balanceOf([wasabiLongPool.address])).to.equal(0n, "Pool should not have any collateral left");

            expect(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!).to.equal(vaultBalanceAfter);

            const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.feeAmount! - position.downPayment;
            expect(totalReturn).to.equal(0, "Total return should be 0 on no price change");

            // Check trader has been paid
            const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
            expect(traderBalanceAfter - traderBalanceBefore + gasUsed).to.equal(closePositionEvent.payout!);

            const gasAmount = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed);
            console.log('gas used to close', gasAmount);

            // Check fees have been paid
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
            expect(feeReceiverSharesAfter - feeReceiverSharesBefore).to.equal(maxInterest / 10n);
        });

        it("Use Custom Interest", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, computeMaxInterest, publicClient, wasabiLongPool, user1, uPPG, feeReceiver, wethAddress, vault } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();

            await time.increase(86400n); // 1 day later

            const interest = (await computeMaxInterest(position)) / 2n;
            // Close Position
            const { request, signature } = await createSignedClosePositionRequest({
                position,
                interest
            });

            const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceBefore = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });
            const feeReceiverSharesBefore = await vault.read.balanceOf([feeReceiver]);

            const hash = await wasabiLongPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account });

            const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceAfter = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });
            const feeReceiverSharesAfter = await vault.read.balanceOf([feeReceiver]);

            // Checks
            const events = await wasabiLongPool.getEvents.PositionClosed();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;
            const totalFeesPaid = closePositionEvent.feeAmount!;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal);
            expect(closePositionEvent.interestPaid!).to.equal(interest);
            expect(await uPPG.read.balanceOf([wasabiLongPool.address])).to.equal(0n, "Pool should not have any collateral left");

            expect(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!).to.equal(vaultBalanceAfter);

            const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.feeAmount! - position.downPayment;
            expect(totalReturn).to.equal(0, "Total return should be 0 on no price change");

            // Check trader has been paid
            const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
            expect(traderBalanceAfter - traderBalanceBefore + gasUsed).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
            expect(feeReceiverSharesAfter - feeReceiverSharesBefore).to.equal(interest / 10n);
        });

        it("Price Increased", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, owner, publicClient, wasabiLongPool, user1, uPPG, mockSwap, feeReceiver, initialPrice, wethAddress, vault } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 2n]); // Price doubled

            // Close Position
            const { request, signature } = await createSignedClosePositionRequest({position});

            const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceBefore = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });

            const hash = await wasabiLongPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account });

            const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceAfter = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });

            // Checks
            const events = await wasabiLongPool.getEvents.PositionClosed();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;
            const totalFeesPaid = closePositionEvent.feeAmount!;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal);
            expect(await uPPG.read.balanceOf([wasabiLongPool.address])).to.equal(0n, "Pool should not have any collateral left");

            expect(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!).to.equal(vaultBalanceAfter);

            const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.feeAmount!;
            expect(totalReturn).to.equal(position.downPayment * 5n, "on 2x price increase, total return should be 5x down payment");

            // Check trader has been paid
            const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
            expect(traderBalanceAfter - traderBalanceBefore + gasUsed).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);

            console.log('gas used to close', formatEthValue(gasUsed, 8));
        });

        it("Price Increased - Deposit Into Vault", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, owner, publicClient, wasabiLongPool, user1, uPPG, mockSwap, feeReceiver, initialPrice, wethAddress, vault } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 2n]); // Price doubled

            // Close Position
            const { request, signature } = await createSignedClosePositionRequest({position});

            const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceBefore = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceBefore = await getBalance(publicClient, wethAddress, feeReceiver);
            const vaultSharesBefore = await vault.read.balanceOf([user1.account.address]);

            const hash = await wasabiLongPool.write.closePosition([PayoutType.VAULT_DEPOSIT, request, signature], { account: user1.account });

            const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceAfter = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceAfter = await getBalance(publicClient, wethAddress, feeReceiver);
            const vaultSharesAfter = await vault.read.balanceOf([user1.account.address]);

            // Checks
            const events = await wasabiLongPool.getEvents.PositionClosed();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;
            const totalFeesPaid = closePositionEvent.feeAmount!;
            const expectedNewVaultShares = await vault.read.convertToShares([closePositionEvent.payout!]);

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal);
            expect(await uPPG.read.balanceOf([wasabiLongPool.address])).to.equal(0n, "Pool should not have any collateral left");

            const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.feeAmount!;
            expect(totalReturn).to.equal(position.downPayment * 5n, "on 2x price increase, total return should be 4x down payment");

            // Check trader has received vault shares instead of payout
            expect(vaultSharesAfter).to.equal(vaultSharesBefore + expectedNewVaultShares, "Trader should have received vault shares instead of payout");
            expect(vaultBalanceAfter).to.equal(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid! + closePositionEvent.payout!, "Vault should have received principal + interest + payout");
            const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
            expect(traderBalanceAfter - traderBalanceBefore + gasUsed).to.equal(0n);

            // Check fees have been paid
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid, "Fees should have been paid to fee receiver");

            console.log('gas used to close', formatEthValue(gasUsed, 8));
        });

        it("Price Decreased", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, publicClient, wasabiLongPool, user1, uPPG, mockSwap, feeReceiver, initialPrice, wethAddress, vault } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 8n / 10n]); // Price fell 20%

            // Close Position
            const { request, signature } = await createSignedClosePositionRequest({position});

            const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceBefore = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });

            const hash = await wasabiLongPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account });

            const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceAfter = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });

            // Checks
            const events = await wasabiLongPool.getEvents.PositionClosed();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;
            const totalFeesPaid = closePositionEvent.feeAmount!;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal);
            expect(await uPPG.read.balanceOf([wasabiLongPool.address])).to.equal(0n, "Pool should not have any collateral left");

            expect(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!).to.equal(vaultBalanceAfter);

            const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.feeAmount! - position.downPayment;
            expect(totalReturn).to.equal(position.downPayment / -5n * 4n, "on 20% price decrease, total return should be -20% * leverage (4) * down payment");

            // Check trader has been paid
            const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
            expect(traderBalanceAfter - traderBalanceBefore + gasUsed).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        describe("Partial Close", function () {
            it("Price not changed", async function () {
                const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, computeMaxInterest, publicClient, wasabiLongPool, user1, uPPG, feeReceiver, wethAddress, vault } = await loadFixture(deployLongPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                await time.increase(86400n); // 1 day later

                // Close Half of the Position
                const closeAmountDenominator = 2n;
                const interest = await computeMaxInterest(position) / closeAmountDenominator;
                const { request, signature } = await createSignedClosePositionRequest({ position, interest, amount: position.collateralAmount / closeAmountDenominator });

                const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
                const vaultBalanceBefore = await getBalance(publicClient, wethAddress, vault.address);
                const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });
                
                const hash = await wasabiLongPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account });

                const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
                const vaultBalanceAfter = await getBalance(publicClient, wethAddress, vault.address);
                const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });

                // Checks
                const events = await wasabiLongPool.getEvents.PositionDecreased();
                expect(events).to.have.lengthOf(1);
                const closePositionEvent = events[0].args;
                const totalFeesPaid = closePositionEvent.closeFee!;

                expect(closePositionEvent.id).to.equal(position.id);
                expect(closePositionEvent.principalRepaid!).to.equal(position.principal / closeAmountDenominator, "Half of the principal should be repaid");
                expect(closePositionEvent.downPaymentReduced!).to.equal(position.downPayment / closeAmountDenominator, "Down payment should be reduced by half");
                expect(closePositionEvent.collateralReduced!).to.equal(position.collateralAmount / closeAmountDenominator, "Half of the collateral should be spent");
                expect(closePositionEvent.interestPaid!).to.equal(interest, "Prorated interest should be paid");
                expect(await uPPG.read.balanceOf([wasabiLongPool.address])).to.equal(position.collateralAmount / closeAmountDenominator, "Pool should have half of the collateral left");

                expect(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!).to.equal(vaultBalanceAfter);

                const downPaymentReduced = position.downPayment / closeAmountDenominator;
                const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.closeFee! - downPaymentReduced;
                expect(totalReturn).to.equal(0, "Total return should be 0 on no price change");

                // Check trader has been paid
                const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
                expect(traderBalanceAfter - traderBalanceBefore + gasUsed).to.equal(closePositionEvent.payout!);

                const gasAmount = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed);
                console.log('gas used to close', gasAmount);

                // Check fees have been paid
                expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
            });

            it("Price Increased", async function () {
                const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, computeMaxInterest, owner, publicClient, wasabiLongPool, user1, uPPG, mockSwap, feeReceiver, initialPrice, wethAddress, vault } = await loadFixture(deployLongPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                await time.increase(86400n); // 1 day later
                await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 2n]); // Price doubled

                // Close Half of the Position
                const closeAmountDenominator = 2n;
                const interest = await computeMaxInterest(position) / closeAmountDenominator;
                const { request, signature } = await createSignedClosePositionRequest({position, interest, amount: position.collateralAmount / closeAmountDenominator});

                const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
                const vaultBalanceBefore = await getBalance(publicClient, wethAddress, vault.address);
                const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });

                const hash = await wasabiLongPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account });

                const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
                const vaultBalanceAfter = await getBalance(publicClient, wethAddress, vault.address);
                const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });

                // Checks
                const events = await wasabiLongPool.getEvents.PositionDecreased();
                expect(events).to.have.lengthOf(1);
                const closePositionEvent = events[0].args;
                const totalFeesPaid = closePositionEvent.closeFee!;

                expect(closePositionEvent.id).to.equal(position.id);
                expect(closePositionEvent.principalRepaid!).to.equal(position.principal / closeAmountDenominator, "Half of the principal should be repaid");
                expect(await uPPG.read.balanceOf([wasabiLongPool.address])).to.equal(position.collateralAmount / closeAmountDenominator, "Pool should have half of the collateral left");

                expect(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!).to.equal(vaultBalanceAfter);

                const downPaymentReduced = position.downPayment / closeAmountDenominator;
                const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.closeFee! - downPaymentReduced;
                expect(totalReturn).to.equal(downPaymentReduced * 4n, "on 2x price increase, total return should be 4x adjusted down payment");

                // Check trader has been paid
                const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
                expect(traderBalanceAfter - traderBalanceBefore + gasUsed).to.equal(closePositionEvent.payout!);

                // Check fees have been paid
                expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
            });

            it("Price Decreased", async function () {
                const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, computeMaxInterest, publicClient, wasabiLongPool, user1, uPPG, mockSwap, feeReceiver, initialPrice, wethAddress, vault } = await loadFixture(deployLongPoolMockEnvironment);
    
                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();
    
                await time.increase(86400n); // 1 day later
                await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 8n / 10n]); // Price fell 20%
    
                // Close Half of the Position
                const closeAmountDenominator = 2n;
                const interest = await computeMaxInterest(position) / closeAmountDenominator;
                const { request, signature } = await createSignedClosePositionRequest({position, interest, amount: position.collateralAmount / closeAmountDenominator});
    
                const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
                const vaultBalanceBefore = await getBalance(publicClient, wethAddress, vault.address);
                const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });
    
                const hash = await wasabiLongPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account });
    
                const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
                const vaultBalanceAfter = await getBalance(publicClient, wethAddress, vault.address);
                const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });
    
                // Checks
                const events = await wasabiLongPool.getEvents.PositionDecreased();
                expect(events).to.have.lengthOf(1);
                const closePositionEvent = events[0].args;
                const totalFeesPaid = closePositionEvent.closeFee!;
    
                expect(closePositionEvent.id).to.equal(position.id);
                expect(closePositionEvent.principalRepaid!).to.equal(position.principal / closeAmountDenominator, "Half of the principal should be repaid");
                expect(await uPPG.read.balanceOf([wasabiLongPool.address])).to.equal(position.collateralAmount / closeAmountDenominator, "Pool should have half of the collateral left");
    
                expect(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!).to.equal(vaultBalanceAfter);
    
                const downPaymentReduced = position.downPayment / closeAmountDenominator;
                const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.closeFee! - downPaymentReduced;
                expect(totalReturn).to.equal(downPaymentReduced / -5n * 4n, "on 20% price decrease, total return should be -20% * leverage (4) * adjusted down payment");
    
                // Check trader has been paid
                const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
                expect(traderBalanceAfter - traderBalanceBefore + gasUsed).to.equal(closePositionEvent.payout!);
    
                // Check fees have been paid
                expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
            });
        });
    });

    describe("Liquidate Position", function () {
        it("liquidate", async function () {
            const { sendDefaultOpenPositionRequest, computeMaxInterest, vault, publicClient, wasabiLongPool, user1, uPPG, mockSwap, feeReceiver, liquidationFeeReceiver, wethAddress, liquidator, computeLiquidationPrice } = await loadFixture(deployLongPoolMockEnvironment);
            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();

            await time.increase(86400n); // 1 day later

            // Liquidate Position
            const functionCallDataList = getApproveAndSwapFunctionCallData(mockSwap.address, uPPG.address, wethAddress, position.collateralAmount);

            const interest = await computeMaxInterest(position);
            const liquidationPrice = await computeLiquidationPrice(position);

            // If the liquidation price is not reached, should revert
            await mockSwap.write.setPrice([uPPG.address, wethAddress, liquidationPrice + 1n]); 
            await expect(wasabiLongPool.write.liquidatePosition([PayoutType.UNWRAPPED, interest, position, functionCallDataList, zeroAddress], { account: liquidator.account }))
                .to.be.rejectedWith("LiquidationThresholdNotReached", "Cannot liquidate position if liquidation price is not reached");

            // Liquidate
            await mockSwap.write.setPrice([uPPG.address, wethAddress, liquidationPrice]); 

            const balancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, vault.address, user1.account.address, feeReceiver, liquidationFeeReceiver);
            const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
            const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });
            const liquidationFeeReceiverBalanceBefore = await publicClient.getBalance({address: liquidationFeeReceiver });            

            const hash = await wasabiLongPool.write.liquidatePosition([PayoutType.UNWRAPPED, interest, position, functionCallDataList, zeroAddress], { account: liquidator.account });

            const balancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, vault.address, user1.account.address, feeReceiver);
            const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
            const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });
            const liquidationFeeReceiverBalanceAfter = await publicClient.getBalance({address: liquidationFeeReceiver });
            // Checks
            const events = await wasabiLongPool.getEvents.PositionLiquidated();
            expect(events).to.have.lengthOf(1);
            const liquidatePositionEvent = events[0].args;
            const totalFeesPaid = liquidatePositionEvent.feeAmount!;

            expect(liquidatePositionEvent.id).to.equal(position.id);
            expect(liquidatePositionEvent.principalRepaid!).to.equal(position.principal);
            expect(await uPPG.read.balanceOf([wasabiLongPool.address])).to.equal(0n, "Pool should not have any collateral left");

            expect(balancesBefore.get(vault.address) + liquidatePositionEvent.principalRepaid! + liquidatePositionEvent.interestPaid!).to.equal(balancesAfter.get(vault.address)!);

            // Check trader has been paid
            expect(traderBalanceAfter - traderBalanceBefore).to.equal(liquidatePositionEvent.payout!);

            // Check fees have been paid
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);

            // Check liquidation fee receiver balance
            const liquidationFeeExpected = position.downPayment * 5n / 100n;
            expect(liquidationFeeReceiverBalanceAfter - liquidationFeeReceiverBalanceBefore).to.equal(liquidationFeeExpected);
        });

        it("liquidate with bad debt", async function () {
            const { sendDefaultOpenPositionRequest, computeMaxInterest, owner, publicClient, wasabiLongPool, user1, uPPG, mockSwap, feeReceiver, liquidationFeeReceiver, wethAddress, liquidator, computeLiquidationPrice } = await loadFixture(deployLongPoolMockEnvironment);
            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();
    
            await time.increase(86400n); // 1 day later
    
            // Liquidate Position
            const functionCallDataList = getApproveAndSwapFunctionCallData(mockSwap.address, uPPG.address, wethAddress, position.collateralAmount);
    
            const interest = await computeMaxInterest(position);
            const liquidationPrice = await computeLiquidationPrice(position);
    
            // If the liquidation price is not reached, should revert
            await mockSwap.write.setPrice([uPPG.address, wethAddress, liquidationPrice + 1n]); 
            await expect(wasabiLongPool.write.liquidatePosition([PayoutType.UNWRAPPED, interest, position, functionCallDataList, zeroAddress], { account: liquidator.account }))
                .to.be.rejectedWith("LiquidationThresholdNotReached", "Cannot liquidate position if liquidation price is not reached");
    
            await mockSwap.write.setPrice([uPPG.address, wethAddress, liquidationPrice / 2n]); 
    
            await wasabiLongPool.write.liquidatePosition([PayoutType.UNWRAPPED, interest, position, functionCallDataList, zeroAddress], { account: liquidator.account });
            // Checks for no payout
            const events = await wasabiLongPool.getEvents.PositionLiquidated();
            expect(events).to.have.lengthOf(1);
            const liquidatePositionEvent = events[0].args;
            expect(liquidatePositionEvent.payout!).to.equal(0n);
            expect(liquidatePositionEvent.principalRepaid!).to.lessThan(position.principal, "Principal should be less than the original principal due to bad debt");
        });
    });

    describe("Remove Collateral", function () {
        it("Price Increased - Remove Profit", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, wasabiLongPool, user1, orderSigner, uPPG, mockSwap, initialPrice, maxLeverage, weth, hasher } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, weth.address, initialPrice * 2n]); // Price doubled

            // Remove Collateral
            const amount = position.downPayment * (maxLeverage - 100n) / 100n - position.principal;
            const removeCollateralRequest: RemoveCollateralRequest = {
                amount: amount,
                expiration: BigInt(await time.latest()) + 86400n,
                position: position
            };
            const removeCollateralSignature = await signRemoveCollateralRequest(orderSigner, "WasabiLongPool", wasabiLongPool.address, removeCollateralRequest);

            const userBalanceBefore = await weth.read.balanceOf([user1.account.address]);
            await wasabiLongPool.write.removeCollateral([removeCollateralRequest, removeCollateralSignature], { account: user1.account });
            const userBalanceAfter = await weth.read.balanceOf([user1.account.address]);

            // Remove Collateral Checks
            expect(userBalanceAfter - userBalanceBefore).to.equal(amount);

            const events = await wasabiLongPool.getEvents.CollateralRemoved();
            expect(events).to.have.lengthOf(1);
            const collateralRemovedEvent = events[0].args;
            expect(collateralRemovedEvent.id).to.equal(position.id);
            expect(collateralRemovedEvent.principalAdded).to.equal(amount);
            expect(collateralRemovedEvent.downPaymentReduced).to.equal(0n);
            expect(collateralRemovedEvent.collateralReduced).to.equal(0n);

            position.principal += amount;
            const hashedPosition = await hasher.read.hashPosition([position]);
            expect(await wasabiLongPool.read.positions([position.id])).to.equal(hashedPosition);

            // Close Position
            const { request, signature } = await createSignedClosePositionRequest({position});

            await wasabiLongPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account });

            // Close Position Checks
            const positionClosedEvents = await wasabiLongPool.getEvents.PositionClosed();
            expect(positionClosedEvents).to.have.lengthOf(1);
            const closePositionEvent = positionClosedEvents[0].args;
            const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.feeAmount!;
            expect(totalReturn).to.equal(position.downPayment * 5n - amount, "on 2x price increase, total return should be 5x down payment minus the amount removed");
        });
    });

    describe("Interest Recording", function () {
        it("Record Interest with one position", async function () {
            const { sendDefaultOpenPositionRequest, computeMaxInterest, liquidator, owner, publicClient, wasabiLongPool, vault, hasher } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();

            await time.increase(86400n); // 1 day later

            const vaultAssetsBefore = await vault.read.totalAssets();
            const feeReceiverSharesBefore = await vault.read.balanceOf([owner.account.address]);

            // Record Interest
            const interest = await computeMaxInterest(position);
            const hash = await wasabiLongPool.write.recordInterest([[position], [interest], []], { account: liquidator.account });
            const timestamp = await time.latest();

            const vaultAssetsAfter = await vault.read.totalAssets();
            const feeReceiverSharesAfter = await vault.read.balanceOf([owner.account.address]);

            const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed);
            console.log('gas used to record interest for 1 position ', gasUsed);

            const events = await wasabiLongPool.getEvents.InterestPaid();
            expect(events).to.have.lengthOf(1);
            const interestPaidEvent = events[0].args;
            expect(interestPaidEvent.id).to.equal(position.id);
            expect(interestPaidEvent.interestPaid).to.equal(interest);

            position.principal += interest;
            position.lastFundingTimestamp = BigInt(timestamp);
            const hashedPosition = await hasher.read.hashPosition([position]);
            expect(await wasabiLongPool.read.positions([position.id])).to.equal(hashedPosition);

            expect(vaultAssetsAfter).to.equal(vaultAssetsBefore + interest);
            expect(feeReceiverSharesAfter - feeReceiverSharesBefore).to.equal(interest / 10n);
        });


        it("Record Interest with 10 positions", async function () {
            const { sendDefaultOpenPositionRequest, computeMaxInterest, liquidator, owner, publicClient, wasabiLongPool, vault, hasher } = await loadFixture(deployLongPoolMockEnvironment);

            // Add more assets to the vault for borrowing
            await vault.write.depositEth([liquidator.account.address], {value: parseEther("100"), account: liquidator.account });

            // Open 10 positions
            const positions = [];
            for (let i = 0; i < 10; i++) {
                const {position} = await sendDefaultOpenPositionRequest(BigInt(i + 1));
                positions.push(position);
            }

            await time.increase(86400n); // 1 day later

            const vaultAssetsBefore = await vault.read.totalAssets();
            const feeReceiverSharesBefore = await vault.read.balanceOf([owner.account.address]);

            const interests = [];
            let totalInterest = 0n;
            for (let i = 0; i < 10; i++) {
                const interest = await computeMaxInterest(positions[i]);
                interests.push(interest);
                totalInterest += interest;
            }

            // Record Interest
            const hash = await wasabiLongPool.write.recordInterest([positions, interests, []], { account: liquidator.account });
            const timestamp = await time.latest();

            const vaultAssetsAfter = await vault.read.totalAssets();
            const feeReceiverSharesAfter = await vault.read.balanceOf([owner.account.address]);

            const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed);
            console.log('gas used to record interest for 10 positions', gasUsed);

            const events = await wasabiLongPool.getEvents.InterestPaid();
            expect(events).to.have.lengthOf(10);
            for (let i = 0; i < 10; i++) {
                const interestPaidEvent = events[i].args;
                const position = positions[i];
                const interest = interests[i];
                expect(interestPaidEvent.id).to.equal(position.id);
                expect(interestPaidEvent.interestPaid).to.equal(interest);

                position.principal += interest;
                position.lastFundingTimestamp = BigInt(timestamp);
                const hashedPosition = await hasher.read.hashPosition([position]);
                expect(await wasabiLongPool.read.positions([position.id])).to.equal(hashedPosition);
            }

            expect(vaultAssetsAfter).to.equal(vaultAssetsBefore + totalInterest);
            expect(feeReceiverSharesAfter - feeReceiverSharesBefore).to.equal(totalInterest / 10n);
        });

        it("Record Interest with 2 different USDC positions", async function () {
            const { getOpenPositionRequest, sendOpenPositionRequest, getTradeAmounts, computeMaxInterest, liquidator, publicClient, wasabiLongPool, vault, usdcVault, hasher, weth, usdc, user1 } = await loadFixture(deployLongPoolMockEnvironment);

            // Add more assets to the vault for borrowing
            await vault.write.depositEth([liquidator.account.address], {value: parseEther("100"), account: liquidator.account });
            await usdc.write.mint([liquidator.account.address, parseUnits("1000", 6)], { account: liquidator.account });
            await usdc.write.approve([usdcVault.address, parseUnits("1000", 6)], { account: liquidator.account });
            await usdcVault.write.deposit([parseUnits("1000", 6), liquidator.account.address], { account: liquidator.account });

            // Open 2 positions
            const positions = [];

            // 3x uPPG/USDC long
            const leverage1 = 3n;
            const totalAmountIn1 = parseUnits("50", 6);
            const { fee: fee1, downPayment: downPayment1, principal: principal1, minTargetAmount: minTargetAmount1 } = 
                await getTradeAmounts(leverage1, totalAmountIn1, usdc.address);
            const request1 = await getOpenPositionRequest({
                id: 1n,
                currency: usdc.address,
                principal: principal1,
                downPayment: downPayment1,
                minTargetAmount: minTargetAmount1,
                expiration: BigInt(await time.latest()) + 86400n,
                fee: fee1
            });
            await usdc.write.mint([user1.account.address, totalAmountIn1], { account: user1.account });
            await usdc.write.approve([wasabiLongPool.address, totalAmountIn1], { account: user1.account });
            const {position: position1} = await sendOpenPositionRequest(request1);
            positions.push(position1);

            // 5x WETH/USDC long
            const leverage2 = 5n;
            const totalAmountIn2 = parseUnits("10", 6);
            const { fee: fee2, downPayment: downPayment2, principal: principal2, minTargetAmount: minTargetAmount2 } = 
                await getTradeAmounts(leverage2, totalAmountIn2, usdc.address);
            const request2 = await getOpenPositionRequest({
                id: 2n,
                currency: usdc.address,
                targetCurrency: weth.address,
                principal: principal2,
                downPayment: downPayment2,
                minTargetAmount: minTargetAmount2,
                expiration: BigInt(await time.latest()) + 86400n,
                fee: fee2
            });
            await usdc.write.mint([user1.account.address, totalAmountIn2], { account: user1.account });
            await usdc.write.approve([wasabiLongPool.address, totalAmountIn2], { account: user1.account });
            const {position: position2} = await sendOpenPositionRequest(request2);
            positions.push(position2);

            await time.increase(86400n); // 1 day later

            const vaultAssetsBefore = await usdcVault.read.totalAssets();

            const interests = [];
            let totalInterest = 0n;
            for (let i = 0; i < 2; i++) {
                const interest = await computeMaxInterest(positions[i]);
                interests.push(interest);
                totalInterest += interest;
            }

            // Record Interest
            const hash = await wasabiLongPool.write.recordInterest([positions, interests, []], { account: liquidator.account });
            const timestamp = await time.latest();

            const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed);
            console.log('gas used to record interest for 2 positions', gasUsed);
            
            const events = await wasabiLongPool.getEvents.InterestPaid();
            expect(events).to.have.lengthOf(2);
            for (let i = 0; i < 2; i++) {
                const interestPaidEvent = events[i].args;
                const position = positions[i];
                const interest = interests[i];
                expect(interestPaidEvent.id).to.equal(position.id);
                expect(interestPaidEvent.interestPaid).to.equal(interest);

                position.principal += interest;
                position.lastFundingTimestamp = BigInt(timestamp);
                const hashedPosition = await hasher.read.hashPosition([position]);
                expect(await wasabiLongPool.read.positions([position.id])).to.equal(hashedPosition);
            }

            const vaultAssetsAfter = await usdcVault.read.totalAssets();
            expect(vaultAssetsAfter).to.equal(vaultAssetsBefore + totalInterest);
        });
    });
})
