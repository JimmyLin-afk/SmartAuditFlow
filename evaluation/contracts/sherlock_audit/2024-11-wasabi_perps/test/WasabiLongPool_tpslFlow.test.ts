import {
    time,
    loadFixture,
} from "@nomicfoundation/hardhat-toolbox-viem/network-helpers";
import { parseEther, zeroAddress } from "viem";
import { expect } from "chai";
import { AddCollateralRequest, ClosePositionRequest, FunctionCallData, OpenPositionRequest, OrderType, PayoutType } from "./utils/PerpStructUtils";
import { deployLongPoolMockEnvironment } from "./fixtures";
import { getBalance } from "./utils/StateUtils";
import { getApproveAndSwapFunctionCallData, getApproveAndSwapFunctionCallDataExact } from "./utils/SwapUtils";
import { signAddCollateralRequest, signClosePositionRequest, signOpenPositionRequest } from "./utils/SigningUtils";

describe("WasabiLongPool - TP/SL Flow Test", function () {
    describe("Take Profit", function () {
        it("Price increased to exact target", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, liquidator, publicClient, wasabiLongPool, user1, uPPG, mockSwap, feeReceiver, initialPrice, wethAddress, vault } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();

            // Take Profit Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.TP,
                traderSigner: user1,
                positionId: position.id,
                makerAmount: position.collateralAmount,
                takerAmount: (position.principal + position.downPayment) * 2n,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 2n]); // Price doubled

            // Close Position
            const { request, signature } = await createSignedClosePositionRequest({position});

            const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceBefore = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });

            const hash = await wasabiLongPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceAfter = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });

            // Checks
            const events = await wasabiLongPool.getEvents.PositionClosedWithOrder();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;
            const totalFeesPaid = closePositionEvent.feeAmount!;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal);
            expect(await uPPG.read.balanceOf([wasabiLongPool.address])).to.equal(0n, "Pool should not have any collateral left");

            expect(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!).to.equal(vaultBalanceAfter);

            const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.feeAmount!;
            expect(totalReturn).to.equal(position.downPayment * 5n, "on 2x price increase, total return should be 4x down payment");

            // Check trader has been paid
            expect(traderBalanceAfter - traderBalanceBefore).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        it("Price increased to exact target - authorized signer", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, liquidator, publicClient, wasabiLongPool, manager, user1, user2, uPPG, mockSwap, feeReceiver, initialPrice, wethAddress, vault } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();

            // Authorize user2 as signer for user1
            await manager.write.setAuthorizedSigner([user2.account.address, true], {account: user1.account});

            // Take Profit Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.TP,
                traderSigner: user2,
                positionId: position.id,
                makerAmount: position.collateralAmount,
                takerAmount: (position.principal + position.downPayment) * 2n,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 2n]); // Price doubled

            // Close Position
            const { request, signature } = await createSignedClosePositionRequest({position});

            const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceBefore = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });

            const hash = await wasabiLongPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceAfter = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });

            // Checks
            const events = await wasabiLongPool.getEvents.PositionClosedWithOrder();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;
            const totalFeesPaid = closePositionEvent.feeAmount!;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal);
            expect(await uPPG.read.balanceOf([wasabiLongPool.address])).to.equal(0n, "Pool should not have any collateral left");

            expect(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!).to.equal(vaultBalanceAfter);

            const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.feeAmount!;
            expect(totalReturn).to.equal(position.downPayment * 5n, "on 2x price increase, total return should be 4x down payment");

            // Check trader has been paid
            expect(traderBalanceAfter - traderBalanceBefore).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        it("Price increased above target", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, liquidator, publicClient, wasabiLongPool, user1, uPPG, mockSwap, feeReceiver, initialPrice, wethAddress, vault } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();

            // Take Profit Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.TP,
                traderSigner: user1,
                positionId: position.id,
                makerAmount: position.collateralAmount,
                takerAmount: (position.principal + position.downPayment) * 3n / 2n, // Expected 1.5x return
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 2n]); // Price doubled

            // Close Position
            const { request, signature } = await createSignedClosePositionRequest({position});

            const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceBefore = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });

            const hash = await wasabiLongPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceAfter = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });

            // Checks
            const events = await wasabiLongPool.getEvents.PositionClosedWithOrder();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;
            const totalFeesPaid = closePositionEvent.feeAmount!;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal);
            expect(await uPPG.read.balanceOf([wasabiLongPool.address])).to.equal(0n, "Pool should not have any collateral left");

            expect(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!).to.equal(vaultBalanceAfter);

            const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.feeAmount!;
            expect(totalReturn).to.equal(position.downPayment * 5n, "on 2x price increase, total return should be 4x down payment");

            // Check trader has been paid
            expect(traderBalanceAfter - traderBalanceBefore).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        it("Price increased to exact target - Partial Close", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, computeMaxInterest, liquidator, publicClient, wasabiLongPool, user1, uPPG, mockSwap, feeReceiver, initialPrice, wethAddress, vault } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();
            const closeAmountDenominator = 2n;

            // Take Profit Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.TP,
                traderSigner: user1,
                positionId: position.id,
                makerAmount: position.collateralAmount / closeAmountDenominator, // Partial close
                takerAmount: position.principal + position.downPayment,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 2n]); // Price doubled

            // Close Position
            const interest = await computeMaxInterest(position) / closeAmountDenominator;
            const { request, signature } = await createSignedClosePositionRequest({position, interest, amount: position.collateralAmount / closeAmountDenominator});

            const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceBefore = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });

            const hash = await wasabiLongPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceAfter = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });

            // Checks
            const events = await wasabiLongPool.getEvents.PositionDecreasedWithOrder();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;
            const totalFeesPaid = closePositionEvent.closeFee!;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal / closeAmountDenominator);
            expect(await uPPG.read.balanceOf([wasabiLongPool.address])).to.equal(position.collateralAmount / closeAmountDenominator, "Pool should have half of the collateral left");

            expect(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!).to.equal(vaultBalanceAfter);

            const downPaymentReduced = position.downPayment / closeAmountDenominator;
            const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.closeFee! - downPaymentReduced;
            expect(totalReturn).to.equal(downPaymentReduced * 4n, "on 2x price increase, total return should be 4x the adjusted down payment");

            // Check trader has been paid
            expect(traderBalanceAfter - traderBalanceBefore).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        it("Price increased to exact target - Partial Close after Increase", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, computeMaxInterest, liquidator, orderSigner, contractName, publicClient, wasabiLongPool, user1, uPPG, mockSwap, feeReceiver, initialPrice, priceDenominator, totalSize, totalAmountIn, wethAddress, vault } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();
            const closeAmountDenominator = 2n;

            // Take Profit Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.TP,
                traderSigner: user1,
                positionId: position.id,
                makerAmount: position.collateralAmount / closeAmountDenominator, // Partial close
                takerAmount: position.principal + position.downPayment,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            // Increase Position
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
            const increaseSignature = await signOpenPositionRequest(orderSigner, contractName, wasabiLongPool.address, openPositionRequest);

            await wasabiLongPool.write.openPosition([openPositionRequest, increaseSignature], { value: totalAmountIn, account: user1.account });

            const increaseEvents = await wasabiLongPool.getEvents.PositionIncreased();
            expect(increaseEvents).to.have.lengthOf(1);
            const increaseEvent = increaseEvents[0].args;
            position.collateralAmount += increaseEvent.collateralAdded!;
            position.downPayment += increaseEvent.downPaymentAdded!;
            position.principal += increaseEvent.principalAdded!;
            position.feesToBePaid += increaseEvent.feesAdded!;

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 2n]); // Price doubled

            // Close Position
            const interest = await computeMaxInterest(position) / closeAmountDenominator;
            const { request, signature } = await createSignedClosePositionRequest({position, interest, amount: position.collateralAmount / closeAmountDenominator});

            const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceBefore = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });

            const hash = await wasabiLongPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceAfter = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });

            // Checks
            const events = await wasabiLongPool.getEvents.PositionDecreasedWithOrder();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;
            const totalFeesPaid = closePositionEvent.closeFee!;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal / closeAmountDenominator);
            expect(await uPPG.read.balanceOf([wasabiLongPool.address])).to.equal(position.collateralAmount / closeAmountDenominator, "Pool should have half of the collateral left");

            expect(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!).to.equal(vaultBalanceAfter);

            const downPaymentReduced = position.downPayment / closeAmountDenominator;
            const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.closeFee! - downPaymentReduced;
            expect(totalReturn).to.equal(downPaymentReduced * 4n, "on 2x price increase, total return should be 4x the adjusted down payment");

            // Check trader has been paid
            expect(traderBalanceAfter - traderBalanceBefore).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        it("Price increased to exact target - Partial Close after Adding Collateral", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, computeMaxInterest, liquidator, orderSigner, contractName, publicClient, wasabiLongPool, user1, uPPG, mockSwap, feeReceiver, initialPrice, priceDenominator, totalAmountIn, wethAddress, vault } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();
            const closeAmountDenominator = 2n;
            
            await time.increase(86400n); // 1 day later

            // Take Profit Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.TP,
                traderSigner: user1,
                positionId: position.id,
                makerAmount: position.collateralAmount / closeAmountDenominator, // Partial close
                takerAmount: position.principal + position.downPayment,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            // Add Collateral
            let interest = await computeMaxInterest(position);
            const addCollateralRequest: AddCollateralRequest = {
                amount: totalAmountIn,
                interest,
                expiration: BigInt(await time.latest()) + 86400n,
                position
            }
            const addCollateralSignature = await signAddCollateralRequest(orderSigner, contractName, wasabiLongPool.address, addCollateralRequest);

            await wasabiLongPool.write.addCollateral([addCollateralRequest, addCollateralSignature], { value: totalAmountIn, account: user1.account });

            const addCollateralEvents = await wasabiLongPool.getEvents.CollateralAdded();
            expect(addCollateralEvents).to.have.lengthOf(1);
            const addCollateralEvent = addCollateralEvents[0].args;
            position.principal -= addCollateralEvent.principalReduced!;
            position.downPayment += addCollateralEvent.downPaymentAdded!;
            position.lastFundingTimestamp = BigInt(await time.latest());
            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 2n]); // Price doubled

            // Close Position
            interest = await computeMaxInterest(position) / closeAmountDenominator;
            const { request, signature } = await createSignedClosePositionRequest({position, interest, amount: position.collateralAmount / closeAmountDenominator});

            const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceBefore = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });

            const hash = await wasabiLongPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceAfter = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });

            // Checks
            const events = await wasabiLongPool.getEvents.PositionDecreasedWithOrder();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;
            const totalFeesPaid = closePositionEvent.closeFee!;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal / closeAmountDenominator);
            expect(await uPPG.read.balanceOf([wasabiLongPool.address])).to.equal(position.collateralAmount / closeAmountDenominator, "Pool should have half of the collateral left");

            expect(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!).to.equal(vaultBalanceAfter);

            const downPaymentReduced = position.downPayment / closeAmountDenominator;
            const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.closeFee! - downPaymentReduced;
            expect(totalReturn).to.equal(downPaymentReduced * (position.downPayment + position.principal) / position.downPayment, "on 2x price increase, total return should be the new leverage (2x) times the adjusted down payment");

            // Check trader has been paid
            expect(traderBalanceAfter - traderBalanceBefore).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        it("Price increased to exact target - Full Close after Decrease", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, computeMaxInterest, liquidator, publicClient, wasabiLongPool, user1, uPPG, mockSwap, feeReceiver, initialPrice, wethAddress, vault } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();
            const closeAmountDenominator = 2n;

            // Take Profit Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.TP,
                traderSigner: user1,
                positionId: position.id,
                makerAmount: position.collateralAmount,
                takerAmount: position.principal + position.downPayment,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            // Close Half of the Position
            const partialInterest = await computeMaxInterest(position) / closeAmountDenominator;
            const { request: partialRequest, signature: partialSignature } = await createSignedClosePositionRequest({ position, interest: partialInterest, amount: position.collateralAmount / closeAmountDenominator });

            await wasabiLongPool.write.closePosition([PayoutType.UNWRAPPED, partialRequest, partialSignature], { account: user1.account });

            const partialCloseEvents = await wasabiLongPool.getEvents.PositionDecreased();
            expect(partialCloseEvents).to.have.lengthOf(1);
            const partialCloseEvent = partialCloseEvents[0].args;
            position.collateralAmount -= partialCloseEvent.collateralReduced!;
            position.downPayment -= partialCloseEvent.downPaymentReduced!;
            position.principal -= partialCloseEvent.principalRepaid!;
            position.feesToBePaid -= partialCloseEvent.pastFees!;

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 2n]); // Price doubled

            // Close Remaining Position using Take Profit Order from before
            const interest = await computeMaxInterest(position);
            const { request, signature } = await createSignedClosePositionRequest({position, interest, amount: position.collateralAmount});

            const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceBefore = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });

            const hash = await wasabiLongPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceAfter = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });

            // Checks
            const events = await wasabiLongPool.getEvents.PositionClosedWithOrder();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;
            const totalFeesPaid = closePositionEvent.feeAmount!;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal);
            expect(await uPPG.read.balanceOf([wasabiLongPool.address])).to.equal(0n, "Pool should not have any collateral left");

            expect(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!).to.equal(vaultBalanceAfter);

            const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.feeAmount!;
            expect(totalReturn).to.equal(position.downPayment * 5n, "on 2x price increase, total return should be 4x down payment");

            // Check trader has been paid
            expect(traderBalanceAfter - traderBalanceBefore).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        describe("Validations", function () {

            it("PriceTargetNotReached", async function () {
                const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, wasabiLongPool, user1, uPPG, mockSwap, initialPrice, wethAddress, liquidator } = await loadFixture(deployLongPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                // Take Profit Order
                const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                    orderType: OrderType.TP,
                    traderSigner: user1,
                    positionId: position.id,
                    makerAmount: position.collateralAmount,
                    takerAmount: (position.principal + position.downPayment) * 3n, // Expected 3x return
                    expiration: await time.latest() + 172800,
                    executionFee: parseEther("0.05"),
                });

                await time.increase(86400n); // 1 day later
                await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 2n]); // Price doubled

                // Try to Close Position
                const { request, signature } = await createSignedClosePositionRequest({position});

                await expect(wasabiLongPool.write.closePosition(
                    [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
                )).to.be.rejectedWith("PriceTargetNotReached");
            });

            it("OrderExpired - Order", async function () {
                const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, wasabiLongPool, mockSwap, user1, uPPG, initialPrice, wethAddress, liquidator } = await loadFixture(deployLongPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                // Take Profit Order
                const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                    orderType: OrderType.TP,
                    traderSigner: user1,
                    positionId: position.id,
                    makerAmount: position.collateralAmount,
                    takerAmount: (position.principal + position.downPayment) * 2n,
                    expiration: await time.latest() + 86399,
                    executionFee: parseEther("0.05"),
                });

                await time.increase(86400); // 1 day later
                await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 2n]); // Price doubled

                // Try to Close Position
                const { request, signature } = await createSignedClosePositionRequest({position});

                await expect(wasabiLongPool.write.closePosition(
                    [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
                )).to.be.rejectedWith("OrderExpired");
            });

            it("OrderExpired - Request", async function () {
                const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, wasabiLongPool, mockSwap, user1, uPPG, initialPrice, wethAddress, liquidator } = await loadFixture(deployLongPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                // Take Profit Order
                const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                    orderType: OrderType.TP,
                    traderSigner: user1,
                    positionId: position.id,
                    makerAmount: position.collateralAmount,
                    takerAmount: (position.principal + position.downPayment) * 2n,
                    expiration: await time.latest() + 86400,
                    executionFee: parseEther("0.05"),
                });

                // Try to Close Position
                const { request, signature } = await createSignedClosePositionRequest({position});

                await time.increase(86400); // 1 day later - skip ahead after creating request
                await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 2n]); // Price doubled

                await expect(wasabiLongPool.write.closePosition(
                    [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
                )).to.be.rejectedWith("OrderExpired");
            });

            it("InvalidOrder - Position ID", async function () {
                const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, wasabiLongPool, user1, uPPG, mockSwap, initialPrice, wethAddress, liquidator } = await loadFixture(deployLongPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                // Take Profit Order - Invalid Position ID
                const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                    orderType: OrderType.TP,
                    traderSigner: user1,
                    positionId: position.id + 1n, // Invalid position id
                    makerAmount: position.collateralAmount,
                    takerAmount: (position.principal + position.downPayment) * 2n,
                    expiration: await time.latest() + 172800,
                    executionFee: parseEther("0.05"),
                });

                await time.increase(86400n); // 1 day later
                await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 2n]); // Price doubled

                // Try to Close Position
                const { request, signature } = await createSignedClosePositionRequest({position});

                await expect(wasabiLongPool.write.closePosition(
                    [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
                )).to.be.rejectedWith("InvalidOrder");
            });

            it("InvalidOrder - Order Type", async function () {
                const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, wasabiLongPool, user1, liquidator } = await loadFixture(deployLongPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                // Take Profit Order - Invalid Order Type
                const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                    orderType: OrderType.INVALID, // Invalid order type
                    traderSigner: user1,
                    positionId: position.id,
                    makerAmount: position.collateralAmount,
                    takerAmount: (position.principal + position.downPayment) * 2n,
                    expiration: await time.latest() + 172800,
                    executionFee: parseEther("0.05"),
                });

                // Try to Close Position
                const { request, signature } = await createSignedClosePositionRequest({position});

                await expect(wasabiLongPool.write.closePosition(
                    [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
                )).to.be.rejectedWith("InvalidOrder");
            });

            it("InvalidSignature - Order", async function () {
                const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, wasabiLongPool, user2, uPPG, mockSwap, initialPrice, wethAddress, liquidator } = await loadFixture(deployLongPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                // Take Profit Order
                const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                    orderType: OrderType.TP,
                    traderSigner: user2, // Wrong signer
                    positionId: position.id,
                    makerAmount: position.collateralAmount,
                    takerAmount: (position.principal + position.downPayment) * 2n, // Expected 2x return
                    expiration: await time.latest() + 172800,
                    executionFee: parseEther("0.05"),
                });

                await time.increase(86400n); // 1 day later
                await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 2n]); // Price doubled

                // Try to Close Position
                const { request, signature } = await createSignedClosePositionRequest({position});

                await expect(wasabiLongPool.write.closePosition(
                    [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
                )).to.be.rejectedWith("InvalidSignature");
            });

            it("InvalidSignature - Request", async function () {
                const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, wasabiLongPool, user1, uPPG, mockSwap, initialPrice, wethAddress, liquidator } = await loadFixture(deployLongPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                // Take Profit Order
                const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                    orderType: OrderType.TP,
                    traderSigner: user1,
                    positionId: position.id,
                    makerAmount: position.collateralAmount,
                    takerAmount: (position.principal + position.downPayment) * 2n, // Expected 2x return
                    expiration: await time.latest() + 172800,
                    executionFee: parseEther("0.05"),
                });

                await time.increase(86400n); // 1 day later
                await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 2n]); // Price doubled

                // Try to Close Position
                const { request } = await createSignedClosePositionRequest({position});

                await expect(wasabiLongPool.write.closePosition(
                    [PayoutType.UNWRAPPED, request, orderSignature, order, orderSignature], {account: liquidator.account} // Wrong signature
                )).to.be.rejectedWith("InvalidSignature");
            });
        });
    });

    describe("Stop Loss", function () {
        it("Price decreased to exact target", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, publicClient, wasabiLongPool, user1, uPPG, mockSwap, feeReceiver, initialPrice, wethAddress, liquidator, vault } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();

            // Stop Loss Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.SL,
                traderSigner: user1,
                positionId: position.id,
                makerAmount: position.collateralAmount,
                takerAmount: (position.principal + position.downPayment) * 8n / 10n,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 8n / 10n]); // Price fell 20%

            // Close Position
            const { request, signature } = await createSignedClosePositionRequest({position});

            const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceBefore = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });

            const hash = await wasabiLongPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceAfter = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });

            // Checks
            const events = await wasabiLongPool.getEvents.PositionClosedWithOrder();
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
            expect(traderBalanceAfter - traderBalanceBefore).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        it("Price decreased to exact target - authorized signer", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, publicClient, wasabiLongPool, user1, user2, manager, uPPG, mockSwap, feeReceiver, initialPrice, wethAddress, liquidator, vault } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();

            // Authorize user2 as signer for user1
            await manager.write.setAuthorizedSigner([user2.account.address, true], {account: user1.account});

            // Stop Loss Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.SL,
                traderSigner: user2,
                positionId: position.id,
                makerAmount: position.collateralAmount,
                takerAmount: (position.principal + position.downPayment) * 8n / 10n,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 8n / 10n]); // Price fell 20%

            // Close Position
            const { request, signature } = await createSignedClosePositionRequest({position});

            const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceBefore = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });

            const hash = await wasabiLongPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceAfter = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });

            // Checks
            const events = await wasabiLongPool.getEvents.PositionClosedWithOrder();
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
            expect(traderBalanceAfter - traderBalanceBefore).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        it("Price decreased below target", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, publicClient, wasabiLongPool, user1, uPPG, mockSwap, feeReceiver, initialPrice, wethAddress, liquidator, vault } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();

            // Stop Loss Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.SL,
                traderSigner: user1,
                positionId: position.id,
                makerAmount: position.collateralAmount,
                takerAmount: (position.principal + position.downPayment) * 9n / 10n,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 8n / 10n]); // Price fell 20%

            // Close Position
            const { request, signature } = await createSignedClosePositionRequest({position});

            const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceBefore = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });

            const hash = await wasabiLongPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceAfter = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });

            // Checks
            const events = await wasabiLongPool.getEvents.PositionClosedWithOrder();
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
            expect(traderBalanceAfter - traderBalanceBefore).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        it("Price decreased to exact target - Partial Close", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, computeMaxInterest, publicClient, wasabiLongPool, user1, uPPG, mockSwap, feeReceiver, initialPrice, wethAddress, liquidator, vault } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();
            const closeAmountDenominator = 2n;

            // Stop Loss Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.SL,
                traderSigner: user1,
                positionId: position.id,
                makerAmount: position.collateralAmount / closeAmountDenominator, // Partial close
                takerAmount: (position.principal + position.downPayment) * 8n / 10n / closeAmountDenominator,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 8n / 10n]); // Price fell 20%

            // Close Position
            const interest = await computeMaxInterest(position) / closeAmountDenominator;
            const { request, signature } = await createSignedClosePositionRequest({position, interest, amount: position.collateralAmount / closeAmountDenominator});

            const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceBefore = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });

            const hash = await wasabiLongPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceAfter = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });

            // Checks
            const events = await wasabiLongPool.getEvents.PositionDecreasedWithOrder();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;
            const totalFeesPaid = closePositionEvent.closeFee!;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal / closeAmountDenominator);
            expect(await uPPG.read.balanceOf([wasabiLongPool.address])).to.equal(position.collateralAmount / closeAmountDenominator, "Pool should have half of the collateral left");

            expect(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!).to.equal(vaultBalanceAfter);

            const downPaymentReduced = position.downPayment / closeAmountDenominator;
            const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.closeFee! - downPaymentReduced;
            expect(totalReturn).to.equal(downPaymentReduced / -5n * 4n, "on 20% price decrease, total return should be -20% * leverage (4) * adjusted down payment");

            // Check trader has been paid
            expect(traderBalanceAfter - traderBalanceBefore).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        it("Price decreased to exact target - Partial Close after Increase", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, computeMaxInterest, publicClient, wasabiLongPool, contractName, user1, uPPG, mockSwap, feeReceiver, initialPrice, priceDenominator, totalSize, totalAmountIn, wethAddress, liquidator, orderSigner, vault } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();
            const closeAmountDenominator = 2n;

            // Stop Loss Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.SL,
                traderSigner: user1,
                positionId: position.id,
                makerAmount: position.collateralAmount / closeAmountDenominator, // Partial close
                takerAmount: (position.principal + position.downPayment) * 8n / 10n / closeAmountDenominator,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            // Increase Position
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
            const increaseSignature = await signOpenPositionRequest(orderSigner, contractName, wasabiLongPool.address, openPositionRequest);

            await wasabiLongPool.write.openPosition([openPositionRequest, increaseSignature], { value: totalAmountIn, account: user1.account });

            const increaseEvents = await wasabiLongPool.getEvents.PositionIncreased();
            expect(increaseEvents).to.have.lengthOf(1);
            const increaseEvent = increaseEvents[0].args;
            position.collateralAmount += increaseEvent.collateralAdded!;
            position.downPayment += increaseEvent.downPaymentAdded!;
            position.principal += increaseEvent.principalAdded!;
            position.feesToBePaid += increaseEvent.feesAdded!;

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 8n / 10n]); // Price fell 20%

            // Close Position
            const interest = await computeMaxInterest(position) / closeAmountDenominator;
            const { request, signature } = await createSignedClosePositionRequest({position, interest, amount: position.collateralAmount / closeAmountDenominator});

            const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceBefore = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });

            const hash = await wasabiLongPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceAfter = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });

            // Checks
            const events = await wasabiLongPool.getEvents.PositionDecreasedWithOrder();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;
            const totalFeesPaid = closePositionEvent.closeFee!;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal / closeAmountDenominator);
            expect(await uPPG.read.balanceOf([wasabiLongPool.address])).to.equal(position.collateralAmount / closeAmountDenominator, "Pool should have half of the collateral left");

            expect(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!).to.equal(vaultBalanceAfter);

            const downPaymentReduced = position.downPayment / closeAmountDenominator;
            const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.closeFee! - downPaymentReduced;
            expect(totalReturn).to.equal(downPaymentReduced / -5n * 4n, "on 20% price decrease, total return should be -20% * leverage (4) * adjusted down payment");

            // Check trader has been paid
            expect(traderBalanceAfter - traderBalanceBefore).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        it("Price decreased to exact target - Partial Close after Adding Collateral", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, computeMaxInterest, publicClient, wasabiLongPool, contractName, user1, uPPG, mockSwap, feeReceiver, initialPrice, priceDenominator, downPayment, wethAddress, liquidator, orderSigner, vault } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();
            const closeAmountDenominator = 2n;
            
            await time.increase(86400n); // 1 day later

            // Stop Loss Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.SL,
                traderSigner: user1,
                positionId: position.id,
                makerAmount: position.collateralAmount / closeAmountDenominator, // Partial close
                takerAmount: (position.principal + position.downPayment) * 8n / 10n / closeAmountDenominator,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            // Add Collateral
            let interest = await computeMaxInterest(position);
            const addCollateralRequest: AddCollateralRequest = {
                amount: downPayment,
                interest,
                expiration: BigInt(await time.latest()) + 86400n,
                position
            }
            const addCollateralSignature = await signAddCollateralRequest(orderSigner, contractName, wasabiLongPool.address, addCollateralRequest);

            await wasabiLongPool.write.addCollateral([addCollateralRequest, addCollateralSignature], { value: position.downPayment, account: user1.account });

            const addCollateralEvents = await wasabiLongPool.getEvents.CollateralAdded();
            expect(addCollateralEvents).to.have.lengthOf(1);
            const addCollateralEvent = addCollateralEvents[0].args;
            position.principal -= addCollateralEvent.principalReduced!;
            position.downPayment += addCollateralEvent.downPaymentAdded!;
            position.lastFundingTimestamp = BigInt(await time.latest());
            const newLeverage = (position.downPayment + position.principal) * 100n / position.downPayment;

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 8n / 10n]); // Price fell 20%

            // Close Position
            interest = await computeMaxInterest(position) / closeAmountDenominator;
            const { request, signature } = await createSignedClosePositionRequest({position, interest, amount: position.collateralAmount / closeAmountDenominator});

            const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceBefore = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });

            const hash = await wasabiLongPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceAfter = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });

            // Checks
            const events = await wasabiLongPool.getEvents.PositionDecreasedWithOrder();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;
            const totalFeesPaid = closePositionEvent.closeFee!;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal / closeAmountDenominator);
            expect(await uPPG.read.balanceOf([wasabiLongPool.address])).to.equal(position.collateralAmount / closeAmountDenominator, "Pool should have half of the collateral left");

            expect(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!).to.equal(vaultBalanceAfter);

            const downPaymentReduced = position.downPayment / closeAmountDenominator;
            const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.closeFee! - downPaymentReduced;
            expect(totalReturn).to.equal(
                downPaymentReduced * (position.downPayment + position.principal) / position.downPayment / -5n, 
                "on 20% price decrease, total return should be -20% * new leverage (2x) * adjusted down payment"
            );

            // Check trader has been paid
            expect(traderBalanceAfter - traderBalanceBefore).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        it("Price decreased to exact target - Full Close after Decrease", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, computeMaxInterest, liquidator, publicClient, wasabiLongPool, user1, uPPG, mockSwap, feeReceiver, initialPrice, wethAddress, vault } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();
            const closeAmountDenominator = 2n;

            // Stop Loss Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.SL,
                traderSigner: user1,
                positionId: position.id,
                makerAmount: position.collateralAmount,
                takerAmount: (position.principal + position.downPayment) * 8n / 10n,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            // Close Half of the Position
            const partialInterest = await computeMaxInterest(position) / closeAmountDenominator;
            const { request: partialRequest, signature: partialSignature } = await createSignedClosePositionRequest({ position, interest: partialInterest, amount: position.collateralAmount / closeAmountDenominator });

            await wasabiLongPool.write.closePosition([PayoutType.UNWRAPPED, partialRequest, partialSignature], { account: user1.account });

            const partialCloseEvents = await wasabiLongPool.getEvents.PositionDecreased();
            expect(partialCloseEvents).to.have.lengthOf(1);
            const partialCloseEvent = partialCloseEvents[0].args;
            position.collateralAmount -= partialCloseEvent.collateralReduced!;
            position.downPayment -= partialCloseEvent.downPaymentReduced!;
            position.principal -= partialCloseEvent.principalRepaid!;
            position.feesToBePaid -= partialCloseEvent.pastFees!;

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 8n / 10n]); // Price fell 20%

            // Close Remaining Position using Stop Loss Order from before
            const interest = await computeMaxInterest(position);
            const { request, signature } = await createSignedClosePositionRequest({position, interest, amount: position.collateralAmount});

            const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceBefore = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });

            const hash = await wasabiLongPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceAfter = await getBalance(publicClient, wethAddress, vault.address);
            const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });

            // Checks
            const events = await wasabiLongPool.getEvents.PositionClosedWithOrder();
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
            expect(traderBalanceAfter - traderBalanceBefore).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        describe("Validations", function () {
            it("PriceTargetNotReached", async function () {
                const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, wasabiLongPool, user1, uPPG, mockSwap, initialPrice, wethAddress, liquidator } = await loadFixture(deployLongPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                // Stop Loss Order
                const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                    orderType: OrderType.SL,
                    traderSigner: user1,
                    positionId: position.id,
                    makerAmount: position.collateralAmount,
                    takerAmount: (position.principal + position.downPayment) * 8n / 10n,
                    expiration: await time.latest() + 172800,
                    executionFee: parseEther("0.05"),
                });

                await time.increase(86400n); // 1 day later
                await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 9n / 10n]); // Price fell 10%

                // Close Position
                const { request, signature } = await createSignedClosePositionRequest({position});

                await expect(wasabiLongPool.write.closePosition(
                    [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
                )).to.be.rejectedWith("PriceTargetNotReached");
            });

            it("InsufficientPrincipalRepaid - Bad debt from bad swap function call", async function () {
                const { sendDefaultOpenPositionRequest, createSignedClosePositionOrder, orderSigner, contractName, wasabiLongPool, user1, uPPG, mockSwap, initialPrice, wethAddress, liquidator } = await loadFixture(deployLongPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                // Stop Loss Order
                const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                    orderType: OrderType.SL,
                    traderSigner: user1,
                    positionId: position.id,
                    makerAmount: position.collateralAmount,
                    takerAmount: (position.principal + position.downPayment) * 8n / 10n,
                    expiration: await time.latest() + 172800,
                    executionFee: parseEther("0.05"),
                });

                await time.increase(86400n); // 1 day later
                await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPrice * 8n / 10n]); // Price fell 20%

                // Craft a ClosePositionRequest with a malicious swap function call using MockSwap.swapExact
                const request: ClosePositionRequest = {
                    expiration: BigInt(await time.latest() + 172800),
                    interest: 0n,
                    amount: 0n,
                    position,
                    functionCallDataList: getApproveAndSwapFunctionCallDataExact(mockSwap.address, position.collateralCurrency, position.currency, position.collateralAmount, 1n), // bad amountOut
                    referrer: zeroAddress
                };
                const signature = await signClosePositionRequest(orderSigner, contractName, wasabiLongPool.address, request);

                await expect(wasabiLongPool.write.closePosition(
                    [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
                )).to.be.rejectedWith("InsufficientPrincipalRepaid");
            });
        });
    });
});