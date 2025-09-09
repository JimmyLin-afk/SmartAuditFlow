import {
    time,
    loadFixture,
} from "@nomicfoundation/hardhat-toolbox-viem/network-helpers";
import {encodeFunctionData, zeroAddress, parseEther} from "viem";
import { expect } from "chai";
import { Position, OrderType, PayoutType, getValueWithoutFee, ClosePositionRequest, FunctionCallData, OpenPositionRequest, AddCollateralRequest } from "./utils/PerpStructUtils";
import { getApproveAndSwapFunctionCallData, getApproveAndSwapFunctionCallDataExact } from "./utils/SwapUtils";
import { deployShortPoolMockEnvironment } from "./fixtures";
import { getBalance, takeBalanceSnapshot } from "./utils/StateUtils";
import { signAddCollateralRequest, signClosePositionRequest, signOpenPositionRequest } from "./utils/SigningUtils";

describe("WasabiShortPool - TP/SL Flow Test", function () {
    describe("Take Profit", function () {
        it("Price decreased to exact target", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, computeMaxInterest, mockSwap, publicClient, wasabiShortPool, user1, uPPG, feeReceiver, wethAddress, initialPPGPrice, liquidator, vault } = await loadFixture(deployShortPoolMockEnvironment);

            // Open Position
            const vaultBalanceInitial = await getBalance(publicClient, uPPG.address, vault.address);
            const {position} = await sendDefaultOpenPositionRequest();

            // Take Profit Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.TP,
                traderSigner: user1,
                positionId: position.id,
                makerAmount: position.collateralAmount / 2n,
                takerAmount: position.principal + position.downPayment,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice / 2n]); // Price halved

            // Close Position
            const maxInterest = await computeMaxInterest(position);
            const { request, signature } = await createSignedClosePositionRequest({ position, interest: maxInterest });
            
            const vaultBalanceBefore = await getBalance(publicClient, uPPG.address, vault.address);
            const userBalanceBefore = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceBefore = await publicClient.getBalance({ address: feeReceiver });
        
            const hash = await wasabiShortPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const vaultBalanceAfter = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
            const userBalanceAfter = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceAfter = await publicClient.getBalance({ address: feeReceiver });

            // Checks
            const events = await wasabiShortPool.getEvents.PositionClosedWithOrder();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal);
            expect(closePositionEvent.interestPaid!).to.equal(maxInterest, "If given interest value is 0, should use max interest");

            // Interest is paid in uPPG, so the principal should be equal before and after the trade
            expect(vaultBalanceAfter).eq(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!, "Invalid repay amount");
            expect(vaultBalanceInitial + closePositionEvent.interestPaid!).eq(vaultBalanceAfter, "Original amount + interest wasn't repayed");

            expect(balancesAfter.get(wasabiShortPool.address)).to.equal(0, "Pool should not have any collateral left");

            const interestPaidInEth = closePositionEvent.interestPaid! / 2n;
            const totalReturn = closePositionEvent.payout! + closePositionEvent.feeAmount! - position.downPayment + interestPaidInEth;
            expect(totalReturn).to.equal(position.downPayment * 5n / 2n, "On 50% price decrease w/ 5x leverage, total return should be 50% * 5 * down payment");

            // Check trader has been paid
            expect(userBalanceAfter - userBalanceBefore).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            const totalFeesPaid = closePositionEvent.feeAmount!;
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        it("Price decreased to below target", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, computeMaxInterest, mockSwap, publicClient, wasabiShortPool, user1, uPPG, feeReceiver, wethAddress, initialPPGPrice, liquidator, vault } = await loadFixture(deployShortPoolMockEnvironment);

            // Open Position
            const vaultBalanceInitial = await getBalance(publicClient, uPPG.address, vault.address);
            const {position} = await sendDefaultOpenPositionRequest();

            // Take Profit Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.TP,
                traderSigner: user1,
                positionId: position.id,
                makerAmount: position.collateralAmount / 2n,
                takerAmount: position.principal + position.downPayment,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice / 4n]); // Price quartered

            // Close Position
            const maxInterest = await computeMaxInterest(position);
            const { request, signature } = await createSignedClosePositionRequest({ position, interest: maxInterest });
            
            const vaultBalanceBefore = await getBalance(publicClient, uPPG.address, vault.address);
            const userBalanceBefore = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceBefore = await publicClient.getBalance({ address: feeReceiver });
        
            const hash = await wasabiShortPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const vaultBalanceAfter = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
            const userBalanceAfter = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceAfter = await publicClient.getBalance({ address: feeReceiver });

            // Checks
            const events = await wasabiShortPool.getEvents.PositionClosedWithOrder();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal);
            expect(closePositionEvent.interestPaid!).to.equal(maxInterest, "If given interest value is 0, should use max interest");

            // Interest is paid in uPPG, so the principal should be equal before and after the trade
            expect(vaultBalanceAfter).eq(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!, "Invalid repay amount");
            expect(vaultBalanceInitial + closePositionEvent.interestPaid!).eq(vaultBalanceAfter, "Original amount + interest wasn't repayed");

            expect(balancesAfter.get(wasabiShortPool.address)).to.equal(0, "Pool should not have any collateral left");

            const interestPaidInEth = closePositionEvent.interestPaid! / 4n;
            const totalReturn = closePositionEvent.payout! + closePositionEvent.feeAmount! - position.downPayment + interestPaidInEth;
            expect(totalReturn).to.equal(position.downPayment * 15n / 4n, "On 75% price decrease w/ 5x leverage, total return should be 75% * 5 * down payment");

            // Check trader has been paid
            expect(userBalanceAfter - userBalanceBefore).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            const totalFeesPaid = closePositionEvent.feeAmount!;
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        it("Price decreased to exact target - Partial close", async function () {
            const { sendDefaultOpenPositionRequest, createClosePositionRequest, createSignedClosePositionOrder, computeMaxInterest, mockSwap, publicClient, wasabiShortPool, user1, uPPG, feeReceiver, wethAddress, initialPPGPrice, liquidator, orderSigner, contractName, vault } = await loadFixture(deployShortPoolMockEnvironment);

            // Open Position
            const vaultBalanceInitial = await getBalance(publicClient, uPPG.address, vault.address);
            const {position} = await sendDefaultOpenPositionRequest();
            const closeAmountDenominator = 2n;

            // Take Profit Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.TP,
                traderSigner: user1,
                positionId: position.id,
                makerAmount: position.collateralAmount / 2n / closeAmountDenominator,
                takerAmount: (position.principal + position.downPayment) / closeAmountDenominator,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice / 2n]); // Price halved

            // Close Position
            const interest = await computeMaxInterest(position) / closeAmountDenominator;
            const amount = position.principal / closeAmountDenominator;
            const request = await createClosePositionRequest({ position, interest, amount }); 
            const signature = await signClosePositionRequest(orderSigner, contractName, wasabiShortPool.address, request);
            
            const vaultBalanceBefore = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
            const userBalanceBefore = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceBefore = await publicClient.getBalance({ address: feeReceiver });
        
            const hash = await wasabiShortPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const vaultBalanceAfter = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
            const userBalanceAfter = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceAfter = await publicClient.getBalance({ address: feeReceiver });

            // Checks
            const events = await wasabiShortPool.getEvents.PositionDecreasedWithOrder();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal / closeAmountDenominator);
            expect(closePositionEvent.interestPaid!).to.equal(interest, "Prorated interest should be paid");

            // Interest is paid in uPPG, so the principal should be equal before and after the trade
            expect(vaultBalanceAfter).eq(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!, "Invalid repay amount");
            expect(vaultBalanceInitial + closePositionEvent.interestPaid! - position.principal / closeAmountDenominator).eq(vaultBalanceAfter, "Half of original amount + interest wasn't repayed");
            expect(balancesAfter.get(wasabiShortPool.address)).to.equal(balancesBefore.get(wasabiShortPool.address) / closeAmountDenominator, "Pool should have half of the collateral left");

            const downPaymentReduced = position.downPayment / closeAmountDenominator;
            const interestPaidInEth = closePositionEvent.interestPaid! / 2n;
            const totalReturn = closePositionEvent.payout! + interestPaidInEth + closePositionEvent.closeFee! - downPaymentReduced;
            expect(totalReturn).to.equal(downPaymentReduced * 5n / 2n, "On 50% price decrease w/ 5x leverage, total return should be 50% * 5 * adjusted down payment");

            // Check trader has been paid
            expect(userBalanceAfter - userBalanceBefore).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            const totalFeesPaid = closePositionEvent.closeFee!;
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        it("Price decreased to exact target - Partial close after increase", async function () {
            const { sendDefaultOpenPositionRequest, createClosePositionRequest, createSignedClosePositionOrder, computeMaxInterest, mockSwap, publicClient, wasabiShortPool, user1, uPPG, feeReceiver, wethAddress, initialPPGPrice, priceDenominator, principal, totalAmountIn, liquidator, orderSigner, contractName, vault } = await loadFixture(deployShortPoolMockEnvironment);

            // Open Position
            const vaultBalanceInitial = await getBalance(publicClient, uPPG.address, vault.address);
            const {position} = await sendDefaultOpenPositionRequest();
            const closeAmountDenominator = 2n;

            // Take Profit Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.TP,
                traderSigner: user1,
                positionId: position.id,
                makerAmount: position.collateralAmount / 2n / closeAmountDenominator,
                takerAmount: (position.principal + position.downPayment) / closeAmountDenominator,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            // Increase Position
            const functionCallDataList: FunctionCallData[] =
                getApproveAndSwapFunctionCallData(mockSwap.address, uPPG.address, wethAddress, principal);
            const openPositionRequest: OpenPositionRequest = {
                id: position.id,
                currency: position.currency,
                targetCurrency: position.collateralCurrency,
                downPayment: position.downPayment,
                principal: position.principal,
                minTargetAmount: principal * initialPPGPrice / priceDenominator,
                expiration: BigInt(await time.latest()) + 86400n,
                fee: position.feesToBePaid,
                functionCallDataList,
                existingPosition: position,
                referrer: zeroAddress
            };
            const increaseSignature = await signOpenPositionRequest(orderSigner, contractName, wasabiShortPool.address, openPositionRequest);

            await wasabiShortPool.write.openPosition([openPositionRequest, increaseSignature], { value: totalAmountIn, account: user1.account });

            const increaseEvents = await wasabiShortPool.getEvents.PositionIncreased();
            expect(increaseEvents).to.have.lengthOf(1);
            const increaseEvent = increaseEvents[0].args;
            position.collateralAmount += increaseEvent.collateralAdded!;
            position.downPayment += increaseEvent.downPaymentAdded!;
            position.principal += increaseEvent.principalAdded!;
            position.feesToBePaid += increaseEvent.feesAdded!;

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice / 2n]); // Price halved

            // Close Position
            const interest = await computeMaxInterest(position) / closeAmountDenominator;
            const amount = position.principal / closeAmountDenominator;
            const request = await createClosePositionRequest({ position, interest, amount }); 
            const signature = await signClosePositionRequest(orderSigner, contractName, wasabiShortPool.address, request);
            
            const vaultBalanceBefore = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
        
            const hash = await wasabiShortPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const vaultBalanceAfter = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);

            // Checks
            const events = await wasabiShortPool.getEvents.PositionDecreasedWithOrder();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal / closeAmountDenominator);
            expect(closePositionEvent.interestPaid!).to.equal(interest, "Prorated interest should be paid");

            // Interest is paid in uPPG, so the principal should be equal before and after the trade
            expect(vaultBalanceAfter).eq(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!, "Invalid repay amount");
            expect(vaultBalanceInitial + closePositionEvent.interestPaid! - position.principal / closeAmountDenominator).eq(vaultBalanceAfter, "Half of original amount + interest wasn't repayed");
            expect(balancesAfter.get(wasabiShortPool.address)).to.equal(balancesBefore.get(wasabiShortPool.address) / closeAmountDenominator, "Pool should have half of the collateral left");

            const downPaymentReduced = position.downPayment / closeAmountDenominator;
            const interestPaidInEth = closePositionEvent.interestPaid! / 2n;
            const totalReturn = closePositionEvent.payout! + interestPaidInEth + closePositionEvent.closeFee! - downPaymentReduced;
            expect(totalReturn).to.equal(downPaymentReduced * 5n / 2n, "On 50% price decrease w/ 5x leverage, total return should be 50% * 5 * adjusted down payment");
        });

        it("Price decreased to exact target - Partial close after adding collateral", async function () {
            const { sendDefaultOpenPositionRequest, createClosePositionRequest, createSignedClosePositionOrder, computeMaxInterest, mockSwap, publicClient, wasabiShortPool, user1, uPPG, feeReceiver, wethAddress, initialPPGPrice, priceDenominator, downPayment, liquidator, orderSigner, contractName, vault } = await loadFixture(deployShortPoolMockEnvironment);

            // Open Position
            const vaultBalanceInitial = await getBalance(publicClient, uPPG.address, vault.address);
            const {position} = await sendDefaultOpenPositionRequest();
            const closeAmountDenominator = 2n;

            // Take Profit Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.TP,
                traderSigner: user1,
                positionId: position.id,
                makerAmount: position.collateralAmount / 2n / closeAmountDenominator,
                takerAmount: (position.principal + position.downPayment) / closeAmountDenominator,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            // Add Collateral
            const addCollateralRequest: AddCollateralRequest = {
                amount: downPayment,
                interest: 0n,
                expiration: BigInt(await time.latest()) + 86400n,
                position
            }
            const addCollateralSignature = await signAddCollateralRequest(orderSigner, contractName, wasabiShortPool.address, addCollateralRequest);

            await wasabiShortPool.write.addCollateral([addCollateralRequest, addCollateralSignature], { value: downPayment, account: user1.account });

            const addCollateralEvents = await wasabiShortPool.getEvents.CollateralAdded();
            expect(addCollateralEvents).to.have.lengthOf(1);
            const addCollateralEvent = addCollateralEvents[0].args;
            position.collateralAmount += addCollateralEvent.collateralAdded!;
            position.downPayment += addCollateralEvent.downPaymentAdded!;
            const newLeverage = position.principal * 100n / (position.downPayment * initialPPGPrice / priceDenominator);

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice / 2n]); // Price halved

            // Close Position
            const interest = await computeMaxInterest(position) / closeAmountDenominator;
            const amount = position.principal / closeAmountDenominator;
            const request = await createClosePositionRequest({ position, interest, amount }); 
            const signature = await signClosePositionRequest(orderSigner, contractName, wasabiShortPool.address, request);
            
            const vaultBalanceBefore = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
        
            const hash = await wasabiShortPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const vaultBalanceAfter = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);

            // Checks
            const events = await wasabiShortPool.getEvents.PositionDecreasedWithOrder();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal / closeAmountDenominator);
            expect(closePositionEvent.interestPaid!).to.equal(interest, "Prorated interest should be paid");

            // Interest is paid in uPPG, so the principal should be equal before and after the trade
            expect(vaultBalanceAfter).eq(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!, "Invalid repay amount");
            expect(vaultBalanceInitial + closePositionEvent.interestPaid! - position.principal / closeAmountDenominator).eq(vaultBalanceAfter, "Half of original amount + interest wasn't repayed");
            expect(balancesAfter.get(wasabiShortPool.address)).to.equal(balancesBefore.get(wasabiShortPool.address) / closeAmountDenominator, "Pool should have half of the collateral left");

            const downPaymentReduced = position.downPayment / closeAmountDenominator;
            const interestPaidInEth = closePositionEvent.interestPaid! / 2n;
            const totalReturn = closePositionEvent.payout! + interestPaidInEth + closePositionEvent.closeFee! - downPaymentReduced;
            expect(totalReturn).to.equal(downPaymentReduced / 2n * newLeverage / 100n, "On 50% price decrease, total return should be 50% * new leverage * adjusted down payment");
        });

        it("Price decreased to exact target - Full close after decrease", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, computeMaxInterest, mockSwap, publicClient, wasabiShortPool, user1, uPPG, feeReceiver, wethAddress, initialPPGPrice, liquidator, vault } = await loadFixture(deployShortPoolMockEnvironment);

            // Open Position
            const vaultBalanceInitial = await getBalance(publicClient, uPPG.address, vault.address);
            const {position} = await sendDefaultOpenPositionRequest();

            // Take Profit Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.TP,
                traderSigner: user1,
                positionId: position.id,
                makerAmount: position.collateralAmount / 2n,
                takerAmount: position.principal + position.downPayment,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            // Close Half of the Position
            const closeAmountDenominator = 2n;
            const interest = await computeMaxInterest(position) / closeAmountDenominator;
            const amount = position.principal / closeAmountDenominator;
            const {request: partialRequest, signature: partialSignature} = await createSignedClosePositionRequest({ position, interest, amount }); 

            await wasabiShortPool.write.closePosition([PayoutType.UNWRAPPED, partialRequest, partialSignature], { account: user1.account });

            const partialCloseEvents = await wasabiShortPool.getEvents.PositionDecreased();
            expect(partialCloseEvents).to.have.lengthOf(1);
            const partialCloseEvent = partialCloseEvents[0].args;
            position.collateralAmount -= partialCloseEvent.collateralReduced!;
            position.downPayment -= partialCloseEvent.downPaymentReduced!;
            position.principal -= partialCloseEvent.principalRepaid!;
            position.feesToBePaid -= partialCloseEvent.pastFees!;

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice / 2n]); // Price halved

            // Close Position
            const maxInterest = await computeMaxInterest(position);
            const { request, signature } = await createSignedClosePositionRequest({ position, interest: maxInterest });
            
            const vaultBalanceBefore = await getBalance(publicClient, uPPG.address, vault.address);
            const userBalanceBefore = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceBefore = await publicClient.getBalance({ address: feeReceiver });
        
            const hash = await wasabiShortPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const vaultBalanceAfter = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
            const userBalanceAfter = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceAfter = await publicClient.getBalance({ address: feeReceiver });

            // Checks
            const events = await wasabiShortPool.getEvents.PositionClosedWithOrder();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal);
            expect(closePositionEvent.interestPaid!).to.equal(maxInterest, "If given interest value is 0, should use max interest");

            // Interest is paid in uPPG, so the principal should be equal before and after the trade
            expect(vaultBalanceAfter).eq(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!, "Invalid repay amount");
            expect(vaultBalanceInitial + closePositionEvent.interestPaid! + partialCloseEvent.interestPaid!).eq(vaultBalanceAfter, "Original amount + interest wasn't repayed");

            expect(balancesAfter.get(wasabiShortPool.address)).to.equal(0, "Pool should not have any collateral left");

            const interestPaidInEth = closePositionEvent.interestPaid! / 2n;
            const totalReturn = closePositionEvent.payout! + closePositionEvent.feeAmount! - position.downPayment + interestPaidInEth;
            expect(totalReturn).to.equal(position.downPayment * 5n / 2n, "On 50% price decrease w/ 5x leverage, total return should be 50% * 5 * down payment");

            // Check trader has been paid
            expect(userBalanceAfter - userBalanceBefore).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            const totalFeesPaid = closePositionEvent.feeAmount!;
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        describe("Validations", function () {
            it("PriceTargetNotReached", async function () {
                const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, mockSwap, publicClient, wasabiShortPool, user1, uPPG, wethAddress, initialPPGPrice, liquidator } = await loadFixture(deployShortPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                // Take Profit Order
                const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                    orderType: OrderType.TP,
                    traderSigner: user1,
                    positionId: position.id,
                    makerAmount: position.collateralAmount / 4n,
                    takerAmount: position.principal + position.downPayment,
                    expiration: await time.latest() + 172800,
                    executionFee: parseEther("0.05"),
                });

                await time.increase(86400n); // 1 day later
                await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice / 2n]); // Price halved

                // Try to Close Position
                const { request, signature } = await createSignedClosePositionRequest({position});

                await expect(wasabiShortPool.write.closePosition(
                    [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
                )).to.be.rejectedWith("PriceTargetNotReached");
            });

            it("OrderExpired - Order", async function () {
                const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, mockSwap, publicClient, wasabiShortPool, user1, uPPG, wethAddress, initialPPGPrice, liquidator } = await loadFixture(deployShortPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                // Take Profit Order
                const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                    orderType: OrderType.TP,
                    traderSigner: user1,
                    positionId: position.id,
                    makerAmount: position.collateralAmount / 2n,
                    takerAmount: position.principal + position.downPayment,
                    expiration: await time.latest() + 86399,
                    executionFee: parseEther("0.05"),
                });

                await time.increase(86400n); // 1 day later
                await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice / 2n]); // Price halved

                // Try to Close Position
                const { request, signature } = await createSignedClosePositionRequest({position});

                await expect(wasabiShortPool.write.closePosition(
                    [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
                )).to.be.rejectedWith("OrderExpired");
            });

            it("OrderExpired - Request", async function () {
                const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, mockSwap, publicClient, wasabiShortPool, user1, uPPG, wethAddress, initialPPGPrice, liquidator } = await loadFixture(deployShortPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                // Take Profit Order
                const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                    orderType: OrderType.TP,
                    traderSigner: user1,
                    positionId: position.id,
                    makerAmount: position.collateralAmount / 2n,
                    takerAmount: position.principal + position.downPayment,
                    expiration: await time.latest() + 172800,
                    executionFee: parseEther("0.05"),
                });

                // Try to Close Position
                const { request, signature } = await createSignedClosePositionRequest({position});

                await time.increase(86400n); // 1 day later
                await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice / 2n]); // Price halved

                await expect(wasabiShortPool.write.closePosition(
                    [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
                )).to.be.rejectedWith("OrderExpired");
            });

            it("InvalidOrder - Position ID", async function () {
                const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, mockSwap, liquidator, wasabiShortPool, user1, uPPG, wethAddress, initialPPGPrice } = await loadFixture(deployShortPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                // Take Profit Order
                const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                    orderType: OrderType.TP,
                    traderSigner: user1,
                    positionId: position.id + 1n, // Invalid Position ID
                    makerAmount: position.collateralAmount / 2n,
                    takerAmount: position.principal + position.downPayment,
                    expiration: await time.latest() + 172800,
                    executionFee: parseEther("0.05"),
                });

                await time.increase(86400n); // 1 day later
                await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice / 2n]); // Price halved

                // Try to Close Position
                const { request, signature } = await createSignedClosePositionRequest({position});

                await expect(wasabiShortPool.write.closePosition(
                    [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
                )).to.be.rejectedWith("InvalidOrder");
            });

            it("InvalidOrder - OrderType", async function () {
                const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, mockSwap, liquidator, wasabiShortPool, user1, uPPG, wethAddress, initialPPGPrice } = await loadFixture(deployShortPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                // Take Profit Order
                const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                    orderType: OrderType.INVALID, // Invalid Order Type
                    traderSigner: user1,
                    positionId: position.id,
                    makerAmount: position.collateralAmount / 2n,
                    takerAmount: position.principal + position.downPayment,
                    expiration: await time.latest() + 172800,
                    executionFee: parseEther("0.05"),
                });

                await time.increase(86400n); // 1 day later
                await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice / 2n]); // Price halved

                // Try to Close Position
                const { request, signature } = await createSignedClosePositionRequest({position});

                await expect(wasabiShortPool.write.closePosition(
                    [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
                )).to.be.rejectedWith("InvalidOrder");
            });

            it("InvalidSignature - Order", async function () {
                const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, mockSwap, liquidator, wasabiShortPool, user2, uPPG, wethAddress, initialPPGPrice } = await loadFixture(deployShortPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                // Take Profit Order
                const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                    orderType: OrderType.TP,
                    traderSigner: user2, // Wrong Signer
                    positionId: position.id,
                    makerAmount: position.collateralAmount / 2n,
                    takerAmount: position.principal + position.downPayment,
                    expiration: await time.latest() + 172800,
                    executionFee: parseEther("0.05"),
                });

                await time.increase(86400n); // 1 day later
                await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice / 2n]); // Price halved

                // Try to Close Position
                const { request, signature } = await createSignedClosePositionRequest({position});

                await expect(wasabiShortPool.write.closePosition(
                    [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
                )).to.be.rejectedWith("InvalidSignature");
            });

            it("InvalidSignature - Request", async function () {
                const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, mockSwap, liquidator, wasabiShortPool, user1, uPPG, wethAddress, initialPPGPrice } = await loadFixture(deployShortPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                // Take Profit Order
                const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                    orderType: OrderType.TP,
                    traderSigner: user1,
                    positionId: position.id,
                    makerAmount: position.collateralAmount / 2n,
                    takerAmount: position.principal + position.downPayment,
                    expiration: await time.latest() + 172800,
                    executionFee: parseEther("0.05"),
                });

                await time.increase(86400n); // 1 day later
                await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice / 2n]); // Price halved

                // Try to Close Position
                const { request } = await createSignedClosePositionRequest({position});

                await expect(wasabiShortPool.write.closePosition(
                    [PayoutType.UNWRAPPED, request, orderSignature, order, orderSignature], {account: liquidator.account}
                )).to.be.rejectedWith("InvalidSignature");
            });
        });
    });

    describe("Stop Loss", function () {
        it("Price increased to exact target", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, computeMaxInterest, mockSwap, publicClient, wasabiShortPool, user1, uPPG, feeReceiver, wethAddress, initialPPGPrice, liquidator, vault } = await loadFixture(deployShortPoolMockEnvironment);

            // Open Position
            const vaultBalanceInitial = await getBalance(publicClient, uPPG.address, vault.address);
            const {position} = await sendDefaultOpenPositionRequest();

            // Stop Loss Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.SL,
                traderSigner: user1,
                positionId: position.id,
                makerAmount: position.collateralAmount * 11n / 10n,
                takerAmount: position.principal + position.downPayment,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice * 11n / 10n]); // Price rose by 10%

            // Close Position
            const maxInterest = await computeMaxInterest(position);
            const { request, signature } = await createSignedClosePositionRequest({ position, interest: maxInterest });
            
            const vaultBalanceBefore = await getBalance(publicClient, uPPG.address, vault.address);
            const userBalanceBefore = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceBefore = await publicClient.getBalance({ address: feeReceiver });
        
            const hash = await wasabiShortPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const vaultBalanceAfter = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
            const userBalanceAfter = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceAfter = await publicClient.getBalance({ address: feeReceiver });

            // Checks
            const events = await wasabiShortPool.getEvents.PositionClosedWithOrder();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal);
            expect(closePositionEvent.interestPaid!).to.equal(maxInterest, "If given interest value is 0, should use max interest");

            expect(vaultBalanceAfter).eq(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!, "Invalid repay amount");
            expect(vaultBalanceInitial + closePositionEvent.interestPaid!).eq(vaultBalanceAfter, "Original amount + interest wasn't repayed");

            expect(balancesAfter.get(wasabiShortPool.address)).to.equal(0, "Pool should not have any collateral left");

            const interestPaidInEth = closePositionEvent.interestPaid! * 11n / 10n;
            const totalReturn = closePositionEvent.payout! + closePositionEvent.feeAmount! - position.downPayment + interestPaidInEth;
            expect(totalReturn).to.be.approximately(position.downPayment / -2n, parseEther("0.001"), "On 10% price increase w/ 5x leverage, total return should be approximately -0.5x down payment");

            // Check trader has been paid
            expect(userBalanceAfter - userBalanceBefore).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            const totalFeesPaid = closePositionEvent.feeAmount!;
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        it("Price increased above target", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, computeMaxInterest, mockSwap, publicClient, wasabiShortPool, user1, uPPG, feeReceiver, wethAddress, initialPPGPrice, liquidator, vault } = await loadFixture(deployShortPoolMockEnvironment);

            // Open Position
            const vaultBalanceInitial = await getBalance(publicClient, uPPG.address, vault.address);
            const {position} = await sendDefaultOpenPositionRequest();

            // Stop Loss Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.SL,
                traderSigner: user1,
                positionId: position.id,
                makerAmount: position.collateralAmount * 21n / 20n,
                takerAmount: position.principal + position.downPayment,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice * 11n / 10n]); // Price rose by 10%

            // Close Position
            const maxInterest = await computeMaxInterest(position);
            const { request, signature } = await createSignedClosePositionRequest({ position, interest: maxInterest });
            
            const vaultBalanceBefore = await getBalance(publicClient, uPPG.address, vault.address);
            const userBalanceBefore = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceBefore = await publicClient.getBalance({ address: feeReceiver });
        
            const hash = await wasabiShortPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const vaultBalanceAfter = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
            const userBalanceAfter = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceAfter = await publicClient.getBalance({ address: feeReceiver });

            // Checks
            const events = await wasabiShortPool.getEvents.PositionClosedWithOrder();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal);
            expect(closePositionEvent.interestPaid!).to.equal(maxInterest, "If given interest value is 0, should use max interest");

            expect(vaultBalanceAfter).eq(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!, "Invalid repay amount");
            expect(vaultBalanceInitial + closePositionEvent.interestPaid!).eq(vaultBalanceAfter, "Original amount + interest wasn't repayed");

            expect(balancesAfter.get(wasabiShortPool.address)).to.equal(0, "Pool should not have any collateral left");

            const interestPaidInEth = closePositionEvent.interestPaid! * 11n / 10n;
            const totalReturn = closePositionEvent.payout! + closePositionEvent.feeAmount! - position.downPayment + interestPaidInEth;
            expect(totalReturn).to.be.approximately(position.downPayment / -2n, parseEther("0.001"), "On 10% price increase w/ 5x leverage, total return should be approximately -0.5x down payment");

            // Check trader has been paid
            expect(userBalanceAfter - userBalanceBefore).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            const totalFeesPaid = closePositionEvent.feeAmount!;
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        it("Price increased to exact target - Partial close", async function () {
            const { sendDefaultOpenPositionRequest, createClosePositionRequest, createSignedClosePositionOrder, computeMaxInterest, mockSwap, publicClient, wasabiShortPool, user1, uPPG, feeReceiver, wethAddress, initialPPGPrice, liquidator, orderSigner, contractName, vault } = await loadFixture(deployShortPoolMockEnvironment);

            // Open Position
            const vaultBalanceInitial = await getBalance(publicClient, uPPG.address, vault.address);
            const {position} = await sendDefaultOpenPositionRequest();
            const closeAmountDenominator = 2n;

            // Stop Loss Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.SL,
                traderSigner: user1,
                positionId: position.id,
                makerAmount: position.collateralAmount / closeAmountDenominator * 11n / 10n,
                takerAmount: (position.principal + position.downPayment) / closeAmountDenominator,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice * 11n / 10n]); // Price rose by 10%

            // Close Position
            const interest = await computeMaxInterest(position) / closeAmountDenominator;
            const amount = position.principal / closeAmountDenominator;
            const request = await createClosePositionRequest({ position, interest, amount }); 
            const signature = await signClosePositionRequest(orderSigner, contractName, wasabiShortPool.address, request);
            
            const vaultBalanceBefore = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
            const userBalanceBefore = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceBefore = await publicClient.getBalance({ address: feeReceiver });
        
            const hash = await wasabiShortPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const vaultBalanceAfter = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
            const userBalanceAfter = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceAfter = await publicClient.getBalance({ address: feeReceiver });

            // Checks
            const events = await wasabiShortPool.getEvents.PositionDecreasedWithOrder();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal / closeAmountDenominator, "Half of the principal should be repaid");
            expect(closePositionEvent.interestPaid!).to.equal(interest, "Prorated interest should be paid");

            expect(vaultBalanceAfter).eq(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!, "Invalid repay amount");
            expect(vaultBalanceInitial + closePositionEvent.interestPaid! - position.principal / closeAmountDenominator).eq(vaultBalanceAfter, "Half of original amount + interest wasn't repayed");

            expect(balancesAfter.get(wasabiShortPool.address)).to.equal(balancesBefore.get(wasabiShortPool.address) / closeAmountDenominator, "Pool should have half of the collateral left");

            const downPaymentReduced = position.downPayment / closeAmountDenominator;
            const interestPaidInEth = closePositionEvent.interestPaid! * 11n / 10n;
            const totalReturn = closePositionEvent.payout! + interestPaidInEth + closePositionEvent.closeFee! - downPaymentReduced;
            expect(totalReturn).to.be.approximately(downPaymentReduced / -2n, parseEther("0.001"), "On 10% price increase w/ 5x leverage, total return should be approximately -0.5x down payment");

            // Check trader has been paid
            expect(userBalanceAfter - userBalanceBefore).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            const totalFeesPaid = closePositionEvent.closeFee!;
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        it("Price increased to exact target - Partial close after increase", async function () {
            const { sendDefaultOpenPositionRequest, createClosePositionRequest, createSignedClosePositionOrder, computeMaxInterest, mockSwap, publicClient, wasabiShortPool, user1, uPPG, feeReceiver, wethAddress, initialPPGPrice, priceDenominator, totalAmountIn, principal, liquidator, orderSigner, contractName, vault } = await loadFixture(deployShortPoolMockEnvironment);

            // Open Position
            const vaultBalanceInitial = await getBalance(publicClient, uPPG.address, vault.address);
            const {position} = await sendDefaultOpenPositionRequest();
            const closeAmountDenominator = 2n;

            // Stop Loss Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.SL,
                traderSigner: user1,
                positionId: position.id,
                makerAmount: position.collateralAmount / closeAmountDenominator * 11n / 10n,
                takerAmount: (position.principal + position.downPayment) / closeAmountDenominator,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            // Increase Position
            const functionCallDataList: FunctionCallData[] =
                getApproveAndSwapFunctionCallData(mockSwap.address, uPPG.address, wethAddress, principal);
            const openPositionRequest: OpenPositionRequest = {
                id: position.id,
                currency: position.currency,
                targetCurrency: position.collateralCurrency,
                downPayment: position.downPayment,
                principal: position.principal,
                minTargetAmount: principal * initialPPGPrice / priceDenominator,
                expiration: BigInt(await time.latest()) + 86400n,
                fee: position.feesToBePaid,
                functionCallDataList,
                existingPosition: position,
                referrer: zeroAddress
            };
            const increaseSignature = await signOpenPositionRequest(orderSigner, contractName, wasabiShortPool.address, openPositionRequest);

            await wasabiShortPool.write.openPosition([openPositionRequest, increaseSignature], { value: totalAmountIn, account: user1.account });

            const increaseEvents = await wasabiShortPool.getEvents.PositionIncreased();
            expect(increaseEvents).to.have.lengthOf(1);
            const increaseEvent = increaseEvents[0].args;
            position.collateralAmount += increaseEvent.collateralAdded!;
            position.downPayment += increaseEvent.downPaymentAdded!;
            position.principal += increaseEvent.principalAdded!;
            position.feesToBePaid += increaseEvent.feesAdded!;

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice * 11n / 10n]); // Price rose by 10%

            // Close Position
            const interest = await computeMaxInterest(position) / closeAmountDenominator;
            const amount = position.principal / closeAmountDenominator;
            const request = await createClosePositionRequest({ position, interest, amount }); 
            const signature = await signClosePositionRequest(orderSigner, contractName, wasabiShortPool.address, request);
            
            const vaultBalanceBefore = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
            const userBalanceBefore = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceBefore = await publicClient.getBalance({ address: feeReceiver });
        
            const hash = await wasabiShortPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const vaultBalanceAfter = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
            const userBalanceAfter = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceAfter = await publicClient.getBalance({ address: feeReceiver });

            // Checks
            const events = await wasabiShortPool.getEvents.PositionDecreasedWithOrder();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal / closeAmountDenominator, "Half of the principal should be repaid");
            expect(closePositionEvent.interestPaid!).to.equal(interest, "Prorated interest should be paid");

            expect(vaultBalanceAfter).eq(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!, "Invalid repay amount");
            expect(vaultBalanceInitial + closePositionEvent.interestPaid! - position.principal / closeAmountDenominator).eq(vaultBalanceAfter, "Half of original amount + interest wasn't repayed");

            expect(balancesAfter.get(wasabiShortPool.address)).to.equal(balancesBefore.get(wasabiShortPool.address) / closeAmountDenominator, "Pool should have half of the collateral left");

            const downPaymentReduced = position.downPayment / closeAmountDenominator;
            const interestPaidInEth = closePositionEvent.interestPaid! * 11n / 10n;
            const totalReturn = closePositionEvent.payout! + interestPaidInEth + closePositionEvent.closeFee! - downPaymentReduced;
            expect(totalReturn).to.be.approximately(downPaymentReduced / -2n, parseEther("0.001"), "On 10% price increase w/ 5x leverage, total return should be approximately -0.5x down payment");

            // Check trader has been paid
            expect(userBalanceAfter - userBalanceBefore).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            const totalFeesPaid = closePositionEvent.closeFee!;
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        it("Price increased to exact target - Partial close after adding collateral", async function () {
            const { sendDefaultOpenPositionRequest, createClosePositionRequest, createSignedClosePositionOrder, computeMaxInterest, mockSwap, publicClient, wasabiShortPool, user1, uPPG, feeReceiver, wethAddress, initialPPGPrice, priceDenominator, downPayment, liquidator, orderSigner, contractName, vault } = await loadFixture(deployShortPoolMockEnvironment);

            // Open Position
            const vaultBalanceInitial = await getBalance(publicClient, uPPG.address, vault.address);
            const {position} = await sendDefaultOpenPositionRequest();
            const closeAmountDenominator = 2n;

            // Stop Loss Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.SL,
                traderSigner: user1,
                positionId: position.id,
                makerAmount: position.collateralAmount / closeAmountDenominator * 11n / 10n,
                takerAmount: (position.principal + position.downPayment) / closeAmountDenominator,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            // Add Collateral
            const addCollateralRequest: AddCollateralRequest = {
                amount: downPayment,
                interest: 0n,
                expiration: BigInt(await time.latest()) + 86400n,
                position
            }
            const addCollateralSignature = await signAddCollateralRequest(orderSigner, contractName, wasabiShortPool.address, addCollateralRequest);

            await wasabiShortPool.write.addCollateral([addCollateralRequest, addCollateralSignature], { value: downPayment, account: user1.account });

            const addCollateralEvents = await wasabiShortPool.getEvents.CollateralAdded();
            expect(addCollateralEvents).to.have.lengthOf(1);
            const addCollateralEvent = addCollateralEvents[0].args;
            position.collateralAmount += addCollateralEvent.collateralAdded!;
            position.downPayment += addCollateralEvent.downPaymentAdded!;
            const newLeverage = position.principal * 100n / (position.downPayment * initialPPGPrice / priceDenominator);

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice * 11n / 10n]); // Price rose by 10%

            // Close Position
            const interest = await computeMaxInterest(position) / closeAmountDenominator;
            const amount = position.principal / closeAmountDenominator;
            const request = await createClosePositionRequest({ position, interest, amount }); 
            const signature = await signClosePositionRequest(orderSigner, contractName, wasabiShortPool.address, request);
            
            const vaultBalanceBefore = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
            const userBalanceBefore = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceBefore = await publicClient.getBalance({ address: feeReceiver });
        
            const hash = await wasabiShortPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const vaultBalanceAfter = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
            const userBalanceAfter = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceAfter = await publicClient.getBalance({ address: feeReceiver });

            // Checks
            const events = await wasabiShortPool.getEvents.PositionDecreasedWithOrder();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal / closeAmountDenominator, "Half of the principal should be repaid");
            expect(closePositionEvent.interestPaid!).to.equal(interest, "Prorated interest should be paid");

            expect(vaultBalanceAfter).eq(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!, "Invalid repay amount");
            expect(vaultBalanceInitial + closePositionEvent.interestPaid! - position.principal / closeAmountDenominator).eq(vaultBalanceAfter, "Half of original amount + interest wasn't repayed");

            expect(balancesAfter.get(wasabiShortPool.address)).to.equal(balancesBefore.get(wasabiShortPool.address) / closeAmountDenominator, "Pool should have half of the collateral left");

            const downPaymentReduced = position.downPayment / closeAmountDenominator;
            const interestPaidInEth = closePositionEvent.interestPaid! * 11n / 10n;
            const totalReturn = closePositionEvent.payout! + interestPaidInEth + closePositionEvent.closeFee! - downPaymentReduced;
            expect(totalReturn).to.be.approximately(downPaymentReduced / -10n * newLeverage / 100n, parseEther("0.001"), "On 10% price increase, total return should be approximately -10% * new leverage * adjusted down payment");

            // Check trader has been paid
            expect(userBalanceAfter - userBalanceBefore).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            const totalFeesPaid = closePositionEvent.closeFee!;
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        it("Price increased to exact target - Full close after decrease", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, computeMaxInterest, mockSwap, publicClient, wasabiShortPool, user1, uPPG, feeReceiver, wethAddress, initialPPGPrice, liquidator, vault } = await loadFixture(deployShortPoolMockEnvironment);

            // Open Position
            const vaultBalanceInitial = await getBalance(publicClient, uPPG.address, vault.address);
            const {position} = await sendDefaultOpenPositionRequest();

            // Stop Loss Order
            const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                orderType: OrderType.SL,
                traderSigner: user1,
                positionId: position.id,
                makerAmount: position.collateralAmount * 11n / 10n,
                takerAmount: position.principal + position.downPayment,
                expiration: await time.latest() + 172800,
                executionFee: parseEther("0.05"),
            });

            // Close Half of the Position
            const closeAmountDenominator = 2n;
            const interest = await computeMaxInterest(position) / closeAmountDenominator;
            const amount = position.principal / closeAmountDenominator;
            const {request: partialRequest, signature: partialSignature} = await createSignedClosePositionRequest({ position, interest, amount }); 

            await wasabiShortPool.write.closePosition([PayoutType.UNWRAPPED, partialRequest, partialSignature], { account: user1.account });

            const partialCloseEvents = await wasabiShortPool.getEvents.PositionDecreased();
            expect(partialCloseEvents).to.have.lengthOf(1);
            const partialCloseEvent = partialCloseEvents[0].args;
            position.collateralAmount -= partialCloseEvent.collateralReduced!;
            position.downPayment -= partialCloseEvent.downPaymentReduced!;
            position.principal -= partialCloseEvent.principalRepaid!;
            position.feesToBePaid -= partialCloseEvent.pastFees!;

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice * 11n / 10n]); // Price rose by 10%

            // Close Position
            const maxInterest = await computeMaxInterest(position);
            const { request, signature } = await createSignedClosePositionRequest({ position, interest: maxInterest });
            
            const vaultBalanceBefore = await getBalance(publicClient, uPPG.address, vault.address);
            const userBalanceBefore = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceBefore = await publicClient.getBalance({ address: feeReceiver });
        
            const hash = await wasabiShortPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
            );

            const vaultBalanceAfter = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
            const userBalanceAfter = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceAfter = await publicClient.getBalance({ address: feeReceiver });

            // Checks
            const events = await wasabiShortPool.getEvents.PositionClosedWithOrder();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal);
            expect(closePositionEvent.interestPaid!).to.equal(maxInterest, "If given interest value is 0, should use max interest");

            expect(vaultBalanceAfter).eq(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!, "Invalid repay amount");
            expect(vaultBalanceInitial + closePositionEvent.interestPaid! + partialCloseEvent.interestPaid!).eq(vaultBalanceAfter, "Original amount + interest wasn't repayed");

            expect(balancesAfter.get(wasabiShortPool.address)).to.equal(0, "Pool should not have any collateral left");

            const interestPaidInEth = closePositionEvent.interestPaid! * 11n / 10n;
            const totalReturn = closePositionEvent.payout! + closePositionEvent.feeAmount! - position.downPayment + interestPaidInEth;
            expect(totalReturn).to.be.approximately(position.downPayment / -2n, parseEther("0.001"), "On 10% price increase w/ 5x leverage, total return should be approximately -0.5x down payment");

            // Check trader has been paid
            expect(userBalanceAfter - userBalanceBefore).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            const totalFeesPaid = closePositionEvent.feeAmount!;
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        describe("Validations", function () {
            it("PriceTargetNotReached", async function () {
                const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, createSignedClosePositionOrder, mockSwap, wasabiShortPool, user1, uPPG, wethAddress, initialPPGPrice, liquidator } = await loadFixture(deployShortPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                // Stop Loss Order
                const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                    orderType: OrderType.SL,
                    traderSigner: user1,
                    positionId: position.id,
                    makerAmount: position.collateralAmount * 11n / 10n,
                    takerAmount: position.principal + position.downPayment,
                    expiration: await time.latest() + 172800,
                    executionFee: parseEther("0.05"),
                });

                await time.increase(86400n); // 1 day later
                await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice * 21n / 20n]); // Price increased by 5%

                // Try to Close Position
                const { request, signature } = await createSignedClosePositionRequest({position});

                await expect(wasabiShortPool.write.closePosition(
                    [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
                )).to.be.rejectedWith("PriceTargetNotReached");
            });

            it("InsufficientPrincipalRepaid - Bad debt from bad swap function call", async function () {
                const { sendDefaultOpenPositionRequest, createSignedClosePositionOrder, orderSigner, contractName, wasabiShortPool, user1, uPPG, mockSwap, initialPPGPrice, wethAddress, liquidator } = await loadFixture(deployShortPoolMockEnvironment);

                // Open Position
                const {position} = await sendDefaultOpenPositionRequest();

                // Stop Loss Order
                const {request: order, signature: orderSignature} = await createSignedClosePositionOrder({
                    orderType: OrderType.SL,
                    traderSigner: user1,
                    positionId: position.id,
                    makerAmount: position.collateralAmount * 11n / 10n,
                    takerAmount: position.principal + position.downPayment,
                    expiration: await time.latest() + 172800,
                    executionFee: parseEther("0.05"),
                });

                await time.increase(86400n); // 1 day later
                await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice * 11n / 10n]); // Price rose by 10%

                // Craft a ClosePositionRequest with a malicious swap function call using MockSwap.swapExact
                const request: ClosePositionRequest = {
                    expiration: BigInt(await time.latest() + 172800),
                    interest: 0n,
                    amount: 0n,
                    position,
                    functionCallDataList: getApproveAndSwapFunctionCallDataExact(mockSwap.address, position.collateralCurrency, position.currency, position.collateralAmount, 1n), // bad amountOut
                    referrer: zeroAddress
                };
                const signature = await signClosePositionRequest(orderSigner, contractName, wasabiShortPool.address, request);

                await expect(wasabiShortPool.write.closePosition(
                    [PayoutType.UNWRAPPED, request, signature, order, orderSignature], {account: liquidator.account}
                )).to.be.rejectedWith("InsufficientPrincipalRepaid");
            });
        });
    });
});