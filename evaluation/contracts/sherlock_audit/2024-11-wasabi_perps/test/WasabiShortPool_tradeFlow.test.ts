import {
    time,
    loadFixture,
} from "@nomicfoundation/hardhat-toolbox-viem/network-helpers";
import { expect } from "chai";
import { getAddress, parseEther, maxUint256, zeroAddress, parseUnits, encodeAbiParameters, keccak256 } from "viem";
import { FunctionCallData, OpenPositionRequest, getFee, PayoutType, Position, AddCollateralRequest, RemoveCollateralRequest } from "./utils/PerpStructUtils";
import { getApproveAndSwapExactlyOutFunctionCallData, getApproveAndSwapFunctionCallData } from "./utils/SwapUtils";
import { deployShortPoolMockEnvironment, deployPoolsAndRouterMockEnvironment } from "./fixtures";
import { getBalance, takeBalanceSnapshot } from "./utils/StateUtils";
import { signAddCollateralRequest, signOpenPositionRequest, signRemoveCollateralRequest } from "./utils/SigningUtils";

describe("WasabiShortPool - Trade Flow Test", function () {

    describe("Open Position", function () {
        it("Open Position", async function () {
            const { wasabiShortPool, tradeFeeValue, publicClient, user1, openPositionRequest, downPayment, signature, feeReceiver, weth } = await loadFixture(deployShortPoolMockEnvironment);

            const feeReceiverBalanceBefore = await weth.read.balanceOf([feeReceiver]);

            const hash = await wasabiShortPool.write.openPosition([openPositionRequest, signature], { account: user1.account });

            const feeReceiverBalanceAfter = await weth.read.balanceOf([feeReceiver]);

            const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed);
            console.log('gas used to open', gasUsed);

            const events = await wasabiShortPool.getEvents.PositionOpened();
            expect(events).to.have.lengthOf(1);
            const event = events[0].args;
            expect(event.positionId).to.equal(openPositionRequest.id);
            expect(event.downPayment).to.equal(downPayment);
            expect(event.collateralAmount!).to.equal(await getBalance(publicClient, weth.address, wasabiShortPool.address));
            expect(event.feesToBePaid!).to.equal(feeReceiverBalanceAfter - feeReceiverBalanceBefore);
        });

        it("Open Position with USDC", async function () {
            const { wasabiShortPool, publicClient, usdc, sendUSDCOpenPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);
            const positionId = 1337n;

            await wasabiShortPool.write.addQuoteToken([usdc.address]);

            const {position, event, downPayment} = await sendUSDCOpenPositionRequest(positionId);
            expect(event.args.positionId).to.equal(positionId);
            expect(event.args.downPayment).to.equal(downPayment);
            expect(event.args.collateralAmount!).to.equal(await getBalance(publicClient, usdc.address, wasabiShortPool.address), "Collateral amount + fees to be paid should be equal to the amount of USDC in the pool after opening the position");
        });

        it("Open and Increase Position", async function () {
            const { wasabiShortPool, user1, totalAmountIn, principal, initialPPGPrice, priceDenominator, wethAddress, mockSwap, uPPG, weth, orderSigner, contractName, sendDefaultOpenPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();

            await time.increase(86400n); // 1 day later

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
            const signature = await signOpenPositionRequest(orderSigner, contractName, wasabiShortPool.address, openPositionRequest);

            // Increase Position
            await wasabiShortPool.write.openPosition([openPositionRequest, signature], { value: totalAmountIn, account: user1.account });

            const events = await wasabiShortPool.getEvents.PositionIncreased();
            expect(events).to.have.lengthOf(1);
            const eventData = events[0].args;
            expect(eventData.id).to.equal(position.id);
            expect(eventData.downPaymentAdded).to.equal(totalAmountIn - eventData.feesAdded!);
            expect(eventData.principalAdded).to.equal(openPositionRequest.principal);
            expect(eventData.collateralAdded! + position.collateralAmount).to.equal(await weth.read.balanceOf([wasabiShortPool.address]));
            expect(eventData.collateralAdded).to.greaterThanOrEqual(openPositionRequest.minTargetAmount);
        });

        it("Open Position and Add Collateral", async function () {
            const { wasabiShortPool, user1, downPayment, weth, orderSigner, contractName, sendDefaultOpenPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();

            await time.increase(86400n); // 1 day later

            const addCollateralRequest: AddCollateralRequest = {
                amount: downPayment,
                interest: 0n,
                expiration: BigInt(await time.latest()) + 86400n,
                position
            };
            const signature = await signAddCollateralRequest(orderSigner, contractName, wasabiShortPool.address, addCollateralRequest);

            // Add Collateral
            await wasabiShortPool.write.addCollateral([addCollateralRequest, signature], { value: downPayment, account: user1.account });

            const events = await wasabiShortPool.getEvents.CollateralAdded();
            expect(events).to.have.lengthOf(1);
            const eventData = events[0].args;
            expect(eventData.id).to.equal(position.id);
            expect(eventData.collateralAdded! + position.collateralAmount).to.equal(await weth.read.balanceOf([wasabiShortPool.address]));
            expect(eventData.collateralAdded).to.equal(downPayment);
            expect(eventData.downPaymentAdded).to.equal(downPayment);
        });
    });

    describe("Close Position", function () {
        it("Price Not Changed", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, computeMaxInterest, mockSwap, publicClient, wasabiShortPool, user1, uPPG, feeReceiver, wethAddress, vault } = await loadFixture(deployShortPoolMockEnvironment);

            // Open Position
            const vaultBalanceInitial = await getBalance(publicClient, uPPG.address, vault.address);
            const {position} = await sendDefaultOpenPositionRequest();

            await time.increase(86400n * 3n); // 3 days later

            // Close Position
            const maxInterest = await computeMaxInterest(position);
            const { request, signature } = await createSignedClosePositionRequest({ position, interest: maxInterest });
            
            const vaultBalanceBefore = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
            const userBalanceBefore = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceBefore = await publicClient.getBalance({ address: feeReceiver });
            const feeReceiverSharesBefore = await vault.read.balanceOf([feeReceiver]);
        
            const hash = await wasabiShortPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account });

            const vaultBalanceAfter = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
            const userBalanceAfter = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceAfter = await publicClient.getBalance({ address: feeReceiver });
            const feeReceiverSharesAfter = await vault.read.balanceOf([feeReceiver]);

            // Checks
            const events = await wasabiShortPool.getEvents.PositionClosed();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal);
            expect(closePositionEvent.interestPaid!).to.equal(maxInterest, "If given interest value is 0, should use max interest");

            // Interest is paid in uPPG, so the principal should be equal before and after the trade
            expect(vaultBalanceAfter).eq(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!, "Invalid repay amount");
            expect(vaultBalanceInitial + closePositionEvent.interestPaid!).eq(vaultBalanceAfter, "Original amount + interest wasn't repayed");

            expect(balancesAfter.get(wasabiShortPool.address)).to.equal(0);

            const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.feeAmount! - position.downPayment;
            expect(totalReturn).to.equal(0, "Total return should be 0 on no price change");

            // Check trader has been paid
            const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
            expect(userBalanceAfter - userBalanceBefore).to.equal(closePositionEvent.payout! - gasUsed);

            // Check fees have been paid
            const totalFeesPaid = closePositionEvent.feeAmount!;
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
            expect(feeReceiverSharesAfter - feeReceiverSharesBefore).to.equal(maxInterest / 10n);
        });

        it("Custom interest", async function () {
            const { sendDefaultOpenPositionRequest, createSignedClosePositionRequest, computeMaxInterest, mockSwap, publicClient, wasabiShortPool, user1, uPPG, feeReceiver, wethAddress, vault } = await loadFixture(deployShortPoolMockEnvironment);

            // Open Position
            const vaultBalanceInitial = await getBalance(publicClient, uPPG.address, vault.address);
            const {position} = await sendDefaultOpenPositionRequest();

            await time.increase(86400n); // 1 day later

            // Close Position
            const maxInterest = await computeMaxInterest(position);
            const interest = maxInterest / 2n;
            const { request, signature } = await createSignedClosePositionRequest({ position, interest });

            const vaultBalanceBefore = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
            const userBalanceBefore = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceBefore = await publicClient.getBalance({ address: feeReceiver });
            const feeReceiverSharesBefore = await vault.read.balanceOf([feeReceiver]);
        
            const hash = await wasabiShortPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account });

            const vaultBalanceAfter = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
            const userBalanceAfter = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceAfter = await publicClient.getBalance({ address: feeReceiver });
            const feeReceiverSharesAfter = await vault.read.balanceOf([feeReceiver]);
            
            // Checks
            const events = await wasabiShortPool.getEvents.PositionClosed();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal);
            expect(closePositionEvent.interestPaid!).to.equal(interest, "Not payed custom interest");

            // Interest is paid in ETH, so the principal should be equal before and after the trade
            expect(vaultBalanceAfter).eq(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!, "Invalid repay amount");
            expect(vaultBalanceInitial + closePositionEvent.interestPaid!).eq(vaultBalanceAfter, "Original amount + interest wasn't repayed");

            expect(balancesAfter.get(wasabiShortPool.address)).to.equal(0);

            const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.feeAmount! - position.downPayment;
            expect(totalReturn).to.equal(0, "Total return should be 0 on no price change");

            // Check trader has been paid
            const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
            expect(userBalanceAfter - userBalanceBefore).to.equal(closePositionEvent.payout! - gasUsed);

            // Check fees have been paid
            const totalFeesPaid = closePositionEvent.feeAmount!;
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
            expect(feeReceiverSharesAfter - feeReceiverSharesBefore).to.equal(interest / 10n);
        });

        it("Price Decreased - USDC payout", async function () {
            const { wasabiShortPool, vault, usdc, uPPG, mockSwap, wethAddress, initialPPGPrice, initialUSDCPrice, priceDenominator, user1, feeReceiver, publicClient, sendUSDCOpenPositionRequest, computeMaxInterest, createSignedClosePositionRequest } = await loadFixture(deployShortPoolMockEnvironment);
            const positionId = 1337n;

            await wasabiShortPool.write.addQuoteToken([usdc.address]);

            // Open Position with USDC
            const vaultBalanceInitial = await getBalance(publicClient, uPPG.address, vault.address);
            const {position} = await sendUSDCOpenPositionRequest(positionId);
            expect(position.trader).to.equal(user1.account.address);

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice / 2n]); // price halved
            await mockSwap.write.setPrice([usdc.address, uPPG.address, initialUSDCPrice * 2n]); // price halved

            // Close Position
            const maxInterest = await computeMaxInterest(position);
            const { request, signature } = await createSignedClosePositionRequest({ position, interest: maxInterest });
            expect(request.position.trader).to.equal(user1.account.address);
            
            const vaultBalanceBefore = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesBefore = await takeBalanceSnapshot(publicClient, usdc.address, user1.account.address, wasabiShortPool.address, feeReceiver);
        
            const hash = await wasabiShortPool.write.closePosition(
                [PayoutType.UNWRAPPED, request, signature],
                { account: user1.account }
            );

            const vaultBalanceAfter = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesAfter = await takeBalanceSnapshot(publicClient, usdc.address, user1.account.address, wasabiShortPool.address, feeReceiver);

            // Checks
            const events = await wasabiShortPool.getEvents.PositionClosed();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal);
            expect(closePositionEvent.interestPaid!).to.equal(maxInterest, "If given interest value is 0, should use max interest");
            
            // Interest is paid in uPPG, so the principal should be equal before and after the trade
            expect(vaultBalanceAfter).eq(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!, "Invalid repay amount");
            expect(vaultBalanceInitial + closePositionEvent.interestPaid!).eq(vaultBalanceAfter, "Original amount + interest wasn't repayed");

            expect(balancesAfter.get(wasabiShortPool.address)).to.equal(0, "Short pool should not have any collateral left");

            const interestPaidInEth = closePositionEvent.interestPaid! / 2n;
            const interestPaidInUsdc =
                interestPaidInEth * priceDenominator / initialUSDCPrice / (10n ** (18n - 6n));
            const totalReturn = closePositionEvent.payout! + closePositionEvent.feeAmount! - position.downPayment + interestPaidInUsdc;
            expect(totalReturn).to.equal(position.downPayment * 5n / 2n, "On 50% price decrease w/ 5x leverage, total return should be 2.5x down payment");

            // Check trader has been paid
            expect(balancesAfter.get(user1.account.address) - balancesBefore.get(user1.account.address)).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            const totalFeesPaid = closePositionEvent.feeAmount!;
            expect(balancesAfter.get(feeReceiver) - balancesBefore.get(feeReceiver)).to.equal(totalFeesPaid);
        });

        it("Price Decreased - USDC vault deposit", async function () {
            const { wasabiShortPool, usdc, uPPG, mockSwap, wethAddress, initialPPGPrice, initialUSDCPrice, priceDenominator, usdcVault, vault, user1, feeReceiver, publicClient, sendUSDCOpenPositionRequest, computeMaxInterest, createSignedClosePositionRequest } = await loadFixture(deployShortPoolMockEnvironment);
            const positionId = 1337n;

            await wasabiShortPool.write.addQuoteToken([usdc.address]);

            // Open Position with USDC
            const vaultBalanceInitial = await getBalance(publicClient, uPPG.address, vault.address);
            const {position} = await sendUSDCOpenPositionRequest(positionId);
            expect(position.trader).to.equal(user1.account.address);

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice / 2n]); // price halved
            await mockSwap.write.setPrice([usdc.address, uPPG.address, initialUSDCPrice * 2n]); // price halved

            // Close Position
            const maxInterest = await computeMaxInterest(position);
            const { request, signature } = await createSignedClosePositionRequest({ position, interest: maxInterest });
            expect(request.position.trader).to.equal(user1.account.address);
            
            const vaultBalanceBefore = await getBalance(publicClient, uPPG.address, vault.address);
            const usdcBalancesBefore = await takeBalanceSnapshot(publicClient, usdc.address, user1.account.address, wasabiShortPool.address, usdcVault.address, feeReceiver);
            const usdcVaultSharesBefore = await takeBalanceSnapshot(publicClient, usdcVault.address, user1.account.address);
        
            const hash = await wasabiShortPool.write.closePosition(
                [PayoutType.VAULT_DEPOSIT, request, signature],
                { account: user1.account }
            );

            const vaultBalanceAfter = await getBalance(publicClient, uPPG.address, vault.address);
            const usdcBalancesAfter = await takeBalanceSnapshot(publicClient, usdc.address, user1.account.address, wasabiShortPool.address, usdcVault.address, feeReceiver);
            const usdcVaultSharesAfter = await takeBalanceSnapshot(publicClient, usdcVault.address, user1.account.address);

            // Checks
            const events = await wasabiShortPool.getEvents.PositionClosed();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;
            
            // Interest is paid in uPPG, so the principal should be equal before and after the trade
            expect(vaultBalanceAfter).eq(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!, "Invalid repay amount");
            expect(vaultBalanceInitial + closePositionEvent.interestPaid!).eq(vaultBalanceAfter, "Original amount + interest wasn't repayed");

            expect(usdcBalancesAfter.get(wasabiShortPool.address)).to.equal(0, "Short pool should not have any collateral left");
            expect(usdcBalancesAfter.get(usdcVault.address)).to.equal(closePositionEvent.payout!, "USDC vault should have received the payout");

            const interestPaidInEth = closePositionEvent.interestPaid! / 2n;
            const interestPaidInUsdc =
                interestPaidInEth * priceDenominator / initialUSDCPrice / (10n ** (18n - 6n));
            const totalReturn = closePositionEvent.payout! + closePositionEvent.feeAmount! - position.downPayment + interestPaidInUsdc;
            expect(totalReturn).to.equal(position.downPayment * 5n / 2n, "On 50% price decrease w/ 5x leverage, total return should be 2.5x down payment");

            // Check trader has received vault shares, not USDC
            const expectedNewVaultShares = await usdcVault.read.convertToShares([closePositionEvent.payout!]);
            expect(usdcVaultSharesAfter.get(user1.account.address) - usdcVaultSharesBefore.get(user1.account.address)).to.equal(expectedNewVaultShares);

            // Check fees have been paid
            const totalFeesPaid = closePositionEvent.feeAmount!;
            expect(usdcBalancesAfter.get(feeReceiver) - usdcBalancesBefore.get(feeReceiver)).to.equal(totalFeesPaid);
        });

        describe("Partial Close", function () {
            it("Price Not Changed", async function () {
                const { sendDefaultOpenPositionRequest, createClosePositionRequest, signClosePositionRequest, computeMaxInterest, mockSwap, publicClient, wasabiShortPool, user1, uPPG, feeReceiver, orderSigner, contractName, wethAddress, vault } = await loadFixture(deployShortPoolMockEnvironment);
    
                // Open Position
                const vaultBalanceInitial = await getBalance(publicClient, uPPG.address, vault.address);
                const {position} = await sendDefaultOpenPositionRequest();
                // console.log('position', position);
    
                await time.increase(86400n * 3n); // 3 days later
    
                // Close Half of the Position
                const closeAmountDenominator = 2n;
                const interest = await computeMaxInterest(position) / closeAmountDenominator;
                const amount = position.principal / closeAmountDenominator;
                const request = await createClosePositionRequest({ position, interest, amount }); 
                const signature = await signClosePositionRequest(orderSigner, contractName, wasabiShortPool.address, request);
                
                const vaultBalanceBefore = await getBalance(publicClient, uPPG.address, vault.address);
                const balancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
                const userBalanceBefore = await publicClient.getBalance({ address: user1.account.address });
                const feeReceiverBalanceBefore = await publicClient.getBalance({ address: feeReceiver });
            
                const hash = await wasabiShortPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account });
    
                const vaultBalanceAfter = await getBalance(publicClient, uPPG.address, vault.address);
                const balancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
                const userBalanceAfter = await publicClient.getBalance({ address: user1.account.address });
                const feeReceiverBalanceAfter = await publicClient.getBalance({ address: feeReceiver });
    
                // Checks
                const events = await wasabiShortPool.getEvents.PositionDecreased();
                expect(events).to.have.lengthOf(1);
                const closePositionEvent = events[0].args;

                const swap = (await mockSwap.getEvents.Swap())[0]!.args!;
                // console.log('swap', swap);
        
                expect(closePositionEvent.id).to.equal(position.id);
                expect(closePositionEvent.principalRepaid!).to.equal(position.principal / closeAmountDenominator, "Half of the principal should be repaid");
                expect(closePositionEvent.interestPaid!).to.equal(interest, "Prorated interest should be paid");
                expect(closePositionEvent.downPaymentReduced!).to.equal(position.downPayment / closeAmountDenominator, "Down payment should be reduced by half");
    
                // Interest is paid in uPPG, so the principal should be equal before and after the trade
                expect(vaultBalanceAfter).eq(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!, "Invalid repay amount");
                expect(vaultBalanceInitial + closePositionEvent.interestPaid! - position.principal / closeAmountDenominator).eq(vaultBalanceAfter, "Half of original amount + interest wasn't repayed");
    
                expect(balancesAfter.get(wasabiShortPool.address)).to.equal(balancesBefore.get(wasabiShortPool.address) / closeAmountDenominator, "Pool should have half of the collateral left");
    
                const downPaymentReduced = position.downPayment / closeAmountDenominator;
                const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.closeFee! - downPaymentReduced;
                expect(totalReturn).to.equal(0, "Total return should be 0 on no price change");
    
                // Check trader has been paid
                const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
                expect(userBalanceAfter - userBalanceBefore).to.equal(closePositionEvent.payout! - gasUsed);
    
                // Check fees have been paid
                const totalFeesPaid = closePositionEvent.closeFee!;
                expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
            });

            it("Price Decreased", async function () {
                const { sendDefaultOpenPositionRequest, createClosePositionRequest, signClosePositionRequest, computeMaxInterest, mockSwap, publicClient, wasabiShortPool, user1, uPPG, feeReceiver, orderSigner, contractName, initialPPGPrice, wethAddress, vault } = await loadFixture(deployShortPoolMockEnvironment);
    
                // Open Position
                const vaultBalanceInitial = await getBalance(publicClient, uPPG.address, vault.address);
                const {position} = await sendDefaultOpenPositionRequest();

                await time.increase(86400n); // 1 day later
                
                await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice / 2n]); // price halved
                
                // Close Half of the Position
                const closeAmountDenominator = 2n;
                const interest = await computeMaxInterest(position) / closeAmountDenominator;
                const amount = position.principal / closeAmountDenominator;
                const request = await createClosePositionRequest({ position, interest, amount }); 
                const signature = await signClosePositionRequest(orderSigner, contractName, wasabiShortPool.address, request);

                const vaultBalanceBefore = await getBalance(publicClient, uPPG.address, vault.address);
                const balancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
                const userBalanceBefore = await publicClient.getBalance({ address: user1.account.address });
                const feeReceiverBalanceBefore = await publicClient.getBalance({ address: feeReceiver });
            
                const hash = await wasabiShortPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account });
    
                const vaultBalanceAfter = await getBalance(publicClient, uPPG.address, vault.address);
                const balancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
                const userBalanceAfter = await publicClient.getBalance({ address: user1.account.address });
                const feeReceiverBalanceAfter = await publicClient.getBalance({ address: feeReceiver });
    
                // Checks
                const events = await wasabiShortPool.getEvents.PositionDecreased();
                expect(events).to.have.lengthOf(1);
                const closePositionEvent = events[0].args;

                expect(closePositionEvent.id).to.equal(position.id);
                expect(closePositionEvent.principalRepaid!).to.equal(position.principal / closeAmountDenominator, "Half of the principal should be repaid");
                expect(closePositionEvent.interestPaid!).to.equal(interest, "Prorated interest should be paid");
                expect(closePositionEvent.downPaymentReduced!).to.equal(position.downPayment / closeAmountDenominator, "Down payment should be reduced by half");

                // Interest is paid in uPPG, so the principal should be equal before and after the trade
                expect(vaultBalanceAfter).eq(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!, "Invalid repay amount");
                expect(vaultBalanceInitial + closePositionEvent.interestPaid! - position.principal / closeAmountDenominator).eq(vaultBalanceAfter, "Half of original amount + interest wasn't repayed");
    
                expect(balancesAfter.get(wasabiShortPool.address)).to.equal(balancesBefore.get(wasabiShortPool.address) / closeAmountDenominator, "Pool should have half of the collateral left");

                const downPaymentReduced = position.downPayment / closeAmountDenominator;
                const interestPaidInEth = closePositionEvent.interestPaid! / 2n;
                const totalReturn = closePositionEvent.payout! + interestPaidInEth + closePositionEvent.closeFee! - downPaymentReduced;
                expect(totalReturn).to.equal(downPaymentReduced * 5n / 2n, "On 50% price decrease w/ 5x leverage, total return should be 2.5x adjusted down payment");

                // Check trader has been paid
                const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
                expect(userBalanceAfter - userBalanceBefore).to.equal(closePositionEvent.payout! - gasUsed);
    
                // Check fees have been paid
                const totalFeesPaid = closePositionEvent.closeFee!;
                expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
            });

            it("Price Increased", async function () {
                const { sendDefaultOpenPositionRequest, createClosePositionRequest, signClosePositionRequest, computeMaxInterest, mockSwap, publicClient, wasabiShortPool, user1, uPPG, feeReceiver, orderSigner, contractName, initialPPGPrice, wethAddress, vault } = await loadFixture(deployShortPoolMockEnvironment);
    
                // Open Position
                const vaultBalanceInitial = await getBalance(publicClient, uPPG.address, vault.address);
                const {position} = await sendDefaultOpenPositionRequest();

                await time.increase(86400n); // 1 day later

                await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice * 11n / 10n]); // price rose by 10%

                // Close Half of the Position
                const closeAmountDenominator = 2n;
                const interest = await computeMaxInterest(position) / closeAmountDenominator;
                const amount = position.principal / closeAmountDenominator;
                const request = await createClosePositionRequest({ position, interest, amount }); 
                const signature = await signClosePositionRequest(orderSigner, contractName, wasabiShortPool.address, request);

                const vaultBalanceBefore = await getBalance(publicClient, uPPG.address, vault.address);
                const balancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
                const userBalanceBefore = await publicClient.getBalance({ address: user1.account.address });
                const feeReceiverBalanceBefore = await publicClient.getBalance({ address: feeReceiver });
            
                const hash = await wasabiShortPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account });
    
                const vaultBalanceAfter = await getBalance(publicClient, uPPG.address, vault.address);
                const balancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver);
                const userBalanceAfter = await publicClient.getBalance({ address: user1.account.address });
                const feeReceiverBalanceAfter = await publicClient.getBalance({ address: feeReceiver });
    
                // Checks
                const events = await wasabiShortPool.getEvents.PositionDecreased();
                expect(events).to.have.lengthOf(1);
                const closePositionEvent = events[0].args;

                const swap = (await mockSwap.getEvents.Swap())[0]!.args!;

                expect(closePositionEvent.id).to.equal(position.id);
                expect(closePositionEvent.principalRepaid!).to.equal(position.principal / closeAmountDenominator, "Half of the principal should be repaid");
                expect(closePositionEvent.interestPaid!).to.equal(interest, "Prorated interest should be paid");
                expect(closePositionEvent.downPaymentReduced!).to.equal(position.downPayment / closeAmountDenominator, "Down payment should be reduced by half");

                // Interest is paid in uPPG, so the principal should be equal before and after the trade
                expect(vaultBalanceAfter).eq(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!, "Invalid repay amount");
                expect(vaultBalanceInitial + closePositionEvent.interestPaid! - position.principal / closeAmountDenominator).eq(vaultBalanceAfter, "Half of original amount + interest wasn't repayed");
    
                expect(balancesAfter.get(wasabiShortPool.address)).to.equal(balancesBefore.get(wasabiShortPool.address) / closeAmountDenominator, "Pool should have half of the collateral left");

                const downPaymentReduced = position.downPayment / closeAmountDenominator;
                const interestPaidInEth = closePositionEvent.interestPaid! * 11n / 10n;
                const totalReturn = closePositionEvent.payout! + interestPaidInEth + closePositionEvent.closeFee! - downPaymentReduced;
                expect(totalReturn).to.be.approximately(downPaymentReduced / -2n, parseEther("0.001"), "On 10% price increase w/ 5x leverage, total return should be approximately -0.5x down payment");

                // Check trader has been paid
                const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
                expect(userBalanceAfter - userBalanceBefore).to.equal(closePositionEvent.payout! - gasUsed);
    
                // Check fees have been paid
                const totalFeesPaid = closePositionEvent.closeFee!;
                expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
            });
        });
    });

    describe("Liquidate Position", function () {
        it("liquidate", async function () {
            const { vault, sendDefaultOpenPositionRequest, liquidator, computeMaxInterest, mockSwap, publicClient, wasabiShortPool, user1, uPPG, feeReceiver, liquidationFeeReceiver, wethAddress, computeLiquidationPrice } = await loadFixture(deployShortPoolMockEnvironment);

            // Open Position
            const vaultBalanceInitial = await getBalance(publicClient, uPPG.address, vault.address);
            const {position} = await sendDefaultOpenPositionRequest();

            await time.increase(86400n); // 1 day later

            const maxInterest = await computeMaxInterest(position);
            const functionCallDataList = getApproveAndSwapExactlyOutFunctionCallData(
                mockSwap.address,
                position.collateralCurrency,
                position.currency,
                position.collateralAmount,
                position.principal + maxInterest);

            await expect(wasabiShortPool.write.liquidatePosition([PayoutType.UNWRAPPED, maxInterest, position, functionCallDataList, zeroAddress], { account: liquidator.account }))
                .to.be.rejectedWith("LiquidationThresholdNotReached", "Cannot liquidate position if liquidation price is not reached");

            // Liquidate Position
            const liquidationPrice = await computeLiquidationPrice(position);
            await mockSwap.write.setPrice([uPPG.address, wethAddress, liquidationPrice - 1n]); 

            const vaultBalanceBefore = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesBefore = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver, liquidationFeeReceiver);
            const userBalanceBefore = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceBefore = await publicClient.getBalance({ address: feeReceiver });
            const liquidationFeeReceiverBalanceBefore = await publicClient.getBalance({address: liquidationFeeReceiver });    

            const hash = await wasabiShortPool.write.liquidatePosition([PayoutType.UNWRAPPED, maxInterest, position, functionCallDataList, zeroAddress], { account: liquidator.account });

            const vaultBalanceAfter = await getBalance(publicClient, uPPG.address, vault.address);
            const balancesAfter = await takeBalanceSnapshot(publicClient, wethAddress, user1.account.address, wasabiShortPool.address, feeReceiver, liquidationFeeReceiver);
            const userBalanceAfter = await publicClient.getBalance({ address: user1.account.address });
            const feeReceiverBalanceAfter = await publicClient.getBalance({ address: feeReceiver });
            const liquidationFeeReceiverBalanceAfter = await publicClient.getBalance({address: liquidationFeeReceiver });

            // Checks
            const events = await wasabiShortPool.getEvents.PositionLiquidated();
            expect(events).to.have.lengthOf(1);
            const liquidateEvent = events[0].args;

            const swap = (await mockSwap.getEvents.Swap())[0]!.args!;

            expect(liquidateEvent.id).to.equal(position.id);
            expect(liquidateEvent.principalRepaid!).to.equal(position.principal);
            expect(liquidateEvent.interestPaid!).to.equal(maxInterest, "If given interest value is 0, should use max interest");

            // Interest is paid in ETH, so the principal should be equal before and after the trade
            expect(vaultBalanceAfter).eq(vaultBalanceBefore + liquidateEvent.principalRepaid! + liquidateEvent.interestPaid!, "Invalid repay amount");
            expect(vaultBalanceInitial + liquidateEvent.interestPaid!).eq(vaultBalanceAfter, "Original amount + interest wasn't repayed");

            expect(balancesAfter.get(wasabiShortPool.address)).to.equal(0);

            // const totalReturn = liquidateEvent.payout! + liquidateEvent.interestPaid! + liquidateEvent.feeAmount! - position.downPayment;
            // expect(totalReturn).to.equal(0, "Total return should be 0 on no price change");

            expect(userBalanceAfter - userBalanceBefore).to.equal(liquidateEvent.payout!);

            // Check fees have been paid
            const totalFeesPaid = liquidateEvent.feeAmount!;
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);

            // Check liquidation fee receiver balance
            const liquidationFeeExpected = position.downPayment * 5n / 100n;
            expect(liquidationFeeReceiverBalanceAfter - liquidationFeeReceiverBalanceBefore).to.equal(liquidationFeeExpected);
        });

        it("liquidate with bad debt", async function () {
            const { sendDefaultOpenPositionRequest, computeMaxInterest, owner, publicClient, wasabiShortPool, user1, uPPG, mockSwap, feeReceiver, liquidationFeeReceiver, wethAddress, liquidator, computeLiquidationPrice } = await loadFixture(deployShortPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();

            await time.increase(86400n); // 1 day later

            const maxInterest = await computeMaxInterest(position);
            const functionCallDataList = getApproveAndSwapFunctionCallData(
                mockSwap.address,
                position.collateralCurrency,
                position.currency,
                position.collateralAmount);

            const liquidationPrice = await computeLiquidationPrice(position);
            
            // Liquidation price reached, should liquidate
            await mockSwap.write.setPrice([wethAddress, uPPG.address, liquidationPrice / 2n]); 
            
            await wasabiShortPool.write.liquidatePosition([PayoutType.UNWRAPPED, maxInterest, position, functionCallDataList, zeroAddress], { account: liquidator.account });
    
            // Checks for no payout
            const events = await wasabiShortPool.getEvents.PositionLiquidated();
            expect(events).to.have.lengthOf(1);
            const liquidatePositionEvent = events[0].args;
            expect(liquidatePositionEvent.payout!).to.equal(0n);

            expect(liquidatePositionEvent.principalRepaid!).to.lessThan(position.principal, "Principal should be less than the original principal due to bad debt");
        });
    });

    describe("Remove Collateral", function () {
        it("Price Decreased - Remove Profit", async function () {
            const { wasabiShortPool, vault, usdc, uPPG, mockSwap, wethAddress, initialPPGPrice, initialUSDCPrice, priceDenominator, user1, publicClient, sendUSDCOpenPositionRequest, createSignedClosePositionRequest, orderSigner, hasher } = await loadFixture(deployShortPoolMockEnvironment);
            const positionId = 1337n;

            await wasabiShortPool.write.addQuoteToken([usdc.address]);

            // Open Position with USDC
            const vaultBalanceInitial = await getBalance(publicClient, uPPG.address, vault.address);
            const {position} = await sendUSDCOpenPositionRequest(positionId);
            expect(position.trader).to.equal(user1.account.address);

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([uPPG.address, wethAddress, initialPPGPrice / 2n]); // price halved
            await mockSwap.write.setPrice([usdc.address, uPPG.address, initialUSDCPrice * 2n]); // price halved

            // Remove Collateral
            const amount = position.collateralAmount / 10n;
            const removeCollateralRequest: RemoveCollateralRequest = {
                amount: amount,
                expiration: BigInt(await time.latest()) + 86400n,
                position: position
            };
            const removeCollateralSignature = await signRemoveCollateralRequest(orderSigner, "WasabiShortPool", wasabiShortPool.address, removeCollateralRequest);

            const userBalanceBefore = await usdc.read.balanceOf([user1.account.address]);
            await wasabiShortPool.write.removeCollateral([removeCollateralRequest, removeCollateralSignature], { account: user1.account });
            const userBalanceAfter = await usdc.read.balanceOf([user1.account.address]);

            expect(userBalanceAfter - userBalanceBefore).to.equal(amount);

            const events = await wasabiShortPool.getEvents.CollateralRemoved();
            expect(events).to.have.lengthOf(1);
            const collateralRemovedEvent = events[0].args;
            expect(collateralRemovedEvent.id).to.equal(position.id);
            expect(collateralRemovedEvent.principalAdded).to.equal(0n);
            expect(collateralRemovedEvent.downPaymentReduced).to.equal(0n);
            expect(collateralRemovedEvent.collateralReduced).to.equal(amount);

            position.collateralAmount -= amount;
            const hashedPosition = await hasher.read.hashPosition([position]);
            expect(await wasabiShortPool.read.positions([position.id])).to.equal(hashedPosition);

            // Close Position
            const { request, signature } = await createSignedClosePositionRequest({position});
            await wasabiShortPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account });

            // Close Position Checks
            const positionClosedEvents = await wasabiShortPool.getEvents.PositionClosed();
            expect(positionClosedEvents).to.have.lengthOf(1);
            const closePositionEvent = positionClosedEvents[0].args;

            const interestPaidInEth = closePositionEvent.interestPaid! / 2n;
            const interestPaidInUsdc =
                interestPaidInEth * priceDenominator / initialUSDCPrice / (10n ** (18n - 6n));
            const totalReturn = closePositionEvent.payout! + closePositionEvent.feeAmount! - position.downPayment + interestPaidInUsdc;
            expect(totalReturn).to.equal(position.downPayment * 5n / 2n - amount, "On 50% price decrease w/ 5x leverage, total return should be 2.5x down payment minus the amount removed");
        });
    });

    describe("Interest Recording", function () {
        it("Record Interest with one position", async function () {
            const { sendDefaultOpenPositionRequest, computeMaxInterest, publicClient, wasabiShortPool, vault, liquidator, owner, hasher, mockSwap } = await loadFixture(deployShortPoolMockEnvironment);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();

            await time.increase(86400n); // 1 day later

            const vaultAssetsBefore = await vault.read.totalAssets();
            const vaultBalanceBefore = await getBalance(publicClient, position.currency, vault.address);
            const feeReceiverSharesBefore = await vault.read.balanceOf([owner.account.address]);

            // Record Interest
            const interest = await computeMaxInterest(position);
            const functionCallDataList = getApproveAndSwapExactlyOutFunctionCallData(
                mockSwap.address,
                position.collateralCurrency,
                position.currency,
                0n,
                interest
            );
            const hash = await wasabiShortPool.write.recordInterest([[position], [interest], functionCallDataList], { account: liquidator.account });
            const timestamp = await time.latest();

            const vaultAssetsAfter = await vault.read.totalAssets();
            const vaultBalanceAfter = await getBalance(publicClient, position.currency, vault.address);
            const feeReceiverSharesAfter = await vault.read.balanceOf([owner.account.address]);

            const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed);
            console.log('gas used to record interest for 1 position ', gasUsed);
            
            const events = await wasabiShortPool.getEvents.InterestPaid();
            expect(events).to.have.lengthOf(1);
            const interestPaidEvent = events[0].args;
            expect(interestPaidEvent.id).to.equal(position.id);
            expect(interestPaidEvent.interestPaid).to.equal(interest);

            position.collateralAmount -= interestPaidEvent.collateralReduced!;
            position.downPayment -= interestPaidEvent.downPaymentReduced!;
            position.lastFundingTimestamp = BigInt(timestamp);
            const hashedPosition = await hasher.read.hashPosition([position]);
            expect(await wasabiShortPool.read.positions([position.id])).to.equal(hashedPosition);

            expect(vaultAssetsAfter).to.equal(vaultAssetsBefore + interest);
            expect(vaultBalanceAfter).to.equal(vaultBalanceBefore + interest);
            expect(feeReceiverSharesAfter - feeReceiverSharesBefore).to.equal(interest / 10n);
        })

        it("Record Interest with 10 positions", async function () {
            const { sendDefaultOpenPositionRequest, computeMaxInterest, publicClient, wasabiShortPool, vault, liquidator, owner, hasher, mockSwap } = await loadFixture(deployShortPoolMockEnvironment);

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

            const functionCallDataList = getApproveAndSwapExactlyOutFunctionCallData(
                mockSwap.address,
                positions[0].collateralCurrency,
                positions[0].currency,
                0n,
                totalInterest
            );
            const hash = await wasabiShortPool.write.recordInterest([positions, interests, functionCallDataList], { account: liquidator.account });
            const timestamp = await time.latest();

            const vaultAssetsAfter = await vault.read.totalAssets();
            const feeReceiverSharesAfter = await vault.read.balanceOf([owner.account.address]);

            const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed);
            console.log('gas used to record interest for 10 positions', gasUsed);

            const events = await wasabiShortPool.getEvents.InterestPaid();
            expect(events).to.have.lengthOf(10);

            for (let i = 0; i < 10; i++) {
                const interestPaidEvent = events[i].args;
                const position = positions[i];
                const interest = interests[i];
                expect(interestPaidEvent.id).to.equal(position.id);
                expect(interestPaidEvent.interestPaid).to.equal(interest);

                position.collateralAmount -= interestPaidEvent.collateralReduced!;
                position.downPayment -= interestPaidEvent.downPaymentReduced!;
                position.lastFundingTimestamp = BigInt(timestamp);
                const hashedPosition = await hasher.read.hashPosition([position]);
                expect(await wasabiShortPool.read.positions([position.id])).to.equal(hashedPosition);
            }

            expect(vaultAssetsAfter).to.equal(vaultAssetsBefore + totalInterest);
            expect(feeReceiverSharesAfter - feeReceiverSharesBefore).to.equal(totalInterest / 10n);
        })

        it("Record Interest with 2 different USDC positions", async function () {
            const { getOpenPositionRequest, sendOpenPositionRequest, getTradeAmounts, computeMaxInterest, publicClient, wasabiShortPool, vault, liquidator, hasher, mockSwap, usdc, user1 } = await loadFixture(deployShortPoolMockEnvironment);

            // Open 2 positions
            const positions = [];

            // 2x uPPG/USDC short
            const leverage1 = 2n;
            const totalAmountIn1 = parseUnits("10", 6);
            const { fee: fee1, downPayment: downPayment1, principal: principal1, minTargetAmount: minTargetAmount1 } = 
                await getTradeAmounts(leverage1, totalAmountIn1, usdc.address);
            const request1 = await getOpenPositionRequest({
                id: 1n,
                targetCurrency: usdc.address,
                principal: principal1,
                downPayment: downPayment1,
                minTargetAmount: minTargetAmount1,
                expiration: BigInt(await time.latest()) + 86400n,
                fee: fee1
            });
            await usdc.write.mint([user1.account.address, totalAmountIn1], { account: user1.account });
            await usdc.write.approve([wasabiShortPool.address, totalAmountIn1], { account: user1.account });
            const {position: position1} = await sendOpenPositionRequest(request1);
            positions.push(position1);

            // 4x uPPG/USDC short
            const leverage2 = 4n;
            const totalAmountIn2 = parseUnits("50", 6);
            const { fee: fee2, downPayment: downPayment2, principal: principal2, minTargetAmount: minTargetAmount2 } = 
                await getTradeAmounts(leverage2, totalAmountIn2, usdc.address);
            const request2 = await getOpenPositionRequest({
                id: 2n,
                targetCurrency: usdc.address,
                principal: principal2,
                downPayment: downPayment2,
                minTargetAmount: minTargetAmount2,
                expiration: BigInt(await time.latest()) + 86400n,
                fee: fee2
            });
            await usdc.write.mint([user1.account.address, totalAmountIn2], { account: user1.account });
            await usdc.write.approve([wasabiShortPool.address, totalAmountIn2], { account: user1.account });
            const {position: position2} = await sendOpenPositionRequest(request2);
            positions.push(position2);
            
            await time.increase(86400n); // 1 day later

            const vaultAssetsBefore = await vault.read.totalAssets();

            const interests = [];
            let totalInterest = 0n;
            for (let i = 0; i < 2; i++) {
                const interest = await computeMaxInterest(positions[i]);
                interests.push(interest);
                totalInterest += interest;
            }

            // Record Interest
            const functionCallDataList = getApproveAndSwapExactlyOutFunctionCallData(
                mockSwap.address,
                positions[0].collateralCurrency,
                positions[0].currency,
                0n,
                totalInterest
            );
            const hash = await wasabiShortPool.write.recordInterest([positions, interests, functionCallDataList], { account: liquidator.account });
            const timestamp = await time.latest();

            const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed);
            console.log('gas used to record interest for 2 positions', gasUsed);
            
            const events = await wasabiShortPool.getEvents.InterestPaid();
            expect(events).to.have.lengthOf(2);
            for (let i = 0; i < 2; i++) {
                const interestPaidEvent = events[i].args;
                const position = positions[i];
                const interest = interests[i];
                expect(interestPaidEvent.id).to.equal(position.id);
                expect(interestPaidEvent.interestPaid).to.equal(interest);

                position.collateralAmount -= interestPaidEvent.collateralReduced!;
                position.downPayment -= interestPaidEvent.downPaymentReduced!;
                position.lastFundingTimestamp = BigInt(timestamp);
                const hashedPosition = await hasher.read.hashPosition([position]);
                expect(await wasabiShortPool.read.positions([position.id])).to.equal(hashedPosition);
            }

            const vaultAssetsAfter = await vault.read.totalAssets();
            expect(vaultAssetsAfter).to.equal(vaultAssetsBefore + totalInterest);
        });
    });
})
