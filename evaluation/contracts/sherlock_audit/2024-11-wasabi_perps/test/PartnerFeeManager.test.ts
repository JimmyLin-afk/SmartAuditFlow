import {
    time,
    loadFixture,
} from "@nomicfoundation/hardhat-toolbox-viem/network-helpers";
import {encodeFunctionData, zeroAddress, parseEther, getAddress} from "viem";
import { expect } from "chai";
import { Position, formatEthValue, getEventPosition, PayoutType, OpenPositionRequest, FunctionCallData } from "./utils/PerpStructUtils";
import { getApproveAndSwapExactlyOutFunctionCallData, getApproveAndSwapFunctionCallData } from "./utils/SwapUtils";
import { deployLongPoolMockEnvironment, deployShortPoolMockEnvironment } from "./fixtures";
import { signOpenPositionRequest } from "./utils/SigningUtils";

describe("PartnerFeeManager", function () {
    describe("Manual accrual", function () {
        it("Should manually accrue fees", async function () {
            const { partnerFeeManager, partner, owner, weth } = await loadFixture(deployLongPoolMockEnvironment);

            const feeAmount = parseEther("1");
            await weth.write.deposit({value: feeAmount});
            await weth.write.approve([partnerFeeManager.address, feeAmount], {account: owner.account});
            await partnerFeeManager.write.adminAddFees([partner.account.address, weth.address, feeAmount], {account: owner.account});

            expect(await weth.read.balanceOf([partnerFeeManager.address])).to.equal(feeAmount);

            const accruedEvents = await partnerFeeManager.getEvents.FeesAccrued();
            expect(accruedEvents).to.have.lengthOf(1);
            expect(accruedEvents[0].args.partner).to.equal(getAddress(partner.account.address));
            expect(accruedEvents[0].args.feeToken).to.equal(getAddress(weth.address));
            expect(accruedEvents[0].args.amount).to.equal(feeAmount);

            await partnerFeeManager.write.claimFees([[weth.address]], {account: partner.account});

            expect(await weth.read.balanceOf([partnerFeeManager.address])).to.equal(0);
            expect(await weth.read.balanceOf([partner.account.address])).to.equal(feeAmount);

            const claimedEvents = await partnerFeeManager.getEvents.FeesClaimed();
            expect(claimedEvents).to.have.lengthOf(1);
            expect(claimedEvents[0].args.partner).to.equal(getAddress(partner.account.address));
            expect(claimedEvents[0].args.feeToken).to.equal(getAddress(weth.address));
            expect(claimedEvents[0].args.amount).to.equal(feeAmount);
        });

        it("Should claim fees for multiple fee tokens", async function () {
            const { partnerFeeManager, partner, owner, weth, usdc } = await loadFixture(deployLongPoolMockEnvironment);

            const wethFeeAmount = parseEther("1");
            await weth.write.deposit({value: wethFeeAmount});
            await weth.write.approve([partnerFeeManager.address, wethFeeAmount], {account: owner.account});
            await partnerFeeManager.write.adminAddFees([partner.account.address, weth.address, wethFeeAmount], {account: owner.account});

            expect(await weth.read.balanceOf([partnerFeeManager.address])).to.equal(wethFeeAmount);

            const accruedWethEvents = await partnerFeeManager.getEvents.FeesAccrued();
            expect(accruedWethEvents).to.have.lengthOf(1);
            expect(accruedWethEvents[0].args.partner).to.equal(getAddress(partner.account.address));
            expect(accruedWethEvents[0].args.feeToken).to.equal(getAddress(weth.address));
            expect(accruedWethEvents[0].args.amount).to.equal(wethFeeAmount);

            const usdcFeeAmount = parseEther("100");
            await usdc.write.mint([owner.account.address, usdcFeeAmount]);
            await usdc.write.approve([partnerFeeManager.address, usdcFeeAmount], {account: owner.account});
            await partnerFeeManager.write.adminAddFees([partner.account.address, usdc.address, usdcFeeAmount], {account: owner.account});

            expect(await usdc.read.balanceOf([partnerFeeManager.address])).to.equal(usdcFeeAmount);

            const accruedUsdcEvents = await partnerFeeManager.getEvents.FeesAccrued();
            expect(accruedUsdcEvents).to.have.lengthOf(1);
            expect(accruedUsdcEvents[0].args.partner).to.equal(getAddress(partner.account.address));
            expect(accruedUsdcEvents[0].args.feeToken).to.equal(getAddress(usdc.address));
            expect(accruedUsdcEvents[0].args.amount).to.equal(usdcFeeAmount);

            await partnerFeeManager.write.claimFees([[weth.address, usdc.address]], {account: partner.account});

            expect(await weth.read.balanceOf([partnerFeeManager.address])).to.equal(0);
            expect(await weth.read.balanceOf([partner.account.address])).to.equal(wethFeeAmount);
            expect(await usdc.read.balanceOf([partnerFeeManager.address])).to.equal(0);
            expect(await usdc.read.balanceOf([partner.account.address])).to.equal(usdcFeeAmount);

            const claimedEvents = await partnerFeeManager.getEvents.FeesClaimed();
            expect(claimedEvents).to.have.lengthOf(2);
            expect(claimedEvents[0].args.partner).to.equal(getAddress(partner.account.address));
            expect(claimedEvents[0].args.feeToken).to.equal(getAddress(weth.address));
            expect(claimedEvents[0].args.amount).to.equal(wethFeeAmount);
            expect(claimedEvents[1].args.partner).to.equal(getAddress(partner.account.address));
            expect(claimedEvents[1].args.feeToken).to.equal(getAddress(usdc.address));
            expect(claimedEvents[1].args.amount).to.equal(usdcFeeAmount);
        });
    });

    describe("Automatic accrual", function () {
        describe("Open position", function () {
            it("Should automatically accrue fees on opening long position", async function () {
                const { partnerFeeManager, partner, feeReceiver, weth, sendReferredOpenPositionRequest } = await loadFixture(deployLongPoolMockEnvironment);

                const feeReceiverBalanceBefore = await weth.read.balanceOf([feeReceiver]);
                const partnerBalanceBefore = await weth.read.balanceOf([partner.account.address]);

                const {position} = await sendReferredOpenPositionRequest();

                const feeReceiverBalanceAfter = await weth.read.balanceOf([feeReceiver]);

                const accruedEvents = await partnerFeeManager.getEvents.FeesAccrued();
                expect(accruedEvents).to.have.lengthOf(1);
                expect(accruedEvents[0].args.partner).to.equal(getAddress(partner.account.address));
                expect(accruedEvents[0].args.feeToken).to.equal(getAddress(weth.address));
                expect(accruedEvents[0].args.amount).to.equal(position.feesToBePaid / 2n);
                
                expect(feeReceiverBalanceAfter).to.equal(feeReceiverBalanceBefore + position.feesToBePaid / 2n);
                expect(await partnerFeeManager.read.getAccruedFees([partner.account.address, weth.address])).to.equal(position.feesToBePaid / 2n);
                
                await partnerFeeManager.write.claimFees([[weth.address]], {account: partner.account});

                const claimedEvents = await partnerFeeManager.getEvents.FeesClaimed();
                expect(claimedEvents).to.have.lengthOf(1);
                expect(claimedEvents[0].args.partner).to.equal(getAddress(partner.account.address));
                expect(claimedEvents[0].args.feeToken).to.equal(getAddress(weth.address));
                expect(claimedEvents[0].args.amount).to.equal(position.feesToBePaid / 2n);

                const partnerBalanceAfter = await weth.read.balanceOf([partner.account.address]);

                expect(partnerBalanceAfter).to.equal(partnerBalanceBefore + position.feesToBePaid / 2n);
            });

            it("Should automatically accrue fees on opening long position (default fee share)", async function () {
                const { partnerFeeManager, partner, feeReceiver, weth, sendReferredOpenPositionRequest } = await loadFixture(deployLongPoolMockEnvironment);

                const feeReceiverBalanceBefore = await weth.read.balanceOf([feeReceiver]);
                const partnerBalanceBefore = await weth.read.balanceOf([partner.account.address]);

                // Remove partner fee share so default fee share of 20% is used
                await partnerFeeManager.write.setFeeShareBips([partner.account.address, 0n]);

                const {position} = await sendReferredOpenPositionRequest();

                const feeReceiverBalanceAfter = await weth.read.balanceOf([feeReceiver]);

                const accruedEvents = await partnerFeeManager.getEvents.FeesAccrued();
                expect(accruedEvents).to.have.lengthOf(1);
                expect(accruedEvents[0].args.partner).to.equal(getAddress(partner.account.address));
                expect(accruedEvents[0].args.feeToken).to.equal(getAddress(weth.address));
                expect(accruedEvents[0].args.amount).to.equal(position.feesToBePaid / 5n);
                
                expect(feeReceiverBalanceAfter).to.equal(feeReceiverBalanceBefore + position.feesToBePaid * 4n / 5n);
                expect(await partnerFeeManager.read.getAccruedFees([partner.account.address, weth.address])).to.equal(position.feesToBePaid / 5n);
                
                await partnerFeeManager.write.claimFees([[weth.address]], {account: partner.account});

                const claimedEvents = await partnerFeeManager.getEvents.FeesClaimed();
                expect(claimedEvents).to.have.lengthOf(1);
                expect(claimedEvents[0].args.partner).to.equal(getAddress(partner.account.address));
                expect(claimedEvents[0].args.feeToken).to.equal(getAddress(weth.address));
                expect(claimedEvents[0].args.amount).to.equal(position.feesToBePaid / 5n);

                const partnerBalanceAfter = await weth.read.balanceOf([partner.account.address]);

                expect(partnerBalanceAfter).to.equal(partnerBalanceBefore + position.feesToBePaid / 5n);
            });

            it("Should automatically accrue fees on opening short position", async function () {
                const { partnerFeeManager, partner, feeReceiver, weth, sendReferredOpenPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);

                const feeReceiverBalanceBefore = await weth.read.balanceOf([feeReceiver]);
                const partnerBalanceBefore = await weth.read.balanceOf([partner.account.address]);

                const {position} = await sendReferredOpenPositionRequest();

                const feeReceiverBalanceAfter = await weth.read.balanceOf([feeReceiver]);

                const accruedEvents = await partnerFeeManager.getEvents.FeesAccrued();
                expect(accruedEvents).to.have.lengthOf(1);
                expect(accruedEvents[0].args.partner).to.equal(getAddress(partner.account.address));
                expect(accruedEvents[0].args.feeToken).to.equal(getAddress(weth.address));
                expect(accruedEvents[0].args.amount).to.equal(position.feesToBePaid / 2n);

                expect(feeReceiverBalanceAfter).to.equal(feeReceiverBalanceBefore + position.feesToBePaid / 2n);
                expect(await partnerFeeManager.read.getAccruedFees([partner.account.address, weth.address])).to.equal(position.feesToBePaid / 2n);
                
                await partnerFeeManager.write.claimFees([[weth.address]], {account: partner.account});

                const claimedEvents = await partnerFeeManager.getEvents.FeesClaimed();
                expect(claimedEvents).to.have.lengthOf(1);
                expect(claimedEvents[0].args.partner).to.equal(getAddress(partner.account.address));
                expect(claimedEvents[0].args.feeToken).to.equal(getAddress(weth.address));
                expect(claimedEvents[0].args.amount).to.equal(position.feesToBePaid / 2n);

                const partnerBalanceAfter = await weth.read.balanceOf([partner.account.address]);

                expect(partnerBalanceAfter).to.equal(partnerBalanceBefore + position.feesToBePaid / 2n);
            });
        });

        describe("Close position", function () {
            it("Should automatically accrue fees on closing long position", async function () {
                const { partnerFeeManager, wasabiLongPool, user1, partner, feeReceiver, weth, sendReferredOpenPositionRequest, createSignedClosePositionRequest } = await loadFixture(deployLongPoolMockEnvironment);

                const {position} = await sendReferredOpenPositionRequest();
                await partnerFeeManager.write.claimFees([[weth.address]], {account: partner.account});

                await time.increase(86400n); // 1 day later

                const partnerBalanceBefore = await weth.read.balanceOf([partner.account.address]);
                const feeReceiverBalanceBefore = await weth.read.balanceOf([feeReceiver]);
                // Close Position
                const { request, signature } = await createSignedClosePositionRequest({ position, interest: 0n, referrer: partner.account.address });

                await wasabiLongPool.write.closePosition([PayoutType.WRAPPED, request, signature], { account: user1.account });

                const accruedFees = await partnerFeeManager.read.getAccruedFees([partner.account.address, weth.address]);
                const feeReceiverBalanceAfter = await weth.read.balanceOf([feeReceiver]);
                const feeReceiverFees = feeReceiverBalanceAfter - feeReceiverBalanceBefore;

                const closePositionEvents = await wasabiLongPool.getEvents.PositionClosed();
                expect(closePositionEvents).to.have.lengthOf(1);
                expect(closePositionEvents[0].args.id).to.equal(position.id);
                expect(closePositionEvents[0].args.feeAmount).to.equal(accruedFees + feeReceiverFees);

                expect(feeReceiverBalanceAfter).to.equal(feeReceiverBalanceBefore + accruedFees);

                const accruedEvents = await partnerFeeManager.getEvents.FeesAccrued();
                expect(accruedEvents).to.have.lengthOf(1);
                expect(accruedEvents[0].args.partner).to.equal(getAddress(partner.account.address));
                expect(accruedEvents[0].args.feeToken).to.equal(getAddress(weth.address));
                expect(accruedEvents[0].args.amount).to.equal(accruedFees);
                
                await partnerFeeManager.write.claimFees([[weth.address]], {account: partner.account});

                const claimedEvents = await partnerFeeManager.getEvents.FeesClaimed();
                expect(claimedEvents).to.have.lengthOf(1);
                expect(claimedEvents[0].args.partner).to.equal(getAddress(partner.account.address));
                expect(claimedEvents[0].args.feeToken).to.equal(getAddress(weth.address));
                expect(claimedEvents[0].args.amount).to.equal(accruedFees);

                const partnerBalanceAfter = await weth.read.balanceOf([partner.account.address]);

                expect(partnerBalanceAfter).to.equal(partnerBalanceBefore + accruedFees);
            });

            it("Should automatically accrue fees on closing short position", async function () {
                const { partnerFeeManager, wasabiShortPool, user1, partner, feeReceiver, weth, sendReferredOpenPositionRequest, createSignedClosePositionRequest } = await loadFixture(deployShortPoolMockEnvironment);

                const {position} = await sendReferredOpenPositionRequest();
                await partnerFeeManager.write.claimFees([[weth.address]], {account: partner.account});

                await time.increase(86400n); // 1 day later

                const partnerBalanceBefore = await weth.read.balanceOf([partner.account.address]);
                const feeReceiverBalanceBefore = await weth.read.balanceOf([feeReceiver]);

                // Close Position
                const { request, signature } = await createSignedClosePositionRequest({ position, interest: 0n, referrer: partner.account.address });

                await wasabiShortPool.write.closePosition([PayoutType.WRAPPED, request, signature], { account: user1.account });

                const accruedFees = await partnerFeeManager.read.getAccruedFees([partner.account.address, weth.address]);
                const feeReceiverBalanceAfter = await weth.read.balanceOf([feeReceiver]);
                const feeReceiverFees = feeReceiverBalanceAfter - feeReceiverBalanceBefore;

                const closePositionEvents = await wasabiShortPool.getEvents.PositionClosed();
                expect(closePositionEvents).to.have.lengthOf(1);
                expect(closePositionEvents[0].args.id).to.equal(position.id);
                expect(closePositionEvents[0].args.feeAmount).to.equal(accruedFees + feeReceiverFees);

                expect(feeReceiverBalanceAfter).to.equal(feeReceiverBalanceBefore + accruedFees);

                const accruedEvents = await partnerFeeManager.getEvents.FeesAccrued();
                expect(accruedEvents).to.have.lengthOf(1);
                expect(accruedEvents[0].args.partner).to.equal(getAddress(partner.account.address));
                expect(accruedEvents[0].args.feeToken).to.equal(getAddress(weth.address));
                expect(accruedEvents[0].args.amount).to.equal(accruedFees);
                
                await partnerFeeManager.write.claimFees([[weth.address]], {account: partner.account});

                const claimedEvents = await partnerFeeManager.getEvents.FeesClaimed();
                expect(claimedEvents).to.have.lengthOf(1);
                expect(claimedEvents[0].args.partner).to.equal(getAddress(partner.account.address));
                expect(claimedEvents[0].args.feeToken).to.equal(getAddress(weth.address));
                expect(claimedEvents[0].args.amount).to.equal(accruedFees);

                const partnerBalanceAfter = await weth.read.balanceOf([partner.account.address]);

                expect(partnerBalanceAfter).to.equal(partnerBalanceBefore + accruedFees);
            });
        });

        describe("Liquidation", function () {
            it("Should automatically accrue fees on long liquidation", async function () {
                const { partnerFeeManager, wasabiLongPool, mockSwap, partner, feeReceiver, liquidator, weth, uPPG, sendReferredOpenPositionRequest, computeMaxInterest, computeLiquidationPrice } = await loadFixture(deployLongPoolMockEnvironment);

                const {position} = await sendReferredOpenPositionRequest();
                await partnerFeeManager.write.claimFees([[weth.address]], {account: partner.account});
                
                await time.increase(86400n); // 1 day later

                const partnerBalanceBefore = await weth.read.balanceOf([partner.account.address]);
                const feeReceiverBalanceBefore = await weth.read.balanceOf([feeReceiver]);

                // Liquidate Position
                const functionCallDataList = getApproveAndSwapFunctionCallData(mockSwap.address, uPPG.address, weth.address, position.collateralAmount);

                const interest = await computeMaxInterest(position);
                const liquidationPrice = await computeLiquidationPrice(position);

                await mockSwap.write.setPrice([uPPG.address, weth.address, liquidationPrice]);

                await wasabiLongPool.write.liquidatePosition([PayoutType.WRAPPED, interest, position, functionCallDataList, partner.account.address], { account: liquidator.account });

                const accruedFees = await partnerFeeManager.read.getAccruedFees([partner.account.address, weth.address]);
                const feeReceiverBalanceAfter = await weth.read.balanceOf([feeReceiver]);
                const feeReceiverFees = feeReceiverBalanceAfter - feeReceiverBalanceBefore;
                
                const liquidationEvents = await wasabiLongPool.getEvents.PositionLiquidated();
                expect(liquidationEvents).to.have.lengthOf(1);
                expect(liquidationEvents[0].args.id).to.equal(position.id);
                expect(liquidationEvents[0].args.feeAmount).to.equal(accruedFees + feeReceiverFees);

                expect(feeReceiverBalanceAfter).to.equal(feeReceiverBalanceBefore + accruedFees);
                
                const accruedEvents = await partnerFeeManager.getEvents.FeesAccrued();
                expect(accruedEvents).to.have.lengthOf(1);
                expect(accruedEvents[0].args.partner).to.equal(getAddress(partner.account.address));
                expect(accruedEvents[0].args.feeToken).to.equal(getAddress(weth.address));
                expect(accruedEvents[0].args.amount).to.equal(accruedFees);
                
                await partnerFeeManager.write.claimFees([[weth.address]], {account: partner.account});

                const claimedEvents = await partnerFeeManager.getEvents.FeesClaimed();
                expect(claimedEvents).to.have.lengthOf(1);
                expect(claimedEvents[0].args.partner).to.equal(getAddress(partner.account.address));
                expect(claimedEvents[0].args.feeToken).to.equal(getAddress(weth.address));
                expect(claimedEvents[0].args.amount).to.equal(accruedFees);

                const partnerBalanceAfter = await weth.read.balanceOf([partner.account.address]);

                expect(partnerBalanceAfter).to.equal(partnerBalanceBefore + accruedFees);
            });

            it("Should automatically accrue fees on short liquidation", async function () {
                const { partnerFeeManager, wasabiShortPool, mockSwap, partner, feeReceiver, liquidator, weth, uPPG, sendReferredOpenPositionRequest, computeMaxInterest, computeLiquidationPrice } = await loadFixture(deployShortPoolMockEnvironment);

                const {position} = await sendReferredOpenPositionRequest();
                await partnerFeeManager.write.claimFees([[weth.address]], {account: partner.account});

                await time.increase(86400n); // 1 day later

                const partnerBalanceBefore = await weth.read.balanceOf([partner.account.address]);
                const feeReceiverBalanceBefore = await weth.read.balanceOf([feeReceiver]);

                // Liquidate Position
                const interest = await computeMaxInterest(position);
                const functionCallDataList = getApproveAndSwapExactlyOutFunctionCallData(
                    mockSwap.address,
                    position.collateralCurrency,
                    position.currency,
                    position.collateralAmount,
                    position.principal + interest);
                const liquidationPrice = await computeLiquidationPrice(position);

                await mockSwap.write.setPrice([uPPG.address, weth.address, liquidationPrice]);

                await wasabiShortPool.write.liquidatePosition([PayoutType.WRAPPED, interest, position, functionCallDataList, partner.account.address], { account: liquidator.account });
                
                const accruedFees = await partnerFeeManager.read.getAccruedFees([partner.account.address, weth.address]);
                const feeReceiverBalanceAfter = await weth.read.balanceOf([feeReceiver]);
                const feeReceiverFees = feeReceiverBalanceAfter - feeReceiverBalanceBefore;
                
                const liquidationEvents = await wasabiShortPool.getEvents.PositionLiquidated();
                expect(liquidationEvents).to.have.lengthOf(1);
                expect(liquidationEvents[0].args.id).to.equal(position.id);
                expect(liquidationEvents[0].args.feeAmount).to.equal(accruedFees + feeReceiverFees);

                expect(feeReceiverBalanceAfter).to.equal(feeReceiverBalanceBefore + accruedFees);
                
                const accruedEvents = await partnerFeeManager.getEvents.FeesAccrued();
                expect(accruedEvents).to.have.lengthOf(1);
                expect(accruedEvents[0].args.partner).to.equal(getAddress(partner.account.address));
                expect(accruedEvents[0].args.feeToken).to.equal(getAddress(weth.address));
                expect(accruedEvents[0].args.amount).to.equal(accruedFees);
                
                await partnerFeeManager.write.claimFees([[weth.address]], {account: partner.account});

                const claimedEvents = await partnerFeeManager.getEvents.FeesClaimed();
                expect(claimedEvents).to.have.lengthOf(1);
                expect(claimedEvents[0].args.partner).to.equal(getAddress(partner.account.address));
                expect(claimedEvents[0].args.feeToken).to.equal(getAddress(weth.address));
                expect(claimedEvents[0].args.amount).to.equal(accruedFees);
                
                const partnerBalanceAfter = await weth.read.balanceOf([partner.account.address]);

                expect(partnerBalanceAfter).to.equal(partnerBalanceBefore + accruedFees);
            });
        });

        describe("Increase position", function () {
            it("Should automatically accrue fees on adding to a long position", async function () {
                const { partnerFeeManager, wasabiLongPool, contractName, partner, mockSwap, user1, feeShareBips, orderSigner, weth, uPPG, totalSize, totalAmountIn, initialPrice, priceDenominator, sendReferredOpenPositionRequest } = await loadFixture(deployLongPoolMockEnvironment);

                const {position} = await sendReferredOpenPositionRequest();
                await partnerFeeManager.write.claimFees([[weth.address]], {account: partner.account});
                
                await time.increase(86400n); // 1 day later

                const functionCallDataList: FunctionCallData[] =
                        getApproveAndSwapFunctionCallData(mockSwap.address, weth.address, uPPG.address, totalSize);
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
                    referrer: partner.account.address
                };
                const signature = await signOpenPositionRequest(orderSigner, contractName, wasabiLongPool.address, openPositionRequest);

                // Increase Position
                await wasabiLongPool.write.openPosition([openPositionRequest, signature], { value: totalAmountIn, account: user1.account });

                const increasedEvents = await wasabiLongPool.getEvents.PositionIncreased();
                expect(increasedEvents).to.have.lengthOf(1);
                const increasedEventData = increasedEvents[0].args;
                const accruedEvents = await partnerFeeManager.getEvents.FeesAccrued();
                expect(accruedEvents).to.have.lengthOf(1);
                expect(accruedEvents[0].args.partner).to.equal(getAddress(partner.account.address));
                expect(accruedEvents[0].args.feeToken).to.equal(getAddress(weth.address));
                expect(accruedEvents[0].args.amount).to.equal(increasedEventData.feesAdded! * feeShareBips / 10000n);
            });

            it("Should automatically accrue fees on adding to a short position", async function () {
                const { partnerFeeManager, wasabiShortPool, contractName, partner, mockSwap, user1, feeShareBips, orderSigner, weth, uPPG, principal, totalAmountIn, initialPPGPrice, priceDenominator, sendReferredOpenPositionRequest } = await loadFixture(deployShortPoolMockEnvironment);

                const {position} = await sendReferredOpenPositionRequest();
                await partnerFeeManager.write.claimFees([[weth.address]], {account: partner.account});
                
                await time.increase(86400n); // 1 day later

                const functionCallDataList: FunctionCallData[] =
                    getApproveAndSwapFunctionCallData(mockSwap.address, uPPG.address, weth.address, principal);
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
                    referrer: partner.account.address
                };
                const signature = await signOpenPositionRequest(orderSigner, contractName, wasabiShortPool.address, openPositionRequest);

                // Increase Position
                await wasabiShortPool.write.openPosition([openPositionRequest, signature], { value: totalAmountIn, account: user1.account });

                const increasedEvents = await wasabiShortPool.getEvents.PositionIncreased();
                expect(increasedEvents).to.have.lengthOf(1);
                const increasedEventData = increasedEvents[0].args;
                const accruedEvents = await partnerFeeManager.getEvents.FeesAccrued();
                expect(accruedEvents).to.have.lengthOf(1);
                expect(accruedEvents[0].args.partner).to.equal(getAddress(partner.account.address));
                expect(accruedEvents[0].args.feeToken).to.equal(getAddress(weth.address));
                expect(accruedEvents[0].args.amount).to.equal(increasedEventData.feesAdded! * feeShareBips / 10000n);
            });
        });

        describe("Decrease position", function () {
            it("Should automatically accrue fees on decreasing a long position", async function () {
                const { partnerFeeManager, wasabiLongPool, user1, partner, feeReceiver, weth, sendReferredOpenPositionRequest, createSignedClosePositionRequest, computeMaxInterest } = await loadFixture(deployLongPoolMockEnvironment);

                const {position} = await sendReferredOpenPositionRequest();
                await partnerFeeManager.write.claimFees([[weth.address]], {account: partner.account});

                await time.increase(86400n); // 1 day later

                const feeReceiverBalanceBefore = await weth.read.balanceOf([feeReceiver]);

                // Close Half of the Position
                const closeAmountDenominator = 2n;
                const interest = await computeMaxInterest(position) / closeAmountDenominator;
                const closeAmount = position.collateralAmount / closeAmountDenominator;
                const { request, signature } = await createSignedClosePositionRequest({ position, interest, referrer: partner.account.address, amount: closeAmount });

                await wasabiLongPool.write.closePosition([PayoutType.WRAPPED, request, signature], { account: user1.account });

                const accruedFees = await partnerFeeManager.read.getAccruedFees([partner.account.address, weth.address]);
                const feeReceiverBalanceAfter = await weth.read.balanceOf([feeReceiver]);
                const feeReceiverFees = feeReceiverBalanceAfter - feeReceiverBalanceBefore;

                const decreasedEvents = await wasabiLongPool.getEvents.PositionDecreased();
                expect(decreasedEvents).to.have.lengthOf(1);
                expect(decreasedEvents[0].args.id).to.equal(position.id);
                expect(decreasedEvents[0].args.closeFee).to.equal(accruedFees + feeReceiverFees);

                expect(feeReceiverBalanceAfter).to.equal(feeReceiverBalanceBefore + accruedFees);

                const accruedEvents = await partnerFeeManager.getEvents.FeesAccrued();
                expect(accruedEvents).to.have.lengthOf(1);
                expect(accruedEvents[0].args.partner).to.equal(getAddress(partner.account.address));
                expect(accruedEvents[0].args.feeToken).to.equal(getAddress(weth.address));
                expect(accruedEvents[0].args.amount).to.equal(accruedFees);
            });

            it("Should automatically accrue fees on decreasing a short position", async function () {
                const { partnerFeeManager, wasabiShortPool, user1, partner, feeReceiver, weth, sendReferredOpenPositionRequest, createSignedClosePositionRequest, computeMaxInterest } = await loadFixture(deployShortPoolMockEnvironment);

                const {position} = await sendReferredOpenPositionRequest();
                await partnerFeeManager.write.claimFees([[weth.address]], {account: partner.account});

                await time.increase(86400n); // 1 day later

                const feeReceiverBalanceBefore = await weth.read.balanceOf([feeReceiver]);

                // Close Half of the Position
                const closeAmountDenominator = 2n;
                const interest = await computeMaxInterest(position) / closeAmountDenominator;
                const closeAmount = position.principal / closeAmountDenominator;
                const { request, signature } = await createSignedClosePositionRequest({ position, interest, referrer: partner.account.address, amount: closeAmount });

                await wasabiShortPool.write.closePosition([PayoutType.WRAPPED, request, signature], { account: user1.account });
                
                const accruedFees = await partnerFeeManager.read.getAccruedFees([partner.account.address, weth.address]);
                const feeReceiverBalanceAfter = await weth.read.balanceOf([feeReceiver]);
                const feeReceiverFees = feeReceiverBalanceAfter - feeReceiverBalanceBefore;

                const decreasedEvents = await wasabiShortPool.getEvents.PositionDecreased();
                expect(decreasedEvents).to.have.lengthOf(1);
                expect(decreasedEvents[0].args.id).to.equal(position.id);
                expect(decreasedEvents[0].args.closeFee).to.equal(accruedFees + feeReceiverFees);

                expect(feeReceiverBalanceAfter).to.equal(feeReceiverBalanceBefore + accruedFees);

                const accruedEvents = await partnerFeeManager.getEvents.FeesAccrued();
                expect(accruedEvents).to.have.lengthOf(1);
                expect(accruedEvents[0].args.partner).to.equal(getAddress(partner.account.address));
                expect(accruedEvents[0].args.feeToken).to.equal(getAddress(weth.address));
                expect(accruedEvents[0].args.amount).to.equal(accruedFees);
            });
        });
    });

    describe("Validations", function () {
        it("Should revert if the accrueFees caller is not a pool", async function () {
            const { partnerFeeManager, user1, owner, weth, tradeFeeValue } = await loadFixture(deployLongPoolMockEnvironment);

            await expect(partnerFeeManager.write.accrueFees(
                [user1.account.address, weth.address, tradeFeeValue], 
                {account: owner.account}
            )).to.be.rejectedWith("CallerNotPool");
        });
    });
});