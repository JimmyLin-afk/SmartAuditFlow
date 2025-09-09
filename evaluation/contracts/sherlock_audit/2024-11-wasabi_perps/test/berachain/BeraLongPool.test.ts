import {
    time,
    loadFixture,
} from "@nomicfoundation/hardhat-toolbox-viem/network-helpers";
import hre from "hardhat";
import { expect } from "chai";
import { getAddress, zeroAddress, parseEther } from "viem";
import { deployLongPoolMockEnvironment } from "./berachainFixtures";
import { getBalance, takeBalanceSnapshot } from "../utils/StateUtils";
import { AddCollateralRequest, FunctionCallData, OpenPositionRequest, PayoutType } from "../utils/PerpStructUtils";
import { getApproveAndSwapFunctionCallData } from "../utils/SwapUtils";
import { signAddCollateralRequest, signOpenPositionRequest } from "../utils/SigningUtils";

describe("BeraLongPool", function () {
    describe("Deployment", function () {
        it("Should deploy StakingAccountFactory correctly", async function () {
            const { addressProvider, stakingAccountFactory, beacon, ibgt, wbera, ibgtInfraredVault, mockInfrared } = await loadFixture(deployLongPoolMockEnvironment);

            expect(await addressProvider.read.getStakingAccountFactory()).to.equal(stakingAccountFactory.address);
            expect(await stakingAccountFactory.read.beacon()).to.equal(beacon.address);
            const stakingContract = await stakingAccountFactory.read.tokenToStakingContract([ibgt.address]);
            expect(stakingContract[0]).to.equal(ibgtInfraredVault.address);
            expect(stakingContract[1]).to.equal(0);
            const [ wberaRewardsDistributor, wberaRewardsDuration, wberaPeriodFinish, wberaRewardRate, wberaLastUpdateTime, wberaRewardPerTokenStored ] 
                = await ibgtInfraredVault.read.rewardData([wbera.address]);
            expect(wberaRewardsDistributor).to.equal(mockInfrared.address);
            expect(wberaRewardsDuration).to.equal(864000n);
            expect(wberaPeriodFinish).to.be.approximately(wberaLastUpdateTime + 864000n, 1n);
            expect(wberaRewardRate).to.equal(parseEther("100") / 864000n);
            expect(wberaRewardPerTokenStored).to.equal(0n);
            const [ ibgtRewardsDistributor, ibgtRewardsDuration, ibgtPeriodFinish, ibgtRewardRate, ibgtLastUpdateTime, ibgtRewardPerTokenStored ] 
                = await ibgtInfraredVault.read.rewardData([ibgt.address]);
            expect(ibgtRewardsDistributor).to.equal(mockInfrared.address);
            expect(ibgtRewardsDuration).to.equal(864000n);
            expect(ibgtPeriodFinish).to.be.approximately(ibgtLastUpdateTime + 864000n, 1n);
            expect(ibgtRewardRate).to.equal(parseEther("100") / 864000n);
            expect(ibgtRewardPerTokenStored).to.equal(0n);
        });

        it("Should deploy StakingAccount correctly", async function () {
            const { user1, stakingAccountFactory, sendStakingOpenPositionRequest } = await loadFixture(deployLongPoolMockEnvironment);

            expect(await stakingAccountFactory.read.userToStakingAccount([user1.account.address])).to.equal(zeroAddress);

            // Test deploying a staking account while opening a position
            await sendStakingOpenPositionRequest();

            const user1StakingAccountAddress = await stakingAccountFactory.read.userToStakingAccount([user1.account.address]);
            expect(user1StakingAccountAddress).to.not.equal(zeroAddress);
        });

        it("Should upgrade beacon", async function () {
            const { stakingAccountFactory, user1, user2, sendStakingOpenPositionRequest } = await loadFixture(deployLongPoolMockEnvironment);

            await sendStakingOpenPositionRequest(1n, user1.account);
            await sendStakingOpenPositionRequest(2n, user2.account);

            const user1StakingAccountAddress = await stakingAccountFactory.read.userToStakingAccount([user1.account.address]);
            const user2StakingAccountAddress = await stakingAccountFactory.read.userToStakingAccount([user2.account.address]);
            
            // Upgrade beacon
            const MockStakingAccountV2 = await hre.ethers.getContractFactory("MockStakingAccountV2");
            const mockStakingAccountV2 = await MockStakingAccountV2.deploy().then(c => c.waitForDeployment()).then(c => c.getAddress());
            await stakingAccountFactory.write.upgradeBeacon([getAddress(mockStakingAccountV2)]);

            // Check that the staking accounts are upgraded
            const user1StakingAccount = await hre.viem.getContractAt("MockStakingAccountV2", user1StakingAccountAddress);
            const user2StakingAccount = await hre.viem.getContractAt("MockStakingAccountV2", user2StakingAccountAddress);

            expect(await user1StakingAccount.read.MAGIC_VALUE()).to.equal(1337n);
            expect(await user2StakingAccount.read.MAGIC_VALUE()).to.equal(1337n);
        })
    });

    describe("Open Position w/ Leveraged Staking", function () {
        it("Should stake a position", async function () {
            const { user1, wasabiLongPool, stakingAccountFactory, sendStakingOpenPositionRequest, openPositionRequest, totalAmountIn, ibgt, ibgtInfraredVault, ibgtRewardVault } = await loadFixture(deployLongPoolMockEnvironment);

            const { position } = await sendStakingOpenPositionRequest();

            const positionOpenedEvents = await wasabiLongPool.getEvents.PositionOpened();
            expect(positionOpenedEvents).to.have.lengthOf(1);
            const positionOpenedEvent = positionOpenedEvents[0].args;

            expect(positionOpenedEvent.positionId).to.equal(openPositionRequest.id);
            expect(positionOpenedEvent.downPayment).to.equal(totalAmountIn - positionOpenedEvent.feesToBePaid!);
            expect(positionOpenedEvent.principal).to.equal(openPositionRequest.principal);
            expect(positionOpenedEvent.collateralAmount).to.equal(await ibgt.read.balanceOf([ibgtRewardVault.address]));
            expect(positionOpenedEvent.collateralAmount).to.greaterThanOrEqual(openPositionRequest.minTargetAmount);

            const user1StakingAccountAddress = await stakingAccountFactory.read.userToStakingAccount([user1.account.address]);
            expect(user1StakingAccountAddress).to.not.equal(zeroAddress);

            const stakedEvents = await ibgtInfraredVault.getEvents.Staked();
            expect(stakedEvents).to.have.lengthOf(1);
            const stakedEvent = stakedEvents[0].args;

            expect(stakedEvent.amount).to.equal(positionOpenedEvent.collateralAmount);
            expect(stakedEvent.user).to.equal(user1StakingAccountAddress);

            const positionStakedEvents = await stakingAccountFactory.getEvents.StakedPosition();
            expect(positionStakedEvents).to.have.lengthOf(1);
            const positionStakedEvent = positionStakedEvents[0].args;

            expect(positionStakedEvent.user).to.equal(getAddress(user1.account.address));
            expect(positionStakedEvent.stakingAccount).to.equal(user1StakingAccountAddress);
            expect(positionStakedEvent.stakingContract).to.equal(ibgtInfraredVault.address);
            expect(positionStakedEvent.stakingType).to.equal(0);
            expect(positionStakedEvent.positionId).to.equal(position.id);
            expect(positionStakedEvent.collateralAmount).to.equal(positionOpenedEvent.collateralAmount);

            expect(await wasabiLongPool.read.isPositionStaked([position.id])).to.equal(true);
            expect(await ibgt.read.balanceOf([wasabiLongPool.address])).to.equal(0n);
        });

        it("Should stake a position after opening", async function () {
            const { user1, wasabiLongPool, stakingAccountFactory, sendDefaultOpenPositionRequest, openPositionRequest, totalAmountIn, ibgt, ibgtInfraredVault, ibgtRewardVault } = await loadFixture(deployLongPoolMockEnvironment);

            const { position } = await sendDefaultOpenPositionRequest();

            const positionOpenedEvents = await wasabiLongPool.getEvents.PositionOpened();
            expect(positionOpenedEvents).to.have.lengthOf(1);
            const positionOpenedEvent = positionOpenedEvents[0].args;

            expect(positionOpenedEvent.positionId).to.equal(openPositionRequest.id);
            expect(positionOpenedEvent.downPayment).to.equal(totalAmountIn - positionOpenedEvent.feesToBePaid!);
            expect(positionOpenedEvent.principal).to.equal(openPositionRequest.principal);
            expect(positionOpenedEvent.collateralAmount).to.equal(await ibgt.read.balanceOf([wasabiLongPool.address]));
            expect(positionOpenedEvent.collateralAmount).to.greaterThanOrEqual(openPositionRequest.minTargetAmount);
            
            await wasabiLongPool.write.stakePosition([position], { account: user1.account });

            const user1StakingAccountAddress = await stakingAccountFactory.read.userToStakingAccount([user1.account.address]);
            expect(user1StakingAccountAddress).to.not.equal(zeroAddress);

            const stakedEvents = await ibgtInfraredVault.getEvents.Staked();
            expect(stakedEvents).to.have.lengthOf(1);
            const stakedEvent = stakedEvents[0].args;

            expect(stakedEvent.amount).to.equal(positionOpenedEvent.collateralAmount);
            expect(stakedEvent.user).to.equal(user1StakingAccountAddress);

            const positionStakedEvents = await stakingAccountFactory.getEvents.StakedPosition();
            expect(positionStakedEvents).to.have.lengthOf(1);
            const positionStakedEvent = positionStakedEvents[0].args;

            expect(positionStakedEvent.user).to.equal(getAddress(user1.account.address));
            expect(positionStakedEvent.stakingAccount).to.equal(user1StakingAccountAddress);
            expect(positionStakedEvent.stakingContract).to.equal(ibgtInfraredVault.address);
            expect(positionStakedEvent.stakingType).to.equal(0);
            expect(positionStakedEvent.positionId).to.equal(position.id);
            expect(positionStakedEvent.collateralAmount).to.equal(positionOpenedEvent.collateralAmount);

            expect(await wasabiLongPool.read.isPositionStaked([position.id])).to.equal(true);
            expect(await ibgt.read.balanceOf([wasabiLongPool.address])).to.equal(0n);
            expect(await ibgt.read.balanceOf([ibgtRewardVault.address])).to.equal(position.collateralAmount);
        });

        describe("Open and Edit Position", function () {
            it("Should open and increase a staked position", async function () {
                const { wasabiLongPool, mockSwap, wbera, ibgt, ibgtRewardVault, user1, totalAmountIn, totalSize, initialPrice, priceDenominator, orderSigner, sendStakingOpenPositionRequest } = await loadFixture(deployLongPoolMockEnvironment);

                // Open Position
                const {position} = await sendStakingOpenPositionRequest();

                await time.increase(86400n); // 1 day later

                const functionCallDataList: FunctionCallData[] =
                        getApproveAndSwapFunctionCallData(mockSwap.address, wbera.address, ibgt.address, totalSize);
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
                const signature = await signOpenPositionRequest(orderSigner, "WasabiLongPool", wasabiLongPool.address, openPositionRequest);

                // Increase Position
                await wasabiLongPool.write.openPositionAndStake([openPositionRequest, signature], { value: totalAmountIn, account: user1.account });

                const events = await wasabiLongPool.getEvents.PositionIncreased();
                expect(events).to.have.lengthOf(1);
                const eventData = events[0].args;
                expect(eventData.id).to.equal(position.id);
                expect(eventData.downPaymentAdded).to.equal(totalAmountIn - eventData.feesAdded!);
                expect(eventData.principalAdded).to.equal(openPositionRequest.principal);
                expect(eventData.collateralAdded! + position.collateralAmount).to.equal(await ibgt.read.balanceOf([ibgtRewardVault.address]));
                expect(eventData.collateralAdded).to.greaterThanOrEqual(openPositionRequest.minTargetAmount);
            });

            it("Should open and add collateral to a staked position", async function () {
                const { wasabiLongPool, vault, user1, downPayment, orderSigner, sendStakingOpenPositionRequest, computeMaxInterest } = await loadFixture(deployLongPoolMockEnvironment);

                // Open Position
                const {position} = await sendStakingOpenPositionRequest();
                const totalAssetValueBefore = await vault.read.totalAssetValue();

                await time.increase(86400n); // 1 day later

                const interest = await computeMaxInterest(position);
                const request: AddCollateralRequest = {
                    amount: downPayment,
                    interest,
                    expiration: BigInt(await time.latest()) + 86400n,
                    position
                }
                const signature = await signAddCollateralRequest(orderSigner, "WasabiLongPool", wasabiLongPool.address, request);

                // Add Collateral
                await wasabiLongPool.write.addCollateral([request, signature], { value: position.downPayment, account: user1.account });

                const events = await wasabiLongPool.getEvents.CollateralAdded();
                expect(events).to.have.lengthOf(1);
                const eventData = events[0].args;
                expect(eventData.id).to.equal(position.id);
                expect(eventData.downPaymentAdded).to.equal(downPayment - interest);
                expect(eventData.principalReduced).to.equal(downPayment - interest);
                expect(eventData.collateralAdded).to.equal(0n);
                const totalAssetValueAfter = await vault.read.totalAssetValue();
                expect(totalAssetValueAfter).to.equal(totalAssetValueBefore + interest);
            });
        });
    });

    describe("Close Staked Position", function () {
        it("Price Not Changed", async function () {
            const { user1, wasabiLongPool, sendStakingOpenPositionRequest, createSignedClosePositionRequest, computeMaxInterest, wbera, ibgt, ibgtRewardVault, vault, feeReceiver, publicClient } = await loadFixture(deployLongPoolMockEnvironment);

            const { position } = await sendStakingOpenPositionRequest();
            
            await time.increase(86400n); // 1 day later

            const { request, signature } = await createSignedClosePositionRequest({ position, interest: 0n });

            const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceBefore = await getBalance(publicClient, wbera.address, vault.address);
            const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });

            const maxInterest = await computeMaxInterest(position);
            
            const hash = await wasabiLongPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account });

            const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceAfter = await getBalance(publicClient, wbera.address, vault.address);
            const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });

            // Checks
            const closePositionEvents = await wasabiLongPool.getEvents.PositionClosed();
            expect(closePositionEvents).to.have.lengthOf(1);
            const closePositionEvent = closePositionEvents[0].args;
            const totalFeesPaid = closePositionEvent.feeAmount!;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal);
            expect(closePositionEvent.interestPaid!).to.equal(maxInterest, "If given interest value is 0, should use max interest");
            expect(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!).to.equal(vaultBalanceAfter);

            const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.feeAmount! - position.downPayment;
            expect(totalReturn).to.equal(0, "Total return should be 0 on no price change");

            expect(await wasabiLongPool.read.isPositionStaked([position.id])).to.equal(false, "Position should not be staked");
            expect(await ibgt.read.balanceOf([wasabiLongPool.address])).to.equal(0n, "Pool should not have any collateral left");
            expect(await ibgt.read.balanceOf([ibgtRewardVault.address])).to.equal(0n, "Reward vault should not have any staked collateral left");

            // Check trader has been paid
            const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
            expect(traderBalanceAfter - traderBalanceBefore + gasUsed).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        it("Price Increased", async function () {
            const { sendStakingOpenPositionRequest, createSignedClosePositionRequest, owner, publicClient, wasabiLongPool, user1, ibgt, ibgtRewardVault, mockSwap, feeReceiver, initialPrice, wbera, vault } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendStakingOpenPositionRequest();

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([ibgt.address, wbera.address, initialPrice * 2n]); // Price doubled

            // Close Position
            const { request, signature } = await createSignedClosePositionRequest({position});

            const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceBefore = await getBalance(publicClient, wbera.address, vault.address);
            const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });

            const hash = await wasabiLongPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account });

            const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceAfter = await getBalance(publicClient, wbera.address, vault.address);
            const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });

            // Checks
            const events = await wasabiLongPool.getEvents.PositionClosed();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;
            const totalFeesPaid = closePositionEvent.feeAmount!;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal);
            expect(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!).to.equal(vaultBalanceAfter);

            const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.feeAmount!;
            expect(totalReturn).to.equal(position.downPayment * 5n, "on 2x price increase, total return should be 4x down payment");

            expect(await ibgt.read.balanceOf([wasabiLongPool.address])).to.equal(0n, "Pool should not have any collateral left");
            expect(await ibgt.read.balanceOf([ibgtRewardVault.address])).to.equal(0n, "Reward vault should not have any staked collateral left");

            // Check trader has been paid
            const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
            expect(traderBalanceAfter - traderBalanceBefore + gasUsed).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        it("Price Decreased", async function () {
            const { sendStakingOpenPositionRequest, createSignedClosePositionRequest, publicClient, wasabiLongPool, user1, ibgt, ibgtRewardVault, mockSwap, feeReceiver, initialPrice, wbera, vault } = await loadFixture(deployLongPoolMockEnvironment);

            // Open Position
            const {position} = await sendStakingOpenPositionRequest();

            await time.increase(86400n); // 1 day later
            await mockSwap.write.setPrice([ibgt.address, wbera.address, initialPrice * 8n / 10n]); // Price fell 20%

            // Close Position
            const { request, signature } = await createSignedClosePositionRequest({position});

            const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceBefore = await getBalance(publicClient, wbera.address, vault.address);
            const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });

            const hash = await wasabiLongPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account });

            const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
            const vaultBalanceAfter = await getBalance(publicClient, wbera.address, vault.address);
            const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });

            // Checks
            const events = await wasabiLongPool.getEvents.PositionClosed();
            expect(events).to.have.lengthOf(1);
            const closePositionEvent = events[0].args;
            const totalFeesPaid = closePositionEvent.feeAmount!;

            expect(closePositionEvent.id).to.equal(position.id);
            expect(closePositionEvent.principalRepaid!).to.equal(position.principal);
            expect(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!).to.equal(vaultBalanceAfter);

            const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.feeAmount! - position.downPayment;
            expect(totalReturn).to.equal(position.downPayment / -5n * 4n, "on 20% price decrease, total return should be -20% * leverage (4) * down payment");

            expect(await ibgt.read.balanceOf([wasabiLongPool.address])).to.equal(0n, "Pool should not have any collateral left");
            expect(await ibgt.read.balanceOf([ibgtRewardVault.address])).to.equal(0n, "Reward vault should not have any staked collateral left");

            // Check trader has been paid
            const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
            expect(traderBalanceAfter - traderBalanceBefore + gasUsed).to.equal(closePositionEvent.payout!);

            // Check fees have been paid
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
        });

        describe("Partial Close", function () {
            it("Price not changed", async function () {
                const { sendStakingOpenPositionRequest, createSignedClosePositionRequest, computeMaxInterest, publicClient, wasabiLongPool, user1, ibgt, ibgtRewardVault, feeReceiver, wbera, vault } = await loadFixture(deployLongPoolMockEnvironment);

                // Open Position
                const {position} = await sendStakingOpenPositionRequest();

                await time.increase(86400n); // 1 day later

                // Close Half of the Position
                const closeAmountDenominator = 2n;
                const interest = await computeMaxInterest(position) / closeAmountDenominator;
                const { request, signature } = await createSignedClosePositionRequest({ position, interest, amount: position.collateralAmount / closeAmountDenominator });

                const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
                const vaultBalanceBefore = await getBalance(publicClient, wbera.address, vault.address);
                const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });
                
                const hash = await wasabiLongPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account });

                const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
                const vaultBalanceAfter = await getBalance(publicClient, wbera.address, vault.address);
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

                expect(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!).to.equal(vaultBalanceAfter);

                const downPaymentReduced = position.downPayment / closeAmountDenominator;
                const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.closeFee! - downPaymentReduced;
                expect(totalReturn).to.equal(0, "Total return should be 0 on no price change");

                expect(await ibgt.read.balanceOf([ibgtRewardVault.address])).to.equal(
                    position.collateralAmount / closeAmountDenominator, 
                    "Pool should have half of the collateral left"
                );
                expect(await ibgt.read.balanceOf([wasabiLongPool.address])).to.equal(0n, "Pool should not hold any collateral");
                expect(await wasabiLongPool.read.isPositionStaked([position.id])).to.equal(true, "Position should still be staked");

                // Check trader has been paid
                const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
                expect(traderBalanceAfter - traderBalanceBefore + gasUsed).to.equal(closePositionEvent.payout!);

                // Check fees have been paid
                expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
            });

            it("Price Increased", async function () {
                const { sendStakingOpenPositionRequest, createSignedClosePositionRequest, computeMaxInterest, owner, publicClient, wasabiLongPool, user1, ibgt, ibgtRewardVault, mockSwap, feeReceiver, initialPrice, wbera, vault } = await loadFixture(deployLongPoolMockEnvironment);

                // Open Position
                const {position} = await sendStakingOpenPositionRequest();

                await time.increase(86400n); // 1 day later
                await mockSwap.write.setPrice([ibgt.address, wbera.address, initialPrice * 2n]); // Price doubled

                // Close Half of the Position
                const closeAmountDenominator = 2n;
                const interest = await computeMaxInterest(position) / closeAmountDenominator;
                const { request, signature } = await createSignedClosePositionRequest({position, interest, amount: position.collateralAmount / closeAmountDenominator});

                const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
                const vaultBalanceBefore = await getBalance(publicClient, wbera.address, vault.address);
                const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });

                const hash = await wasabiLongPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account });

                const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
                const vaultBalanceAfter = await getBalance(publicClient, wbera.address, vault.address);
                const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });

                // Checks
                const events = await wasabiLongPool.getEvents.PositionDecreased();
                expect(events).to.have.lengthOf(1);
                const closePositionEvent = events[0].args;
                const totalFeesPaid = closePositionEvent.closeFee!;

                expect(closePositionEvent.id).to.equal(position.id);
                expect(closePositionEvent.principalRepaid!).to.equal(position.principal / closeAmountDenominator, "Half of the principal should be repaid");

                expect(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!).to.equal(vaultBalanceAfter);

                const downPaymentReduced = position.downPayment / closeAmountDenominator;
                const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.closeFee! - downPaymentReduced;
                expect(totalReturn).to.equal(downPaymentReduced * 4n, "on 2x price increase, total return should be 4x adjusted down payment");

                expect(await ibgt.read.balanceOf([ibgtRewardVault.address])).to.equal(position.collateralAmount / closeAmountDenominator, "Pool should have half of the collateral left");
                expect(await ibgt.read.balanceOf([wasabiLongPool.address])).to.equal(0n, "Pool should not hold any collateral");
                expect(await wasabiLongPool.read.isPositionStaked([position.id])).to.equal(true, "Position should still be staked");

                // Check trader has been paid
                const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
                expect(traderBalanceAfter - traderBalanceBefore + gasUsed).to.equal(closePositionEvent.payout!);

                // Check fees have been paid
                expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
            });

            it("Price Decreased", async function () {
                const { sendStakingOpenPositionRequest, createSignedClosePositionRequest, computeMaxInterest, publicClient, wasabiLongPool, user1, ibgt, ibgtRewardVault, mockSwap, feeReceiver, initialPrice, wbera, vault } = await loadFixture(deployLongPoolMockEnvironment);
    
                // Open Position
                const {position} = await sendStakingOpenPositionRequest();
    
                await time.increase(86400n); // 1 day later
                await mockSwap.write.setPrice([ibgt.address, wbera.address, initialPrice * 8n / 10n]); // Price fell 20%
    
                // Close Half of the Position
                const closeAmountDenominator = 2n;
                const interest = await computeMaxInterest(position) / closeAmountDenominator;
                const { request, signature } = await createSignedClosePositionRequest({position, interest, amount: position.collateralAmount / closeAmountDenominator});
    
                const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
                const vaultBalanceBefore = await getBalance(publicClient, wbera.address, vault.address);
                const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });
    
                const hash = await wasabiLongPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account });
    
                const traderBalanceAfter = await publicClient.getBalance({address: user1.account.address });
                const vaultBalanceAfter = await getBalance(publicClient, wbera.address, vault.address);
                const feeReceiverBalanceAfter = await publicClient.getBalance({address: feeReceiver });
    
                // Checks
                const events = await wasabiLongPool.getEvents.PositionDecreased();
                expect(events).to.have.lengthOf(1);
                const closePositionEvent = events[0].args;
                const totalFeesPaid = closePositionEvent.closeFee!;
    
                expect(closePositionEvent.id).to.equal(position.id);
                expect(closePositionEvent.principalRepaid!).to.equal(position.principal / closeAmountDenominator, "Half of the principal should be repaid");
    
                expect(vaultBalanceBefore + closePositionEvent.principalRepaid! + closePositionEvent.interestPaid!).to.equal(vaultBalanceAfter);
    
                const downPaymentReduced = position.downPayment / closeAmountDenominator;
                const totalReturn = closePositionEvent.payout! + closePositionEvent.interestPaid! + closePositionEvent.closeFee! - downPaymentReduced;
                expect(totalReturn).to.equal(downPaymentReduced / -5n * 4n, "on 20% price decrease, total return should be -20% * leverage (4) * adjusted down payment");

                expect(await ibgt.read.balanceOf([ibgtRewardVault.address])).to.equal(position.collateralAmount / closeAmountDenominator, "Pool should have half of the collateral left");
                expect(await ibgt.read.balanceOf([wasabiLongPool.address])).to.equal(0n, "Pool should not hold any collateral");
                expect(await wasabiLongPool.read.isPositionStaked([position.id])).to.equal(true, "Position should still be staked");
    
                // Check trader has been paid
                const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
                expect(traderBalanceAfter - traderBalanceBefore + gasUsed).to.equal(closePositionEvent.payout!);
    
                // Check fees have been paid
                expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);
            });
        });
    });

    describe("Liquidate Staked Position", function () {
        it("Liquidate", async function () {
            const { sendStakingOpenPositionRequest, computeMaxInterest, vault, publicClient, wasabiLongPool, user1, ibgt, ibgtRewardVault, mockSwap, feeReceiver, liquidationFeeReceiver, wbera, liquidator, computeLiquidationPrice } = await loadFixture(deployLongPoolMockEnvironment);
            // Open Position
            const {position} = await sendStakingOpenPositionRequest();

            await time.increase(86400n); // 1 day later

            // Liquidate Position
            const functionCallDataList = getApproveAndSwapFunctionCallData(mockSwap.address, ibgt.address, wbera.address, position.collateralAmount);

            const interest = await computeMaxInterest(position);
            const liquidationPrice = await computeLiquidationPrice(position);

            // If the liquidation price is not reached, should revert
            await mockSwap.write.setPrice([ibgt.address, wbera.address, liquidationPrice + 1n]); 
            await expect(wasabiLongPool.write.liquidatePosition([PayoutType.UNWRAPPED, interest, position, functionCallDataList, zeroAddress], { account: liquidator.account }))
                .to.be.rejectedWith("LiquidationThresholdNotReached", "Cannot liquidate position if liquidation price is not reached");

            // Liquidate
            await mockSwap.write.setPrice([ibgt.address, wbera.address, liquidationPrice]); 

            const balancesBefore = await takeBalanceSnapshot(publicClient, wbera.address, vault.address, user1.account.address, feeReceiver, liquidationFeeReceiver);
            const traderBalanceBefore = await publicClient.getBalance({address: user1.account.address });
            const feeReceiverBalanceBefore = await publicClient.getBalance({address: feeReceiver });
            const liquidationFeeReceiverBalanceBefore = await publicClient.getBalance({address: liquidationFeeReceiver });            

            const hash = await wasabiLongPool.write.liquidatePosition([PayoutType.UNWRAPPED, interest, position, functionCallDataList, zeroAddress], { account: liquidator.account });

            const balancesAfter = await takeBalanceSnapshot(publicClient, wbera.address, vault.address, user1.account.address, feeReceiver);
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
            expect(await ibgt.read.balanceOf([wasabiLongPool.address])).to.equal(0n, "Pool should not have any collateral left");
            expect(await ibgt.read.balanceOf([ibgtRewardVault.address])).to.equal(0n, "Reward vault should not have any staked collateral left");

            expect(balancesBefore.get(vault.address) + liquidatePositionEvent.principalRepaid! + liquidatePositionEvent.interestPaid!).to.equal(balancesAfter.get(vault.address)!);

            // Check trader has been paid
            expect(traderBalanceAfter - traderBalanceBefore).to.equal(liquidatePositionEvent.payout!);

            // Check fees have been paid
            expect(feeReceiverBalanceAfter - feeReceiverBalanceBefore).to.equal(totalFeesPaid);

            // Check liquidation fee receiver balance
            const liquidationFeeExpected = position.downPayment * 5n / 100n;
            expect(liquidationFeeReceiverBalanceAfter - liquidationFeeReceiverBalanceBefore).to.equal(liquidationFeeExpected);
        });

        it("Liquidate with no payout", async function () {
            const { sendStakingOpenPositionRequest, computeMaxInterest, wasabiLongPool, ibgt, mockSwap, wbera, liquidator, computeLiquidationPrice } = await loadFixture(deployLongPoolMockEnvironment);
            // Open Position
            const {position} = await sendStakingOpenPositionRequest();
    
            await time.increase(86400n); // 1 day later
    
            // Liquidate Position
            const functionCallDataList = getApproveAndSwapFunctionCallData(mockSwap.address, ibgt.address, wbera.address, position.collateralAmount);
    
            const interest = await computeMaxInterest(position);
            const liquidationPrice = await computeLiquidationPrice(position);
    
            // If the liquidation price is not reached, should revert
            await mockSwap.write.setPrice([ibgt.address, wbera.address, liquidationPrice + 1n]); 
            await expect(wasabiLongPool.write.liquidatePosition([PayoutType.UNWRAPPED, interest, position, functionCallDataList, zeroAddress], { account: liquidator.account }))
                .to.be.rejectedWith("LiquidationThresholdNotReached", "Cannot liquidate position if liquidation price is not reached");
    
            await mockSwap.write.setPrice([ibgt.address, wbera.address, liquidationPrice / 2n]); 
    
            await wasabiLongPool.write.liquidatePosition([PayoutType.UNWRAPPED, interest, position, functionCallDataList, zeroAddress], { account: liquidator.account });
            // Checks for no payout
            const events = await wasabiLongPool.getEvents.PositionLiquidated();
            expect(events).to.have.lengthOf(1);
            const liquidatePositionEvent = events[0].args;
            expect(liquidatePositionEvent.payout!).to.equal(0n);
        });
    });

    describe("Claim Rewards", function () {
        it("Claim rewards automatically when closing position", async function () {
            const { user1, wasabiLongPool, sendStakingOpenPositionRequest, createSignedClosePositionRequest, wbera, ibgt, ibgtInfraredVault, stakingAccountFactory } = await loadFixture(deployLongPoolMockEnvironment);

            const { position } = await sendStakingOpenPositionRequest();
            const user1StakingAccountAddress = await stakingAccountFactory.read.userToStakingAccount([user1.account.address]);
            
            await time.increase(86400n); // 1 day later

            const wberaBalanceBefore = await wbera.read.balanceOf([user1.account.address]);
            const ibgtBalanceBefore = await ibgt.read.balanceOf([user1.account.address]);

            const { request, signature } = await createSignedClosePositionRequest({ position, interest: 0n });
            await wasabiLongPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account });

            const wberaBalanceAfter = await wbera.read.balanceOf([user1.account.address]);
            const ibgtBalanceAfter = await ibgt.read.balanceOf([user1.account.address]);

            // Checks
            const rewardsClaimedEvents = await stakingAccountFactory.getEvents.StakingRewardsClaimed();
            expect(rewardsClaimedEvents).to.have.lengthOf(2);
            const wberaRewardsClaimedEvent = rewardsClaimedEvents[0].args;
            const ibgtRewardsClaimedEvent = rewardsClaimedEvents[1].args;

            expect(wberaRewardsClaimedEvent.user).to.equal(getAddress(user1.account.address));
            expect(wberaRewardsClaimedEvent.stakingAccount).to.equal(user1StakingAccountAddress);
            expect(wberaRewardsClaimedEvent.stakingContract).to.equal(ibgtInfraredVault.address);
            expect(wberaRewardsClaimedEvent.stakingType).to.equal(0);
            expect(wberaRewardsClaimedEvent.rewardToken).to.equal(getAddress(wbera.address));
            expect(wberaRewardsClaimedEvent.amount).to.be.gt(0n);
            expect(wberaBalanceAfter).to.equal(wberaBalanceBefore + wberaRewardsClaimedEvent.amount!);

            expect(ibgtRewardsClaimedEvent.user).to.equal(getAddress(user1.account.address));
            expect(ibgtRewardsClaimedEvent.stakingAccount).to.equal(user1StakingAccountAddress);
            expect(ibgtRewardsClaimedEvent.stakingContract).to.equal(ibgtInfraredVault.address);
            expect(ibgtRewardsClaimedEvent.stakingType).to.equal(0);
            expect(ibgtRewardsClaimedEvent.rewardToken).to.equal(getAddress(ibgt.address));
            expect(ibgtRewardsClaimedEvent.amount).to.be.gt(0n);
            expect(ibgtBalanceAfter).to.equal(ibgtBalanceBefore + ibgtRewardsClaimedEvent.amount!);
        });

        it("Claim rewards manually after rewards period has ended", async function () {
            const { user1, sendStakingOpenPositionRequest, wbera, ibgt, ibgtInfraredVault, stakingAccountFactory } = await loadFixture(deployLongPoolMockEnvironment);

            await sendStakingOpenPositionRequest();
            const user1StakingAccountAddress = await stakingAccountFactory.read.userToStakingAccount([user1.account.address]);
            
            await time.increase(864000n); // 10 days later

            const wberaBalanceBefore = await wbera.read.balanceOf([user1.account.address]);
            const ibgtBalanceBefore = await ibgt.read.balanceOf([user1.account.address]);

            await stakingAccountFactory.write.claimRewards([ibgt.address], { account: user1.account });

            const wberaBalanceAfter = await wbera.read.balanceOf([user1.account.address]);
            const ibgtBalanceAfter = await ibgt.read.balanceOf([user1.account.address]);

            // Checks
            const rewardsClaimedEvents = await stakingAccountFactory.getEvents.StakingRewardsClaimed();
            expect(rewardsClaimedEvents).to.have.lengthOf(2);
            const wberaRewardsClaimedEvent = rewardsClaimedEvents[0].args;
            const ibgtRewardsClaimedEvent = rewardsClaimedEvents[1].args;

            expect(wberaRewardsClaimedEvent.user).to.equal(getAddress(user1.account.address));
            expect(wberaRewardsClaimedEvent.stakingAccount).to.equal(user1StakingAccountAddress);
            expect(wberaRewardsClaimedEvent.stakingContract).to.equal(ibgtInfraredVault.address);
            expect(wberaRewardsClaimedEvent.stakingType).to.equal(0);
            expect(wberaRewardsClaimedEvent.rewardToken).to.equal(getAddress(wbera.address));
            expect(wberaRewardsClaimedEvent.amount).to.be.approximately(parseEther("100"), parseEther("0.01"));
            expect(wberaBalanceAfter).to.equal(wberaBalanceBefore + wberaRewardsClaimedEvent.amount!);

            expect(ibgtRewardsClaimedEvent.user).to.equal(getAddress(user1.account.address));
            expect(ibgtRewardsClaimedEvent.stakingAccount).to.equal(user1StakingAccountAddress);
            expect(ibgtRewardsClaimedEvent.stakingContract).to.equal(ibgtInfraredVault.address);
            expect(ibgtRewardsClaimedEvent.stakingType).to.equal(0);
            expect(ibgtRewardsClaimedEvent.rewardToken).to.equal(getAddress(ibgt.address));
            expect(ibgtRewardsClaimedEvent.amount).to.be.approximately(parseEther("100"), parseEther("0.01"));
            expect(ibgtBalanceAfter).to.equal(ibgtBalanceBefore + ibgtRewardsClaimedEvent.amount!);
        })
    })

    describe("Validations", function () {
        it("Cannot stake position for another user", async function () {
            const { sendStakingOpenPositionRequest, user2, wasabiLongPool } = await loadFixture(deployLongPoolMockEnvironment);
            const { position } = await sendStakingOpenPositionRequest();

            await expect(wasabiLongPool.write.stakePosition([position], { account: user2.account }))
                .to.be.rejectedWith("SenderNotTrader", "Cannot stake position for another user");
        })

        it("Cannot stake position if already staked", async function () {
            const { sendStakingOpenPositionRequest, user1, wasabiLongPool } = await loadFixture(deployLongPoolMockEnvironment);
            const { position } = await sendStakingOpenPositionRequest();

            await expect(wasabiLongPool.write.stakePosition([position], { account: user1.account }))
                .to.be.rejectedWith("PositionAlreadyStaked", "Cannot stake position if already staked");
        })

        it("Cannot increase a staked position without staking", async function () {
            const { sendStakingOpenPositionRequest, user1, wasabiLongPool, openPositionRequest, orderSigner } = await loadFixture(deployLongPoolMockEnvironment);
            const { position } = await sendStakingOpenPositionRequest();

            const increaseOpenPositionRequest = {...openPositionRequest, id: position.id, existingPosition: position}
            const increaseSignature = await signOpenPositionRequest(orderSigner, "WasabiLongPool", wasabiLongPool.address, increaseOpenPositionRequest);
            await expect(wasabiLongPool.write.openPosition([increaseOpenPositionRequest, increaseSignature], { account: user1.account }))
                .to.be.rejectedWith("CannotPartiallyStakePosition", "Cannot increase a staked position without staking");
        })

        it("Cannot increase and stake an unstaked position", async function () {
            const { sendDefaultOpenPositionRequest, user1, wasabiLongPool, openPositionRequest, orderSigner } = await loadFixture(deployLongPoolMockEnvironment);
            const { position } = await sendDefaultOpenPositionRequest();

            const increaseOpenPositionRequest = {...openPositionRequest, id: position.id, existingPosition: position}
            const increaseSignature = await signOpenPositionRequest(orderSigner, "WasabiLongPool", wasabiLongPool.address, increaseOpenPositionRequest);
            await expect(wasabiLongPool.write.openPositionAndStake([increaseOpenPositionRequest, increaseSignature], { account: user1.account }))
                .to.be.rejectedWith("CannotPartiallyStakePosition", "Cannot increase and stake an unstaked position");
        })
    });
});