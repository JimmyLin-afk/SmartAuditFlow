import { loadFixture } from "@nomicfoundation/hardhat-toolbox-viem/network-helpers";
import hre from "hardhat";
import { expect } from "chai";
import { parseEther, zeroAddress, maxUint256 } from "viem";
import { deployLongPoolMockEnvironment } from "./berachainFixtures";
import { getBalance, takeBalanceSnapshot } from "../utils/StateUtils";
import { checkDepositEvents, checkWithdrawEvents, splitSharesWithFee } from "./berachainHelpers";
import { VAULT_ADMIN_ROLE } from "../utils/constants";

describe("BeraVault", function () {
    describe("Deployment", function () {
        it("Should deploy BeraVault correctly", async function () {
            const { vault, rewardVault, infraredVault, bgt } = await loadFixture(deployLongPoolMockEnvironment);
            expect(vault.address).to.not.equal(zeroAddress);
            expect(await vault.read.getRewardVault()).to.equal(rewardVault.address);
            expect(await vault.read.getInfraredVault()).to.equal(infraredVault.address);
            expect(await rewardVault.read.stakeToken()).to.equal(vault.address);
            expect(await rewardVault.read.rewardToken()).to.equal(bgt.address);
            expect(await vault.read.allowance([vault.address, infraredVault.address])).to.equal(maxUint256);
        });
    });

    describe("Deposit and withdraw", function () {
        it("Should deposit WBERA and stake fees in RewardVault", async function () {
            const { vault, rewardVault, wbera, user1, publicClient } = await loadFixture(deployLongPoolMockEnvironment);

            const amount = parseEther("100");
            await wbera.write.deposit({ value: amount, account: user1.account });
            await wbera.write.approve([vault.address, amount], { account: user1.account });

            const wberaBalancesBefore = await takeBalanceSnapshot(publicClient, wbera.address, user1.account.address, vault.address);
            const vaultSharesBefore = await takeBalanceSnapshot(publicClient, vault.address, user1.account.address, vault.address, rewardVault.address);

            await vault.write.deposit([amount, user1.account.address], { account: user1.account });

            const wberaBalancesAfter = await takeBalanceSnapshot(publicClient, wbera.address, user1.account.address, vault.address);
            const vaultSharesAfter = await takeBalanceSnapshot(publicClient, vault.address, user1.account.address, vault.address, rewardVault.address);
            const { sharesMinusFee, rewardFee } = await splitSharesWithFee(hre, vault.address, amount);

            expect(wberaBalancesBefore.get(user1.account.address) - amount).to.equal(wberaBalancesAfter.get(user1.account.address));
            expect(wberaBalancesBefore.get(vault.address) + amount).to.equal(wberaBalancesAfter.get(vault.address));
            expect(vaultSharesBefore.get(user1.account.address) + sharesMinusFee).to.equal(vaultSharesAfter.get(user1.account.address));
            expect(vaultSharesBefore.get(rewardVault.address) + rewardFee).to.equal(vaultSharesAfter.get(rewardVault.address));
            expect(vaultSharesAfter.get(vault.address)).to.equal(0n);

            await checkDepositEvents(hre, vault.address, user1.account.address, amount);
        });

        it("Should deposit BERA and stake fees in RewardVault", async function () {
            const { vault, rewardVault, wbera, user1, publicClient } = await loadFixture(deployLongPoolMockEnvironment);

            const amount = parseEther("100");

            const beraBalanceBefore = await getBalance(publicClient, zeroAddress, user1.account.address);
            const wberaBalancesBefore = await takeBalanceSnapshot(publicClient, wbera.address, user1.account.address, vault.address);
            const vaultSharesBefore = await takeBalanceSnapshot(publicClient, vault.address, user1.account.address, vault.address, rewardVault.address);

            const hash = await vault.write.depositEth([user1.account.address], { account: user1.account, value: amount });
            const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);

            const beraBalanceAfter = await getBalance(publicClient, zeroAddress, user1.account.address);
            const wberaBalancesAfter = await takeBalanceSnapshot(publicClient, wbera.address, user1.account.address, vault.address);
            const vaultSharesAfter = await takeBalanceSnapshot(publicClient, vault.address, user1.account.address, vault.address, rewardVault.address);
            const { sharesMinusFee, rewardFee } = await splitSharesWithFee(hre, vault.address, amount);

            expect(beraBalanceBefore - amount - gasUsed).to.equal(beraBalanceAfter);
            expect(wberaBalancesBefore.get(user1.account.address)).to.equal(wberaBalancesAfter.get(user1.account.address));
            expect(wberaBalancesBefore.get(vault.address) + amount).to.equal(wberaBalancesAfter.get(vault.address));
            expect(vaultSharesBefore.get(user1.account.address) + sharesMinusFee).to.equal(vaultSharesAfter.get(user1.account.address));
            expect(vaultSharesBefore.get(rewardVault.address) + rewardFee).to.equal(vaultSharesAfter.get(rewardVault.address));
            expect(vaultSharesAfter.get(vault.address)).to.equal(0n);

            await checkDepositEvents(hre, vault.address, user1.account.address, amount);
        });

        it("Should unstake fees and withdraw", async function () {
            const { vault, wbera, owner, publicClient } = await loadFixture(deployLongPoolMockEnvironment);

            // Owner already deposited in fixture
            const depositAmount = await vault.read.cumulativeBalanceOf([owner.account.address]);
            const wberaBalancesBefore = await takeBalanceSnapshot(publicClient, wbera.address, owner.account.address, vault.address);

            await vault.write.withdraw([depositAmount, owner.account.address, owner.account.address], { account: owner.account });

            const wberaBalancesAfter = await takeBalanceSnapshot(publicClient, wbera.address, owner.account.address, vault.address);
            const userSharesAfter = await vault.read.cumulativeBalanceOf([owner.account.address]);

            expect(wberaBalancesBefore.get(owner.account.address) + depositAmount).to.equal(wberaBalancesAfter.get(owner.account.address));
            expect(wberaBalancesBefore.get(vault.address) - depositAmount).to.equal(wberaBalancesAfter.get(vault.address));
            expect(userSharesAfter).to.equal(0);

            await checkWithdrawEvents(hre, vault.address, owner.account.address, depositAmount, depositAmount);
        });

        it("Should partially unstake fees and withdraw", async function () {
            const { vault, wbera, owner, publicClient } = await loadFixture(deployLongPoolMockEnvironment);

            // Owner already deposited in fixture
            const depositAmount = await vault.read.cumulativeBalanceOf([owner.account.address]);
            const withdrawAmount = depositAmount / 7n;
            const expectedShares = await vault.read.previewWithdraw([withdrawAmount]);
            const wberaBalancesBefore = await takeBalanceSnapshot(publicClient, wbera.address, owner.account.address, vault.address);
            
            await vault.write.withdraw([withdrawAmount, owner.account.address, owner.account.address], { account: owner.account });

            const wberaBalancesAfter = await takeBalanceSnapshot(publicClient, wbera.address, owner.account.address, vault.address);
            const userSharesAfter = await vault.read.cumulativeBalanceOf([owner.account.address]);

            expect(wberaBalancesBefore.get(owner.account.address) + withdrawAmount).to.equal(wberaBalancesAfter.get(owner.account.address));
            expect(wberaBalancesBefore.get(vault.address) - withdrawAmount).to.equal(wberaBalancesAfter.get(vault.address));
            expect(userSharesAfter).to.equal(depositAmount - withdrawAmount);

            await checkWithdrawEvents(hre, vault.address, owner.account.address, withdrawAmount, expectedShares);

            // Withdraw the rest
            const expectedAssets = await vault.read.previewRedeem([userSharesAfter]);

            await vault.write.withdraw([userSharesAfter, owner.account.address, owner.account.address], { account: owner.account });

            const finalWethBalances = await takeBalanceSnapshot(publicClient, wbera.address, owner.account.address, vault.address);
            const userSharesFinal = await vault.read.cumulativeBalanceOf([owner.account.address]);

            expect(wberaBalancesAfter.get(owner.account.address) + userSharesAfter).to.equal(finalWethBalances.get(owner.account.address));
            expect(wberaBalancesAfter.get(vault.address) - userSharesAfter).to.equal(finalWethBalances.get(vault.address));
            expect(userSharesFinal).to.equal(0);

            await checkWithdrawEvents(hre, vault.address, owner.account.address, expectedAssets, userSharesAfter);
        });
    });

    describe("Share transfers", function () {
        it("Should transfer all shares to user, including reward fee balance", async function () {
            const { vault, user1, owner, publicClient } = await loadFixture(deployLongPoolMockEnvironment);

            // Owner already deposited in fixture
            const ownerBalanceBefore = await vault.read.cumulativeBalanceOf([owner.account.address]);
            const rewardFee = await vault.read.getRewardFeeUserBalance([owner.account.address]);
            const sharesBefore = await takeBalanceSnapshot(publicClient, vault.address, owner.account.address, user1.account.address);

            await vault.write.transfer([user1.account.address, sharesBefore.get(owner.account.address)], { account: owner.account });

            const sharesAfter = await takeBalanceSnapshot(publicClient, vault.address, owner.account.address, user1.account.address);
            const userBalanceAfter = await vault.read.cumulativeBalanceOf([user1.account.address]);

            expect(sharesAfter.get(user1.account.address)).to.equal(sharesBefore.get(owner.account.address));
            expect(sharesAfter.get(owner.account.address)).to.equal(0n);
            expect(userBalanceAfter).to.equal(ownerBalanceBefore);
            expect(await vault.read.getRewardFeeUserBalance([user1.account.address])).to.equal(rewardFee);
            expect(await vault.read.getRewardFeeUserBalance([owner.account.address])).to.equal(0n);
        });

        it("Should transfer partial shares to user, including reward fee balance", async function () {
            const { vault, user1, owner, publicClient } = await loadFixture(deployLongPoolMockEnvironment);

            // Owner already deposited in fixture
            const ownerBalanceBefore = await vault.read.cumulativeBalanceOf([owner.account.address]);
            const rewardFee = await vault.read.getRewardFeeUserBalance([owner.account.address]);
            const sharesBefore = await takeBalanceSnapshot(publicClient, vault.address, owner.account.address, user1.account.address);
            const transferAmount = sharesBefore.get(owner.account.address) / 7n;

            await vault.write.transfer([user1.account.address, transferAmount], { account: owner.account });

            const sharesAfter = await takeBalanceSnapshot(publicClient, vault.address, owner.account.address, user1.account.address);
            const userBalanceAfter = await vault.read.cumulativeBalanceOf([user1.account.address]);

            expect(sharesAfter.get(user1.account.address)).to.equal(transferAmount);
            expect(sharesAfter.get(owner.account.address)).to.equal(sharesBefore.get(owner.account.address) - transferAmount);
            expect(userBalanceAfter).to.equal(ownerBalanceBefore / 7n);
            expect(await vault.read.getRewardFeeUserBalance([user1.account.address])).to.equal(rewardFee / 7n);
            expect(await vault.read.getRewardFeeUserBalance([owner.account.address])).to.equal(rewardFee - (rewardFee / 7n));
        });


        it("Should transfer partial shares to user with random percent, including reward fee balance", async function () {
            const { vault, user1, owner, publicClient } = await loadFixture(deployLongPoolMockEnvironment);

            const denom = 1000n;

            let ownerCumBalanceBefore = await vault.read.cumulativeBalanceOf([owner.account.address]);
            let user1CumBalanceBefore = await vault.read.cumulativeBalanceOf([user1.account.address]);
            let ownerSharesBefore = await vault.read.balanceOf([owner.account.address]);
            let user1SharesBefore = await vault.read.balanceOf([user1.account.address]);
            let ownerRewardFeeBefore = await vault.read.getRewardFeeUserBalance([owner.account.address]);
            let user1RewardFeeBefore = await vault.read.getRewardFeeUserBalance([user1.account.address]);

            for (let i = 0; i < 10; i++) {
                const numerator = BigInt(Math.floor(Math.random() * Number(denom)));
    
                const transferAmount = ownerSharesBefore * numerator / denom;
    
                await vault.write.transfer([user1.account.address, transferAmount], { account: owner.account });
    
                const sharesAfter = await takeBalanceSnapshot(publicClient, vault.address, owner.account.address, user1.account.address);
                const userCumBalanceAfter = await vault.read.cumulativeBalanceOf([user1.account.address]);

                // Check Balances
                expect(sharesAfter.get(user1.account.address)).to.equal(user1SharesBefore + transferAmount);
                expect(sharesAfter.get(owner.account.address)).to.equal(ownerSharesBefore - transferAmount);
                expect(userCumBalanceAfter).to.be.approximately(user1CumBalanceBefore + ownerCumBalanceBefore * numerator / denom, 2n);

                // Check Reward Fee Balances
                const user1RewardFeeAfter = await vault.read.getRewardFeeUserBalance([user1.account.address]);
                const ownerRewardFeeAfter = await vault.read.getRewardFeeUserBalance([owner.account.address]);

                expect(user1RewardFeeAfter).to.be.approximately(user1RewardFeeBefore + ownerRewardFeeBefore * numerator / denom, 2n);
                expect(ownerRewardFeeAfter).to.be.approximately(ownerRewardFeeBefore - (ownerRewardFeeBefore * numerator / denom), 2n);

                // Set for next iteration
                ownerSharesBefore = sharesAfter.get(owner.account.address);
                user1SharesBefore = sharesAfter.get(user1.account.address);

                user1CumBalanceBefore = userCumBalanceAfter;
                ownerCumBalanceBefore = await vault.read.cumulativeBalanceOf([owner.account.address]);
                
                ownerRewardFeeBefore = ownerRewardFeeAfter;
                user1RewardFeeBefore = user1RewardFeeAfter;
            }
        });

        it("Should transfer all shares from owner, including reward fee balance", async function () {
            const { vault, user1, owner, publicClient } = await loadFixture(deployLongPoolMockEnvironment);

            // Owner already deposited in fixture
            const ownerBalanceBefore = await vault.read.cumulativeBalanceOf([owner.account.address]);
            const rewardFee = await vault.read.getRewardFeeUserBalance([owner.account.address]);
            const sharesBefore = await takeBalanceSnapshot(publicClient, vault.address, owner.account.address, user1.account.address);

            await vault.write.approve([user1.account.address, sharesBefore.get(owner.account.address)], { account: owner.account });
            await vault.write.transferFrom(
                [owner.account.address, user1.account.address, sharesBefore.get(owner.account.address)], 
                { account: user1.account });

            const sharesAfter = await takeBalanceSnapshot(publicClient, vault.address, owner.account.address, user1.account.address);
            const userBalanceAfter = await vault.read.cumulativeBalanceOf([user1.account.address]);

            expect(sharesAfter.get(user1.account.address)).to.equal(sharesBefore.get(owner.account.address));
            expect(sharesAfter.get(owner.account.address)).to.equal(0n);
            expect(userBalanceAfter).to.equal(ownerBalanceBefore);
            expect(await vault.read.getRewardFeeUserBalance([user1.account.address])).to.equal(rewardFee);
            expect(await vault.read.getRewardFeeUserBalance([owner.account.address])).to.equal(0n);
        });

        it("Should transfer partial shares from owner, including reward fee balance", async function () {
            const { vault, user1, owner, publicClient } = await loadFixture(deployLongPoolMockEnvironment);

            // Owner already deposited in fixture
            const ownerBalanceBefore = await vault.read.cumulativeBalanceOf([owner.account.address]);
            const rewardFee = await vault.read.getRewardFeeUserBalance([owner.account.address]);
            const sharesBefore = await takeBalanceSnapshot(publicClient, vault.address, owner.account.address, user1.account.address);
            const transferAmount = sharesBefore.get(owner.account.address) / 7n;

            await vault.write.approve([user1.account.address, transferAmount], { account: owner.account });
            await vault.write.transferFrom(
                [owner.account.address, user1.account.address, transferAmount], 
                { account: user1.account });

            const sharesAfter = await takeBalanceSnapshot(publicClient, vault.address, owner.account.address, user1.account.address);
            const userBalanceAfter = await vault.read.cumulativeBalanceOf([user1.account.address]);

            expect(sharesAfter.get(user1.account.address)).to.equal(transferAmount);
            expect(sharesAfter.get(owner.account.address)).to.equal(sharesBefore.get(owner.account.address) - transferAmount);
            expect(userBalanceAfter).to.equal(ownerBalanceBefore / 7n);
            expect(await vault.read.getRewardFeeUserBalance([user1.account.address])).to.equal(rewardFee / 7n);
            expect(await vault.read.getRewardFeeUserBalance([owner.account.address])).to.equal(rewardFee - (rewardFee / 7n));
        });

        it("Should not update reward fee balance on staking/unstaking with RewardVault", async function () {
            const { vault, rewardVault, owner, publicClient } = await loadFixture(deployLongPoolMockEnvironment);

            // Owner already deposited in fixture
            const shares = await vault.read.balanceOf([owner.account.address]);
            const userBalanceBefore = await vault.read.cumulativeBalanceOf([owner.account.address]);
            const rewardFee = await vault.read.getRewardFeeUserBalance([owner.account.address]);

            await vault.write.approve([rewardVault.address, shares], { account: owner.account });
            await rewardVault.write.stake([shares], { account: owner.account });

            const sharesAfterStake = await takeBalanceSnapshot(publicClient, vault.address, owner.account.address, rewardVault.address);
            const userBalanceAfterStake = await vault.read.cumulativeBalanceOf([owner.account.address]);

            expect(sharesAfterStake.get(owner.account.address)).to.equal(0n);
            expect(sharesAfterStake.get(rewardVault.address)).to.equal(shares + rewardFee);
            expect(userBalanceAfterStake).to.equal(userBalanceBefore);
            expect(await vault.read.getRewardFeeUserBalance([owner.account.address])).to.equal(rewardFee);

            await rewardVault.write.withdraw([shares], { account: owner.account });

            const sharesAfterUnstake = await takeBalanceSnapshot(publicClient, vault.address, owner.account.address, rewardVault.address);
            const userBalanceAfterUnstake = await vault.read.cumulativeBalanceOf([owner.account.address]);

            expect(sharesAfterUnstake.get(owner.account.address)).to.equal(shares);
            expect(sharesAfterUnstake.get(rewardVault.address)).to.equal(rewardFee);
            expect(userBalanceAfterUnstake).to.equal(userBalanceBefore);
            expect(await vault.read.getRewardFeeUserBalance([owner.account.address])).to.equal(rewardFee);
        });

        it("Should not update reward fee balance on staking/unstaking with InfraredVault", async function () {
            const { vault, infraredVault, rewardVault, owner, publicClient } = await loadFixture(deployLongPoolMockEnvironment);

            // Owner already deposited in fixture
            const shares = await vault.read.balanceOf([owner.account.address]);
            const userBalanceBefore = await vault.read.cumulativeBalanceOf([owner.account.address]);
            const rewardFee = await vault.read.getRewardFeeUserBalance([owner.account.address]);

            await vault.write.approve([infraredVault.address, shares], { account: owner.account });
            await infraredVault.write.stake([shares], { account: owner.account });

            const sharesAfterStake = await takeBalanceSnapshot(publicClient, vault.address, owner.account.address, rewardVault.address);
            const userBalanceAfterStake = await vault.read.cumulativeBalanceOf([owner.account.address]);

            expect(sharesAfterStake.get(owner.account.address)).to.equal(0n);
            expect(sharesAfterStake.get(rewardVault.address)).to.equal(shares + rewardFee);
            expect(userBalanceAfterStake).to.equal(userBalanceBefore);
            expect(await vault.read.getRewardFeeUserBalance([owner.account.address])).to.equal(rewardFee);

            await infraredVault.write.withdraw([shares], { account: owner.account });

            const sharesAfterUnstake = await takeBalanceSnapshot(publicClient, vault.address, owner.account.address, rewardVault.address);
            const userBalanceAfterUnstake = await vault.read.cumulativeBalanceOf([owner.account.address]);

            expect(sharesAfterUnstake.get(owner.account.address)).to.equal(shares);
            expect(sharesAfterUnstake.get(rewardVault.address)).to.equal(rewardFee);
            expect(userBalanceAfterUnstake).to.equal(userBalanceBefore);
            expect(await vault.read.getRewardFeeUserBalance([owner.account.address])).to.equal(rewardFee);
        });

        describe("Validations", function () {
            it("Should revert if transfer to RewardVault directly", async function () {
                const { vault, rewardVault, owner } = await loadFixture(deployLongPoolMockEnvironment);

                // Owner already deposited in fixture
                const shares = await vault.read.balanceOf([owner.account.address]);
                await expect(vault.write.transfer([rewardVault.address, shares], { account: owner.account }))
                    .to.be.rejectedWith("ERC20InvalidReceiver");
            });

            it("Should revert if transfer to InfraredVault directly", async function () {
                const { vault, infraredVault, owner } = await loadFixture(deployLongPoolMockEnvironment);

                // Owner already deposited in fixture
                const shares = await vault.read.balanceOf([owner.account.address]);
                await expect(vault.write.transfer([infraredVault.address, shares], { account: owner.account }))
                    .to.be.rejectedWith("ERC20InvalidReceiver");
            });
            
            it("Should revert if transfer to vault directly", async function () {
                const { vault, owner } = await loadFixture(deployLongPoolMockEnvironment);

                // Owner already deposited in fixture
                const shares = await vault.read.balanceOf([owner.account.address]);
                await expect(vault.write.transfer([vault.address, shares], { account: owner.account }))
                    .to.be.rejectedWith("ERC20InvalidReceiver");
            });
        });
    });

    describe("Deposit cap", function () {
        it("Can only deposit up to the deposit cap", async function () {
            const {vault, owner, user1, wbera, vaultAdmin} = await loadFixture(deployLongPoolMockEnvironment);

            // Owner already deposited in fixture
            const ownerShares = await vault.read.cumulativeBalanceOf([owner.account.address]);
            const ownerAssets = await vault.read.convertToAssets([ownerShares]);

            // Set the deposit cap to the owner's deposit
            await vault.write.setDepositCap([ownerAssets], { account: vaultAdmin.account });

            // Check that user1's deposit is not allowed
            const depositAmount = parseEther("1");
            await wbera.write.approve([vault.address, depositAmount], { account: user1.account });
            await expect(vault.write.deposit(
                [depositAmount, user1.account.address], 
                { account: user1.account }
            )).to.be.rejectedWith("ERC4626ExceededMaxDeposit");

            // Raise the deposit cap
            await vault.write.setDepositCap([ownerAssets + depositAmount], { account: vaultAdmin.account });

            // Check that user1's deposit is allowed
            await vault.write.deposit(
                [depositAmount, user1.account.address], 
                { account: user1.account }
            );

            // Check that user1 can't deposit more than the new cap
            await expect(vault.write.deposit(
                [depositAmount, user1.account.address], 
                { account: user1.account }
            )).to.be.rejectedWith("ERC4626ExceededMaxDeposit");
        });

        it("Competition depositor can increase deposit cap and deposit", async function () {
            const {vault, owner, user1, wbera, competitionDepositor, manager, vaultAdmin} = await loadFixture(deployLongPoolMockEnvironment);

            // Owner already deposited in fixture
            const ownerShares = await vault.read.cumulativeBalanceOf([owner.account.address]);
            const ownerAssets = await vault.read.convertToAssets([ownerShares]);
            
            // Set the deposit cap to the owner's deposit
            await vault.write.setDepositCap([ownerAssets], { account: vaultAdmin.account });

            // Grant admin role to competition depositor
            await manager.write.grantRole(
                [VAULT_ADMIN_ROLE, competitionDepositor.address, 0],
                { account: owner.account }
            );
            
            // Add allocation for user1 to competition depositor
            const depositAmount = parseEther("1");
            await wbera.write.approve([competitionDepositor.address, depositAmount], { account: user1.account });
            await competitionDepositor.write.setAllocations(
                [[user1.account.address], [depositAmount]],
                { account: owner.account }
            );

            // Deposit from user1 through competition depositor
            await competitionDepositor.write.deposit({ account: user1.account });
            
            // Check that user1's deposit was successful
            const user1Shares = await vault.read.cumulativeBalanceOf([user1.account.address]);
            const user1Assets = await vault.read.convertToAssets([user1Shares]);
            expect(user1Assets).to.equal(depositAmount);

            // Check that user1's allocation is removed from competition depositor
            const user1Allocation = await competitionDepositor.read.depositAllocations([user1.account.address]);
            expect(user1Allocation).to.equal(0n);
        });
    });


    describe("Max redeem and withdraw", function () {
        describe("No stake", function () {
            it("Max redeem should be full balance", async function () {
                const { vault, owner } = await loadFixture(deployLongPoolMockEnvironment);

                // Owner already deposited in fixture
                const cumulativeBalance = await vault.read.cumulativeBalanceOf([owner.account.address]);
                const maxRedeem = await vault.read.maxRedeem([owner.account.address]);

                expect(maxRedeem).to.equal(cumulativeBalance);
                expect(await vault.write.redeem(
                    [maxRedeem, owner.account.address, owner.account.address], 
                    { account: owner.account }
                )).to.emit(vault, "Withdraw").withArgs(
                    owner.account.address, owner.account.address, owner.account.address, maxRedeem, maxRedeem
                );
            });

            it("Max withdraw should be full balance with interest", async function () {
                const { vault, weth, owner, vaultAdmin } = await loadFixture(deployLongPoolMockEnvironment);

                // Donate 1 WETH to the vault as interest
                const interestAmount = parseEther("1");
                await weth.write.deposit({ value: interestAmount, account: vaultAdmin.account });
                await weth.write.approve([vault.address, interestAmount], { account: vaultAdmin.account });
                await vault.write.donate([interestAmount], { account: vaultAdmin.account });

                // Owner already deposited in fixture
                const cumulativeBalance = await vault.read.cumulativeBalanceOf([owner.account.address]);
                const maxWithdraw = await vault.read.maxWithdraw([owner.account.address]);

                expect(maxWithdraw).to.be.approximately(cumulativeBalance + interestAmount, 1n);
                expect(await vault.write.withdraw(
                    [maxWithdraw, owner.account.address, owner.account.address], 
                    { account: owner.account }
                )).to.emit(vault, "Withdraw").withArgs(
                    owner.account.address, owner.account.address, owner.account.address, maxWithdraw, cumulativeBalance
                );
            });
        });

        describe("Half staked", function () {
            it("Max redeem should be half of cumulative balance", async function () {
                const { vault, infraredVault, owner } = await loadFixture(deployLongPoolMockEnvironment);

                // Owner already deposited in fixture
                const cumulativeBalance = await vault.read.cumulativeBalanceOf([owner.account.address]);
                const shareBalance = await vault.read.balanceOf([owner.account.address]);
                await vault.write.approve([infraredVault.address, shareBalance / 2n], { account: owner.account });
                await infraredVault.write.stake([shareBalance / 2n], { account: owner.account });

                const maxRedeem = await vault.read.maxRedeem([owner.account.address]);

                expect(maxRedeem).to.equal(cumulativeBalance / 2n);
                expect(await vault.write.redeem(
                    [maxRedeem, owner.account.address, owner.account.address], 
                    { account: owner.account }
                )).to.emit(vault, "Withdraw").withArgs(
                    owner.account.address, owner.account.address, owner.account.address, maxRedeem, maxRedeem
                );
            });

            it("Max withdraw should be half of cumulative balance with interest", async function () {
                const { vault, infraredVault, weth, owner, vaultAdmin } = await loadFixture(deployLongPoolMockEnvironment);

                // Donate 1 WETH to the vault as interest
                const interestAmount = parseEther("1");
                await weth.write.deposit({ value: interestAmount, account: vaultAdmin.account });
                await weth.write.approve([vault.address, interestAmount], { account: vaultAdmin.account });
                await vault.write.donate([interestAmount], { account: vaultAdmin.account });

                // Owner already deposited in fixture
                const cumulativeBalance = await vault.read.cumulativeBalanceOf([owner.account.address]);
                const shareBalance = await vault.read.balanceOf([owner.account.address]);
                await vault.write.approve([infraredVault.address, shareBalance / 2n], { account: owner.account });
                await infraredVault.write.stake([shareBalance / 2n], { account: owner.account });

                const maxWithdraw = await vault.read.maxWithdraw([owner.account.address]);

                expect(maxWithdraw).to.be.approximately((cumulativeBalance + interestAmount) / 2n, 1n);
                expect(await vault.write.withdraw(
                    [maxWithdraw, owner.account.address, owner.account.address], 
                    { account: owner.account }
                )).to.emit(vault, "Withdraw").withArgs(
                    owner.account.address, owner.account.address, owner.account.address, maxWithdraw, cumulativeBalance
                );
            });
        });

        describe("All staked", function () {
            it("Max redeem should be 0", async function () {
                const { vault, infraredVault, owner } = await loadFixture(deployLongPoolMockEnvironment);

                // Owner already deposited in fixture
                const shareBalance = await vault.read.balanceOf([owner.account.address]);
                await vault.write.approve([infraredVault.address, shareBalance], { account: owner.account });
                await infraredVault.write.stake([shareBalance], { account: owner.account });

                const maxRedeem = await vault.read.maxRedeem([owner.account.address]);

                expect(maxRedeem).to.equal(0n);
            });

            it("Max withdraw should be 0", async function () {
                const { vault, infraredVault, weth, owner, vaultAdmin } = await loadFixture(deployLongPoolMockEnvironment);

                // Donate 1 WETH to the vault as interest
                const interestAmount = parseEther("1");
                await weth.write.deposit({ value: interestAmount, account: vaultAdmin.account });
                await weth.write.approve([vault.address, interestAmount], { account: vaultAdmin.account });
                await vault.write.donate([interestAmount], { account: vaultAdmin.account });

                // Owner already deposited in fixture
                const shareBalance = await vault.read.balanceOf([owner.account.address]);
                await vault.write.approve([infraredVault.address, shareBalance], { account: owner.account });
                await infraredVault.write.stake([shareBalance], { account: owner.account });

                const maxWithdraw = await vault.read.maxWithdraw([owner.account.address]);

                expect(maxWithdraw).to.equal(0n);
            });
        });
    });
});