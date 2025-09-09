import {
    time,
    loadFixture,
} from "@nomicfoundation/hardhat-toolbox-viem/network-helpers";
import hre from "hardhat";
import { expect } from "chai";
import { formatEther, maxUint256, parseEther } from "viem";
import { deployLongPoolMockEnvironment, deployMockV2VaultImpl } from "./fixtures";
import { getBalance, takeBalanceSnapshot } from "./utils/StateUtils";
import { PayoutType } from "./utils/PerpStructUtils";
import { VAULT_ADMIN_ROLE } from "./utils/constants";

describe("WasabiVault", function () {

    describe("Interest earned", function () {
        it("Interest earned and sold", async function () {
            const {
                sendDefaultOpenPositionRequest,
                createSignedClosePositionRequest,
                wasabiLongPool,
                user1,
                owner,
                vault,
                publicClient,
                wethAddress,
            } = await loadFixture(deployLongPoolMockEnvironment);
            
            // Owner already deposited in fixture
            const depositAmount = await getBalance(publicClient, wethAddress, vault.address);
            const sharesPerEthBefore = await vault.read.convertToShares([parseEther("1")]);

            const shares = await vault.read.balanceOf([owner.account.address]);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();

            await time.increase(86400n); // 1 day later

            // Close Position
            const { request, signature } = await createSignedClosePositionRequest({ position });

            const hash = await wasabiLongPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account });
            // const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
            // console.log("Close position gas cost: ", formatEther(gasUsed));

            // Checks
            const closePositionEvents = await wasabiLongPool.getEvents.PositionClosed();
            expect(closePositionEvents).to.have.lengthOf(1, "PositionClosed event not emitted");
            const closePositionEvent = closePositionEvents[0].args;
            const interest = closePositionEvent.interestPaid!;

            const wethBalanceBefore = await getBalance(publicClient, wethAddress, owner.account.address);
            
            await vault.write.redeem([shares, owner.account.address, owner.account.address], { account: owner.account });

            const event = (await vault.getEvents.Withdraw())[0].args!;
            const wethBalanceAfter = await getBalance(publicClient, wethAddress, owner.account.address);
            const sharesPerEthAfter = await vault.read.convertToShares([parseEther("1")]);
            const withdrawAmount = event.assets!;
            
            expect(await vault.read.balanceOf([owner.account.address])).to.equal(interest / 10n, "Fee receiver should have received 10% interest fee");
            expect(wethBalanceAfter - wethBalanceBefore).to.equal(withdrawAmount, "Balance change does not match withdraw amount");
            expect(withdrawAmount).to.be.gt(depositAmount, "Withdraw amount should be greater than deposit amount");
            expect(sharesPerEthAfter).to.be.lt(sharesPerEthBefore, "Shares per ETH should decrease due to interest fee shares");
        });

        it("Very small interest fee", async function () {
            const {
                sendDefaultOpenPositionRequest,
                createSignedClosePositionRequest,
                wasabiLongPool,
                user1,
                owner,
                vault,
                publicClient,
                wethAddress,
            } = await loadFixture(deployLongPoolMockEnvironment);
            
            // Owner already deposited in fixture
            const depositAmount = await getBalance(publicClient, wethAddress, vault.address);
            const sharesPerEthBefore = await vault.read.convertToShares([parseEther("1")]);

            const shares = await vault.read.balanceOf([owner.account.address]);

            // Open Position
            const {position} = await sendDefaultOpenPositionRequest();

            await time.increase(60n); // 1 minute later

            // Close Position
            const smallInterest = 1n;
            const { request, signature } = await createSignedClosePositionRequest({ position, interest: smallInterest });

            await wasabiLongPool.write.closePosition([PayoutType.UNWRAPPED, request, signature], { account: user1.account });

            // Checks
            const closePositionEvents = await wasabiLongPool.getEvents.PositionClosed();
            expect(closePositionEvents).to.have.lengthOf(1, "PositionClosed event not emitted");
            const closePositionEvent = closePositionEvents[0].args;
            const interest = closePositionEvent.interestPaid!;

            const wethBalanceBefore = await getBalance(publicClient, wethAddress, owner.account.address);
            
            const hash =
                await vault.write.redeem([shares, owner.account.address, owner.account.address], { account: owner.account });
            // const gasUsed = await publicClient.getTransactionReceipt({hash}).then(r => r.gasUsed * r.effectiveGasPrice);
            // console.log("Redeem gas cost: ", formatEther(gasUsed));

            const withdrawEvent = (await vault.getEvents.Withdraw())[0].args!;
            const wethBalanceAfter = await getBalance(publicClient, wethAddress, owner.account.address);
            const sharesPerEthAfter = await vault.read.convertToShares([parseEther("1")]);
            const withdrawAmount = withdrawEvent.assets!;

            expect(interest).to.equal(smallInterest, "Interest should be equal to the small interest");
            expect(wethBalanceAfter - wethBalanceBefore).to.equal(withdrawAmount, "Balance change does not match withdraw amount");
            expect(withdrawAmount).to.be.gt(depositAmount, "Withdraw amount should be greater than deposit amount");
            expect(sharesPerEthAfter).to.equal(sharesPerEthBefore, "Shares per ETH should not change due to no interest fee");
        });
    });

    describe("Strategies", function () {
        it("Aave strategy deposit, claim and withdraw", async function () {
            const {
                strategy,
                mockAToken,
                mockAavePool,
                owner,
                vault,
                publicClient,
                weth,
                strategyDeposit,
                strategyClaim,
                strategyWithdraw,
                strategyAddInterest,
            } = await loadFixture(deployLongPoolMockEnvironment);

            // Owner already deposited in fixture
            const ownerShares = await vault.read.balanceOf([owner.account.address]);
            const assetsPerShareBefore = await vault.read.convertToAssets([parseEther("1")]);
            const wethBalancesBefore = await takeBalanceSnapshot(
                publicClient, weth.address, strategy.address, vault.address, mockAavePool.address
            );

            // Deposit from vault into strategy
            const depositAmount = ownerShares / 2n;
            await strategyDeposit(depositAmount);

            // StrategyDeposit checks
            const wethBalancesAfterDeposit = await takeBalanceSnapshot(
                publicClient, weth.address, strategy.address, vault.address, mockAavePool.address
            );
            const strategyCollateralBalanceAfterDeposit = await mockAToken.read.balanceOf([strategy.address]);
            expect(wethBalancesAfterDeposit.get(vault.address)).to.equal(
                wethBalancesBefore.get(vault.address) - depositAmount, 
                "Vault WETH balance should be reduced by deposit amount"
            );
            expect(wethBalancesAfterDeposit.get(mockAavePool.address)).to.equal(
                wethBalancesBefore.get(mockAavePool.address) + depositAmount, 
                "Aave pool should have received deposit amount"
            );
            expect(strategyCollateralBalanceAfterDeposit).to.equal(
                depositAmount,
                "Strategy should have received deposit amount in collateral"
            );

            // Claim interest earned
            const interest = depositAmount / 100n;
            await strategyAddInterest(interest);
            await strategyClaim(interest);

            // StrategyClaim checks
            const wethBalancesAfterClaim = await takeBalanceSnapshot(
                publicClient, weth.address, strategy.address, vault.address, mockAavePool.address
            );
            const strategyCollateralBalanceAfterClaim = await mockAToken.read.balanceOf([strategy.address]);
            expect(wethBalancesAfterClaim.get(vault.address)).to.equal(
                wethBalancesAfterDeposit.get(vault.address),
                "Vault WETH balance should not change"
            );
            expect(wethBalancesAfterClaim.get(mockAavePool.address)).to.equal(
                wethBalancesAfterDeposit.get(mockAavePool.address) + interest,
                "Aave pool WETH balance should be increased by interest"
            );
            expect(strategyCollateralBalanceAfterClaim).to.equal(
                depositAmount + interest,
                "Strategy should have received interest in collateral"
            );

            // Add more interest without claiming - it will be claimed when withdrawing
            await strategyAddInterest(interest);

            // Withdraw from strategy 
            const withdrawAmount = depositAmount + 2n * interest;
            await strategyWithdraw(withdrawAmount);

            // StrategyWithdraw checks
            const wethBalancesAfterWithdraw = await takeBalanceSnapshot(
                publicClient, weth.address, strategy.address, vault.address, mockAavePool.address
            );
            const strategyCollateralBalanceAfterWithdraw = await mockAToken.read.balanceOf([strategy.address]);
            const strategyDebtAfterWithdraw = await vault.read.strategyDebt([strategy.address]);
            const assetsPerShareAfter = await vault.read.convertToAssets([parseEther("1")]);
            expect(assetsPerShareAfter).to.be.gt(
                assetsPerShareBefore, 
                "Assets per share should increase after interest is paid"
            );
            expect(wethBalancesAfterWithdraw.get(vault.address)).to.equal(
                wethBalancesBefore.get(vault.address) + 2n * interest,
                "Vault WETH balance should be increased by interest and debt repaid"
            );
            expect(wethBalancesAfterWithdraw.get(mockAavePool.address)).to.equal(
                wethBalancesBefore.get(mockAavePool.address),
                "Aave pool WETH balance should be back to where it started"
            );
            expect(strategyCollateralBalanceAfterWithdraw).to.equal(
                0n,
                "Strategy should have no collateral left"
            );
            expect(strategyDebtAfterWithdraw).to.equal(
                0n, 
                "Strategy debt should be zero after full withdraw"
            );
        });
    });

    describe("Dust cleaning", function () {
        it("Clean dust from vault", async function () {
            const {
                user1,
                owner,
                vaultAdmin,
                vault,
                publicClient,
                weth,
            } = await loadFixture(deployLongPoolMockEnvironment);
            
            // Owner already deposited in fixture
            const ownerShares = await vault.read.balanceOf([owner.account.address]);

            // Deposit from user1
            const depositAmount = parseEther("1");
            await weth.write.approve([vault.address, depositAmount], { account: user1.account });
            await vault.write.deposit([depositAmount, user1.account.address], { account: user1.account });
            const userShares = await vault.read.balanceOf([user1.account.address]);

            // Withdraw all deposits
            await vault.write.redeem([ownerShares, owner.account.address, owner.account.address], { account: owner.account });
            await vault.write.redeem([userShares, user1.account.address, user1.account.address], { account: user1.account });

            expect(await vault.read.balanceOf([owner.account.address])).to.equal(0n);
            expect(await vault.read.balanceOf([user1.account.address])).to.equal(0n);
            expect(await getBalance(publicClient, weth.address, vault.address)).to.equal(0n);

            // Donate dust
            const dust = 1n;
            await weth.write.deposit({ value: dust, account: vaultAdmin.account });
            await weth.write.approve([vault.address, dust], { account: vaultAdmin.account });
            await vault.write.donate([dust], { account: vaultAdmin.account });

            // Check distorting effect of dust
            const sharesPerEthBefore = await vault.read.convertToShares([parseEther("1")]);
            expect(sharesPerEthBefore).to.equal(parseEther("0.5"));

            // Clean dust
            await vault.write.cleanDust({ account: vaultAdmin.account });

            // Checks
            const sharesPerEthAfter = await vault.read.convertToShares([parseEther("1")]);
            expect(sharesPerEthAfter).to.equal(parseEther("1"));
            expect(await getBalance(publicClient, weth.address, vault.address)).to.equal(0n);
        });

        it("Very small deposit is not treated as dust", async function () {
            const {
                user1,
                owner,
                vault,
                publicClient,
                weth,
            } = await loadFixture(deployLongPoolMockEnvironment);
            
            // Owner already deposited in fixture
            const ownerShares = await vault.read.balanceOf([owner.account.address]);

            // Deposit very small amount from user1
            const depositAmount = parseEther("0.000000001");
            await weth.write.approve([vault.address, depositAmount], { account: user1.account });
            await vault.write.deposit([depositAmount, user1.account.address], { account: user1.account });
            const userShares = await vault.read.balanceOf([user1.account.address]);

            // Withdraw owner's large deposit first and check that user's small deposit is still there
            await vault.write.redeem([ownerShares, owner.account.address, owner.account.address], { account: owner.account });

            expect(await vault.read.balanceOf([owner.account.address])).to.equal(0n);
            expect(await vault.read.balanceOf([user1.account.address])).to.be.greaterThan(0n);
            expect(await getBalance(publicClient, weth.address, vault.address)).to.be.greaterThan(0n);

            // Withdraw user's small deposit
            await vault.write.redeem([userShares, user1.account.address, user1.account.address], { account: user1.account });

            expect(await vault.read.balanceOf([user1.account.address])).to.equal(0n);
            expect(await getBalance(publicClient, weth.address, vault.address)).to.equal(0n);

            // Check that dust is not left behind after withdrawing user's small deposit
            const sharesPerEth = await vault.read.convertToShares([parseEther("1")]);
            expect(sharesPerEth).to.equal(parseEther("1"));
        });
    });

    describe("Deposit cap", function () {
        it("Can only deposit up to the deposit cap", async function () {
            const {vault, owner, user1, vaultAdmin, weth} = await loadFixture(deployLongPoolMockEnvironment);

            // Owner already deposited in fixture
            const ownerShares = await vault.read.balanceOf([owner.account.address]);
            const ownerAssets = await vault.read.convertToAssets([ownerShares]);

            // Set the deposit cap to the owner's deposit
            await vault.write.setDepositCap([ownerAssets], { account: vaultAdmin.account });

            // Check that user1's deposit is not allowed
            const depositAmount = parseEther("1");
            await weth.write.approve([vault.address, depositAmount], { account: user1.account });
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
            const {vault, owner, vaultAdmin, user1, weth, competitionDepositor, manager} = await loadFixture(deployLongPoolMockEnvironment);

            // Owner already deposited in fixture
            const ownerShares = await vault.read.balanceOf([owner.account.address]);
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
            await weth.write.approve([competitionDepositor.address, depositAmount], { account: user1.account });
            await competitionDepositor.write.setAllocations(
                [[user1.account.address], [depositAmount]],
                { account: owner.account }
            );

            // Deposit from user1 through competition depositor
            await competitionDepositor.write.deposit({ account: user1.account });
            
            // Check that user1's deposit was successful
            const user1Shares = await vault.read.balanceOf([user1.account.address]);
            const user1Assets = await vault.read.convertToAssets([user1Shares]);
            expect(user1Assets).to.equal(depositAmount);

            // Check that user1's allocation is removed from competition depositor
            const user1Allocation = await competitionDepositor.read.depositAllocations([user1.account.address]);
            expect(user1Allocation).to.equal(0n);
        });
    });

    describe("Timelock", function () {
        it("Upgrade to timelock", async function () {
            const {vault, owner, upgradeVaultToTimelock} = await loadFixture(deployLongPoolMockEnvironment);

            const cooldownDuration = 864000n; // 10 days
            await upgradeVaultToTimelock(cooldownDuration);
            const timelockVault = await hre.viem.getContractAt("TimelockWasabiVault", vault.address);

            expect(await timelockVault.read.getCooldownDuration()).to.equal(cooldownDuration);
            // Owner deposited before timelock upgrade, so timelock end should be zero timestamp plus 10 days
            expect(await timelockVault.read.getCooldowns([owner.account.address])).to.be.empty;
        });

        it("Withdrawals with multiple cooldowns", async function () {
            const {vault, owner, upgradeVaultToTimelock} = await loadFixture(deployLongPoolMockEnvironment);

            await upgradeVaultToTimelock();
            const timelockVault = await hre.viem.getContractAt("TimelockWasabiVault", vault.address);

            await timelockVault.write.startCooldown([parseEther("1")], { account: owner.account });
            await time.increase(432000n);

            await timelockVault.write.startCooldown([parseEther("4")], { account: owner.account });
            await time.increase(432000n);

            await timelockVault.write.startCooldown([parseEther("5")], { account: owner.account });
            await time.increase(432000n);

            let cooldowns = await timelockVault.read.getCooldowns([owner.account.address]);
            expect(cooldowns).to.have.lengthOf(3);
            expect(cooldowns[0].amount).to.equal(parseEther("1"));
            expect(cooldowns[1].amount).to.equal(parseEther("4"));
            expect(cooldowns[2].amount).to.equal(parseEther("5"));
            
            await timelockVault.write.withdraw(
                [parseEther("2"), owner.account.address, owner.account.address],
                { account: owner.account }
            );

            cooldowns = await timelockVault.read.getCooldowns([owner.account.address]);
            expect(cooldowns).to.have.lengthOf(3);
            expect(cooldowns[0].amount).to.equal(0n);
            expect(cooldowns[0].cooldownStart).to.equal(0n);
            expect(cooldowns[1].amount).to.equal(parseEther("3"));
            expect(cooldowns[2].amount).to.equal(parseEther("5"));

            await timelockVault.write.withdraw(
                [parseEther("1"), owner.account.address, owner.account.address],
                { account: owner.account }
            );

            cooldowns = await timelockVault.read.getCooldowns([owner.account.address]);
            expect(cooldowns).to.have.lengthOf(3);
            expect(cooldowns[1].amount).to.equal(parseEther("2"));
            expect(cooldowns[2].amount).to.equal(parseEther("5"));

            await time.increase(432000n);
            await timelockVault.write.withdraw(
                [parseEther("6"), owner.account.address, owner.account.address],
                { account: owner.account }
            );

            cooldowns = await timelockVault.read.getCooldowns([owner.account.address]);
            expect(cooldowns).to.have.lengthOf(3);
            expect(cooldowns[1].amount).to.equal(0n);
            expect(cooldowns[1].cooldownStart).to.equal(0n);
            expect(cooldowns[2].amount).to.equal(parseEther("1"));
        });

        it("Can withdraw immediately if cooldown is 0", async function () {
            const {vault, owner, upgradeVaultToTimelock} = await loadFixture(deployLongPoolMockEnvironment);

            await upgradeVaultToTimelock();
            const timelockVault = await hre.viem.getContractAt("TimelockWasabiVault", vault.address);

            await expect(timelockVault.write.withdraw(
                [parseEther("1"), owner.account.address, owner.account.address],
                { account: owner.account }
            )).to.be.rejectedWith("InsufficientCooldown");

            await timelockVault.write.setCooldownDuration([0n], { account: owner.account });

            await expect(timelockVault.write.withdraw(
                [parseEther("1"), owner.account.address, owner.account.address],
                { account: owner.account }
            )).to.be.fulfilled;
        })

        it("Clears cooldowns when withdrawing with 0 cooldown duration", async function () {
            const {vault, owner, upgradeVaultToTimelock} = await loadFixture(deployLongPoolMockEnvironment);

            await upgradeVaultToTimelock();
            const timelockVault = await hre.viem.getContractAt("TimelockWasabiVault", vault.address);
        

            await timelockVault.write.startCooldown([parseEther("1")], { account: owner.account });
            await time.increase(432000n);

            await timelockVault.write.startCooldown([parseEther("4")], { account: owner.account });
            await time.increase(432000n);

            await timelockVault.write.startCooldown([parseEther("5")], { account: owner.account });
            await time.increase(432000n);

            let cooldowns = await timelockVault.read.getCooldowns([owner.account.address]);
            expect(cooldowns).to.have.lengthOf(3);

            await timelockVault.write.setCooldownDuration([0n], { account: owner.account });

            await timelockVault.write.withdraw(
                [parseEther("10"), owner.account.address, owner.account.address],
                { account: owner.account }
            );

            cooldowns = await timelockVault.read.getCooldowns([owner.account.address]);
            expect(cooldowns).to.have.lengthOf(3);
            expect(cooldowns[0].amount).to.equal(0n);
            expect(cooldowns[0].cooldownStart).to.equal(0n);
            expect(cooldowns[1].amount).to.equal(0n);
            expect(cooldowns[1].cooldownStart).to.equal(0n);
            expect(cooldowns[2].amount).to.equal(0n);
            expect(cooldowns[2].cooldownStart).to.equal(0n);
        })
    });

    describe("Validations", function () {
        it("Only pools can borrow", async function () {
            const {vault, user1} = await loadFixture(deployLongPoolMockEnvironment);

            await expect(vault.write.borrow(
                [1n],
                { account: user1.account }
            )).to.be.rejectedWith("CallerNotPool");
        })

        it("Only pools can record repayment", async function () {
            const {vault, user1} = await loadFixture(deployLongPoolMockEnvironment);

            await expect(vault.write.recordRepayment(
                [1n, 1n, false],
                { account: user1.account }
            )).to.be.rejectedWith("CallerNotPool");

            await expect(vault.write.recordRepayment(
                [1n, 1n, true],
                { account: user1.account }
            )).to.be.rejectedWith("CallerNotPool");
        })

        it("Only admin can upgrade", async function () {
            const {vault, owner, user1} = await loadFixture(deployLongPoolMockEnvironment);

            const {newVaultImpl} = await loadFixture(deployMockV2VaultImpl)

            await expect(vault.write.upgradeToAndCall(
                [newVaultImpl.address, "0x"],
                { account: user1.account }
            )).to.be.rejectedWith("AccessManagerUnauthorizedAccount");

            await expect(vault.write.upgradeToAndCall(
                [newVaultImpl.address, "0x"],
                { account: owner.account }
            )).to.be.fulfilled;
        })

        it("Only vault admin can donate", async function () {
            const {vault, owner, vaultAdmin, weth} = await loadFixture(deployLongPoolMockEnvironment);

            await expect(vault.write.donate(
                [1n],
                { account: owner.account }
            )).to.be.rejectedWith("AccessManagerUnauthorizedAccount");

            await weth.write.deposit({ value: 1n, account: vaultAdmin.account });
            await weth.write.approve([vault.address, 1n], { account: vaultAdmin.account });

            await expect (vault.write.donate(
                [1n],
                { account: vaultAdmin.account }
            )).to.be.fulfilled;
        })

        it("Only vault admin can clean dust", async function () {
            const {vault, user1} = await loadFixture(deployLongPoolMockEnvironment);

            await expect(vault.write.cleanDust(
                { account: user1.account }
            )).to.be.rejectedWith("AccessManagerUnauthorizedAccount");
        })

        it("Only admin can claim interest", async function () {
            const {vault, weth, strategy, user2, owner, vaultAdmin} = await loadFixture(deployLongPoolMockEnvironment);

            const depositAmount = parseEther("1");
            const interest = depositAmount / 100n;
            await vault.write.strategyDeposit([strategy.address, depositAmount], { account: owner.account });

            await expect(vault.write.strategyClaim(
                [strategy.address],
                { account: user2.account }
            )).to.be.rejectedWith("AccessManagerUnauthorizedAccount");

            await expect(vault.write.strategyClaim(
                [strategy.address],
                { account: vaultAdmin.account }
            )).to.be.rejectedWith("AccessManagerUnauthorizedAccount");

            await weth.write.deposit({ value: interest, account: owner.account });
            await weth.write.approve([vault.address, interest], { account: owner.account });
            await expect(vault.write.strategyClaim(
                [strategy.address],
                { account: owner.account })
            ).to.be.fulfilled;
        })
        
        it("Only depositor can redeem", async function () {
            const {vault, owner, user1, orderExecutor} = await loadFixture(deployLongPoolMockEnvironment);

            // Owner already deposited in fixture
            const shares = await vault.read.balanceOf([owner.account.address]);

            await expect(vault.write.redeem(
                [shares, owner.account.address, owner.account.address], 
                { account: user1.account }
            )).to.be.rejectedWith("ERC20InsufficientAllowance");

            await expect(vault.write.redeem(
                [shares, owner.account.address, owner.account.address], 
                { account: orderExecutor.account }
            )).to.be.rejectedWith("ERC20InsufficientAllowance");
        });

        it("Only depositor can withdraw", async function () {
            const {vault, owner, user1, orderExecutor} = await loadFixture(deployLongPoolMockEnvironment);

            // Owner already deposited in fixture
            const shares = await vault.read.balanceOf([owner.account.address]);
            const assets = await vault.read.convertToAssets([shares]);

            await expect(vault.write.withdraw(
                [assets, owner.account.address, owner.account.address], 
                { account: user1.account }
            )).to.be.rejectedWith("ERC20InsufficientAllowance");

            await expect(vault.write.withdraw(
                [assets, owner.account.address, owner.account.address], 
                { account: orderExecutor.account }
            )).to.be.rejectedWith("ERC20InsufficientAllowance");
        });

        it("Deposits cannot exceed cap", async function () {
            const {vault, owner, user1, weth, vaultAdmin} = await loadFixture(deployLongPoolMockEnvironment);

            // Owner already deposited in fixture
            const shares = await vault.read.balanceOf([owner.account.address]);
            const assets = await vault.read.convertToAssets([shares]);

            expect(await vault.read.maxDeposit([user1.account.address])).to.equal(maxUint256);

            // Set the deposit cap to 2x the current assets
            await vault.write.setDepositCap([assets * 2n], { account: vaultAdmin.account });

            expect(await vault.read.maxDeposit([user1.account.address])).to.equal(assets);

            await expect(vault.write.depositEth(
                [user1.account.address], 
                { value: assets + 1n, account: user1.account }
            )).to.be.rejectedWith("ERC4626ExceededMaxDeposit");

            await weth.write.deposit({ value: assets + 1n, account: user1.account });
            await weth.write.approve([vault.address, assets + 1n], { account: user1.account });

            await expect(vault.write.deposit(
                [assets + 1n, user1.account.address], 
                { account: user1.account }
            )).to.be.rejectedWith("ERC4626ExceededMaxDeposit");
        });

        describe("Strategies", function () {
            it("Only admin can deposit vault assets into strategies", async function () {
                const {vault, vaultAdmin, user1, owner, strategy} = await loadFixture(deployLongPoolMockEnvironment);
    
                await expect(vault.write.strategyDeposit(
                    [strategy.address, 1n],
                    { account: user1.account }
                )).to.be.rejectedWith("AccessManagerUnauthorizedAccount");
    
                await expect(vault.write.strategyDeposit(
                    [strategy.address, 1n],
                    { account: vaultAdmin.account }
                )).to.be.rejectedWith("AccessManagerUnauthorizedAccount");
    
                await expect(vault.write.strategyDeposit(
                    [strategy.address, 1n],
                    { account: owner.account }
                )).to.be.fulfilled;
            })

            it("Only admin can withdraw from strategy", async function () {
                const {vault, weth, strategy, user2, owner, vaultAdmin} = await loadFixture(deployLongPoolMockEnvironment);
    
                const depositAmount = parseEther("1");
                await vault.write.strategyDeposit([strategy.address, depositAmount], { account: owner.account });
    
                await expect(vault.write.strategyWithdraw(
                    [strategy.address, depositAmount],
                    { account: user2.account }
                )).to.be.rejectedWith("AccessManagerUnauthorizedAccount");
    
                await expect(vault.write.strategyWithdraw(
                    [strategy.address, depositAmount],
                    { account: vaultAdmin.account }
                )).to.be.rejectedWith("AccessManagerUnauthorizedAccount");
    
                await weth.write.approve([vault.address, depositAmount], { account: owner.account });
                await expect(vault.write.strategyWithdraw(
                    [strategy.address, depositAmount],
                    { account: owner.account })
                ).to.be.fulfilled;
            })

            it("Cannot repay more than debt with strategyWithdraw", async function () {
                const {vault, strategy, owner} = await loadFixture(deployLongPoolMockEnvironment);
    
                const depositAmount = parseEther("1");
                await vault.write.strategyDeposit([strategy.address, depositAmount], { account: owner.account });
    
                const repayAmount = depositAmount + 1n;
                await expect(vault.write.strategyWithdraw(
                    [strategy.address, repayAmount],
                    { account: owner.account }
                )).to.be.rejectedWith("AmountExceedsDebt");
            })
        });

        describe("Competition depositor", function () {
            it("Cannot deposit if not allocated", async function () {
                const {vault, manager, owner, user1, competitionDepositor, vaultAdmin} = await loadFixture(deployLongPoolMockEnvironment);

                // Owner already deposited in fixture
                const ownerShares = await vault.read.balanceOf([owner.account.address]);
                const ownerAssets = await vault.read.convertToAssets([ownerShares]);
    
                // Set the deposit cap to the owner's deposit
                await vault.write.setDepositCap([ownerAssets], { account: vaultAdmin.account });

                // Grant admin role to competition depositor
                await manager.write.grantRole(
                    [VAULT_ADMIN_ROLE, competitionDepositor.address, 0],
                    { account: owner.account }
                );
                
                await expect(competitionDepositor.write.deposit({ account: user1.account }))
                    .to.be.rejectedWith("SenderNotAllocated");
            });
            
            it("Cannot deposit if competition depositor renounces admin role", async function () {
                const {vault, manager, owner, user1, competitionDepositor, vaultAdmin, weth} = await loadFixture(deployLongPoolMockEnvironment);

                // Owner already deposited in fixture
                const ownerShares = await vault.read.balanceOf([owner.account.address]);
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
                await weth.write.approve([competitionDepositor.address, depositAmount], { account: user1.account });
                await competitionDepositor.write.setAllocations(
                    [[user1.account.address], [depositAmount]],
                    { account: owner.account }
                );

                // Renounce admin role from competition depositor
                await competitionDepositor.write.renounceVaultAdmin({ account: owner.account });

                await expect(competitionDepositor.write.deposit({ account: user1.account }))
                    .to.be.rejectedWith("DepositWindowClosed");
            });

            it("Cannot deposit if user doesn't have full allocation in wallet", async function () {
                const {vault, manager, owner, user1, user2, competitionDepositor, vaultAdmin, weth} = await loadFixture(deployLongPoolMockEnvironment);

                // Owner already deposited in fixture
                const ownerShares = await vault.read.balanceOf([owner.account.address]);
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
                await weth.write.approve([competitionDepositor.address, depositAmount], { account: user1.account });
                await competitionDepositor.write.setAllocations(
                    [[user1.account.address], [depositAmount]],
                    { account: owner.account }
                );

                // User1 transfers their weth to another user
                const userBalance = await weth.read.balanceOf([user1.account.address]);
                await weth.write.transfer(
                    [user2.account.address, userBalance - depositAmount + 1n],
                    { account: user1.account }
                );

                await expect(competitionDepositor.write.deposit({ account: user1.account }))
                    .to.be.rejectedWith("InsufficientBalance");
            });

            it("Cannot set allocations if not admin", async function () {
                const {user1, competitionDepositor} = await loadFixture(deployLongPoolMockEnvironment);

                await expect(competitionDepositor.write.setAllocations(
                    [[user1.account.address], [parseEther("1")]],
                    { account: user1.account }
                )).to.be.rejectedWith("AccessManagerUnauthorizedAccount");
            });

            it("Cannot set allocations if array lengths are not equal", async function () {
                const {user1, owner, competitionDepositor} = await loadFixture(deployLongPoolMockEnvironment);

                await expect(competitionDepositor.write.setAllocations(
                    [[user1.account.address], [parseEther("1"), parseEther("2")]],
                    { account: owner.account }
                )).to.be.rejectedWith("AllocationsLengthMismatch");
            });
        });

        describe("Timelock", function () {
            it("Cannot withdraw from timelocked vault without cooldown", async function () {
                const {vault, owner, upgradeVaultToTimelock} = await loadFixture(deployLongPoolMockEnvironment);

                await upgradeVaultToTimelock();

                await expect(vault.write.withdraw(
                    [parseEther("1"), owner.account.address, owner.account.address],
                    { account: owner.account }
                )).to.be.rejectedWith("InsufficientCooldown");
            });

            it("Cannot withdraw from timelocked vault until cooldown ends", async function () {
                const {vault, owner, upgradeVaultToTimelock} = await loadFixture(deployLongPoolMockEnvironment);

                await upgradeVaultToTimelock();
                const timelockVault = await hre.viem.getContractAt("TimelockWasabiVault", vault.address);

                await timelockVault.write.startCooldown([parseEther("1")], { account: owner.account });

                await time.increase(432000n);

                await expect(vault.write.withdraw(
                    [parseEther("1"), owner.account.address, owner.account.address],
                    { account: owner.account }
                )).to.be.rejectedWith("InsufficientCooldown");

                await time.increase(432000n);

                await expect(vault.write.withdraw(
                    [parseEther("1"), owner.account.address, owner.account.address],
                    { account: owner.account }
                )).to.be.fulfilled;
            });

            it("Cannot withdraw from timelocked vault if all cooldowns have been withdrawn", async function () {
                const {vault, owner, upgradeVaultToTimelock} = await loadFixture(deployLongPoolMockEnvironment);

                await upgradeVaultToTimelock();
                const timelockVault = await hre.viem.getContractAt("TimelockWasabiVault", vault.address);
                
                await timelockVault.write.startCooldown([parseEther("1")], { account: owner.account });
                await time.increase(864000n);

                await expect(vault.write.withdraw(
                    [parseEther("1"), owner.account.address, owner.account.address],
                    { account: owner.account }
                )).to.be.fulfilled;

                await expect(vault.write.withdraw(
                    [parseEther("1"), owner.account.address, owner.account.address],
                    { account: owner.account }
                )).to.be.rejectedWith("InsufficientCooldown");
            })
            
            it("Cannot withdraw more than cooldown amount from timelocked vault", async function () {
                const {vault, owner, upgradeVaultToTimelock} = await loadFixture(deployLongPoolMockEnvironment);

                await upgradeVaultToTimelock();
                const timelockVault = await hre.viem.getContractAt("TimelockWasabiVault", vault.address);
                await timelockVault.write.startCooldown([parseEther("1")], { account: owner.account });

                await time.increase(864000n);

                await expect(vault.write.withdraw(
                    [parseEther("2"), owner.account.address, owner.account.address],
                    { account: owner.account }
                )).to.be.rejectedWith("InsufficientCooldown");
            });

            it("Cannot start cooldown with 0 amount", async function () {
                const {vault, owner, upgradeVaultToTimelock} = await loadFixture(deployLongPoolMockEnvironment);

                await upgradeVaultToTimelock();
                const timelockVault = await hre.viem.getContractAt("TimelockWasabiVault", vault.address);

                await expect(timelockVault.write.startCooldown([0n], { account: owner.account }))
                    .to.be.rejectedWith("InvalidCooldownAmount");
            });

            it("Cannot start cooldown with amount greater than balance", async function () {
                const {vault, owner, upgradeVaultToTimelock} = await loadFixture(deployLongPoolMockEnvironment);

                await upgradeVaultToTimelock();
                const timelockVault = await hre.viem.getContractAt("TimelockWasabiVault", vault.address);

                // Owner deposited 20 WETH in fixture
                await expect(timelockVault.write.startCooldown([parseEther("21")], { account: owner.account }))
                    .to.be.rejectedWith("InvalidCooldownAmount");
            });

            it("Cannot start cooldown with amount greater than shares already in cooldown", async function () {
                const {vault, owner, upgradeVaultToTimelock} = await loadFixture(deployLongPoolMockEnvironment);

                await upgradeVaultToTimelock();
                const timelockVault = await hre.viem.getContractAt("TimelockWasabiVault", vault.address);

                await timelockVault.write.startCooldown([parseEther("1")], { account: owner.account });
                await time.increase(432000n);

                await timelockVault.write.startCooldown([parseEther("4")], { account: owner.account });
                await time.increase(432000n);

                await timelockVault.write.startCooldown([parseEther("5")], { account: owner.account });
                await time.increase(432000n);

                // Owner deposited 20 WETH in fixture, but 10 WETH is already in cooldown
                await expect(timelockVault.write.startCooldown([parseEther("15")], { account: owner.account }))
                    .to.be.rejectedWith("InvalidCooldownAmount");
            });
        });
    });
});
