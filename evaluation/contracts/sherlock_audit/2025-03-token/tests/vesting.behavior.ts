import { loadFixture } from "@nomicfoundation/hardhat-network-helpers"
import { expect } from "chai"
import { ethers, network, upgrades } from "hardhat";
import { Symmio, SymmVesting, VestingPlanOps__factory } from "../typechain-types";
import { initializeFixture, RunContext } from "./Initialize.fixture"
import { Signer } from "ethers";
import {e} from "../utils"

export function ShouldBehaveLikeVesting() {
	let context: RunContext
	let symmVesting: SymmVesting
	let vestingPlanOps: VestingPlanOps__factory
	let admin: Signer, user1: Signer
	let symmToken: Symmio

	beforeEach(async () => {
		context = await loadFixture(initializeFixture)
		symmVesting = await context.vesting
		vestingPlanOps = await ethers.getContractFactory("VestingPlanOps")
		symmToken = context.symmioToken
		admin = context.signers.admin
		user1 = context.signers.user1
	})

	describe("__vesting_init", () => {
		it("Should grant the admin role to the deployer", async () => {
			for (const role of [
				await symmVesting.DEFAULT_ADMIN_ROLE(),
				await symmVesting.SETTER_ROLE(),
				await symmVesting.PAUSER_ROLE(),
				await symmVesting.UNPAUSER_ROLE(),
				await symmVesting.OPERATOR_ROLE(),
			]) {
				const hasRole = await symmVesting.hasRole(role, context.signers.admin.address)
				await expect (hasRole).to.be.true
			}
		})
	})

	describe("setupVestingPlans", () => {
		it("Should fail if users and amount arrays mismatch", async () => {
			const users = [await context.signers.user1.getAddress(), await context.signers.user2.getAddress()]
			const amounts = ["1000"]

			await expect(symmVesting.setupVestingPlans(await context.symmioToken.getAddress(), "0", "0", users, amounts)).to.be.revertedWithCustomError(
				symmVesting,
				"MismatchArrays",
			)
		})

		it("Should fail if vestingPlan setup before", async () => {
			const users = [await context.signers.user1.getAddress()]
			const amounts = ["1000"]

			await symmVesting.setupVestingPlans(await context.symmioToken.getAddress(), "0", "0", users, amounts)
			await expect(symmVesting.setupVestingPlans(await context.symmioToken.getAddress(), "0", "0", users, amounts)).to.be.revertedWithCustomError(
				vestingPlanOps,
				"AlreadySetup",
			)
		})

		it("Should setup vestingPlan successfully", async () => {
			const users = [await context.signers.user1.getAddress()]
			const amounts = ["1000"]
			const oldTotalVesting = await symmVesting.totalVested(await context.symmioToken.getAddress())

			await expect (await symmVesting.setupVestingPlans(await context.symmioToken.getAddress(), "0", "0", users, amounts)).to.be.not.reverted

			const plan = await symmVesting.vestingPlans(context.symmioToken, await context.signers.user1.getAddress())
			const newTotalVesting = await symmVesting.totalVested(await context.symmioToken.getAddress())

			await expect (newTotalVesting).to.be.equal(oldTotalVesting + "1000")

			await expect (plan.startTime).to.be.equal("0")
			await expect (plan.endTime).to.be.equal("0")
			await expect (plan.amount).to.be.equal(amounts[0])
			await expect (plan.claimedAmount).to.be.equal(0)
		})
	})

	describe("claimUnlockedToken", () => {
		beforeEach(async () => {
			await context.symmioToken.connect(context.signers.admin).mint(await symmVesting.getAddress(), 5000)
			const users = [await context.signers.user1.getAddress()]
			const amounts = ["1000"]
			const now = new Date()
			const startTime = Math.floor(now.getTime() / 1000)

			now.setMonth(now.getMonth() + 9)
			const endTime = Math.floor(now.getTime() / 1000)

			await symmVesting.setupVestingPlans(await context.symmioToken.getAddress(), startTime, endTime, users, amounts)
		})

		it("Should unlockedAmount be zero before vesting starts", async () => {
			const plan = await symmVesting.vestingPlans(context.symmioToken, await context.signers.user1.getAddress())
			const latestBlock = await ethers.provider.getBlock("latest")
			const safeTimestamp = Number(plan.startTime) - 100

			if (safeTimestamp <= (latestBlock?.timestamp ?? 0)) {
				await network.provider.send("evm_setNextBlockTimestamp", [(latestBlock?.timestamp ?? 0) + 1])
			} else {
				await network.provider.send("evm_setNextBlockTimestamp", [safeTimestamp])
			}

			await network.provider.send("evm_mine")

			await expect (await symmVesting.getUnlockedAmountForToken(await context.signers.user1.getAddress(), context.symmioToken)).to.be.equal(0)
		})

		it("Should unlockedAmount be zero at the exact start time", async () => {
			const plan = await symmVesting.vestingPlans(context.symmioToken, await context.signers.user1.getAddress())
			const latestBlock = await ethers.provider.getBlock("latest")
			const safeTimestamp = Number(plan.startTime)

			if (safeTimestamp <= (latestBlock?.timestamp ?? 0)) {
				await network.provider.send("evm_setNextBlockTimestamp", [(latestBlock?.timestamp ?? 0) + 1])
			} else {
				await network.provider.send("evm_setNextBlockTimestamp", [safeTimestamp])
			}

			await network.provider.send("evm_mine")

			await expect (await symmVesting.getUnlockedAmountForToken(await context.signers.user1.getAddress(), context.symmioToken)).to.be.equal(0)
		})

		it("Should unlockedAmount be partial during the vesting period", async () => {
			const plan = await symmVesting.vestingPlans(context.symmioToken, await context.signers.user1.getAddress())
			const midTime = Math.floor(Number((plan.startTime + plan.endTime) / BigInt(2)))

			await network.provider.send("evm_setNextBlockTimestamp", [midTime]) // half of vesting lock
			await network.provider.send("evm_mine")

			const expectedUnlocked = Math.floor(Number((BigInt(1000) * (BigInt(midTime) - plan.startTime)) / (plan.endTime - plan.startTime)))

			await expect (await symmVesting.getUnlockedAmountForToken(await context.signers.user1.getAddress(), context.symmioToken)).to.be.equal(expectedUnlocked)
		})

		it("Should unlockedAmount be the full amount at the exact end time", async () => {
			const plan = await symmVesting.vestingPlans(context.symmioToken, await context.signers.user1.getAddress())
			await network.provider.send("evm_setNextBlockTimestamp", [Number(plan.endTime)])
			await network.provider.send("evm_mine")

			await expect (await symmVesting.getUnlockedAmountForToken(await context.signers.user1.getAddress(), context.symmioToken)).to.be.equal(1000)
		})

		it("Should claimUnlockedToken successfully", async () => {
			let plan = await symmVesting.vestingPlans(context.symmioToken, await context.signers.user1.getAddress())
			await network.provider.send("evm_setNextBlockTimestamp", [Number(plan.endTime)])
			await network.provider.send("evm_mine")

			const oldTotalVested = await symmVesting.totalVested(context.symmioToken)
			const oldClaimedAmount = plan.claimedAmount
			const oldContractBalance = await context.symmioToken.balanceOf(symmVesting)
			const oldUserBalance = await context.symmioToken.balanceOf(context.signers.user1)

			await expect (await symmVesting.connect(context.signers.user1).claimUnlockedToken(await context.symmioToken.getAddress())).to.be.not.reverted
			plan = await symmVesting.vestingPlans(context.symmioToken, await context.signers.user1.getAddress())

			const newTotalVested = await symmVesting.totalVested(context.symmioToken)
			const newClaimedAmount = plan.claimedAmount
			const newContractBalance = await context.symmioToken.balanceOf(symmVesting)
			const newUserBalance = await context.symmioToken.balanceOf(context.signers.user1)

			await expect (newTotalVested).to.be.equal(oldTotalVested - BigInt(1000))
			await expect (newClaimedAmount).to.be.equal(oldClaimedAmount + BigInt(1000))
			await expect (newContractBalance).to.be.equal(oldContractBalance - BigInt(1000))
			await expect (newUserBalance).to.be.equal(oldUserBalance + BigInt(1000))
		})
	})

	describe("claimLockedToken", () => {
		beforeEach(async () => {
			await context.symmioToken.connect(context.signers.admin).mint(await symmVesting.getAddress(), 5000)
			const users = [await context.signers.user1.getAddress()]
			const amounts = ["1000"]
			const now = new Date()
			const startTime = Math.floor(now.getTime() / 1000)

			now.setMonth(now.getMonth() + 9)
			const endTime = Math.floor(now.getTime() / 1000)

			await symmVesting.setupVestingPlans(await context.symmioToken.getAddress(), startTime, endTime, users, amounts)
		})

		it("Should fail if amount be greater than lockedAmount", async () => {
			await expect(symmVesting.claimLockedToken(context.symmioToken, 1001)).to.be.revertedWithCustomError(symmVesting, "InvalidAmount")
		})

		it("Should not revert when claiming locked tokens within allowed amount", async () => {
			await expect(symmVesting.connect(context.signers.user1).claimLockedToken(context.symmioToken, 1000)).to.be.not.reverted
		})

		it("Should decrease total vested amount after claiming", async () => {
			const oldTotalVested = await symmVesting.totalVested(context.symmioToken)

			await symmVesting.connect(context.signers.user1).claimLockedToken(context.symmioToken, 1000)

			const newTotalVested = await symmVesting.totalVested(context.symmioToken)
			await expect (newTotalVested).to.be.equal(oldTotalVested - BigInt(1000))
		})

		it("Should distribute claimed amount correctly", async () => {
			const oldPenaltyContractBalance = await context.symmioToken.balanceOf(await symmVesting.lockedClaimPenaltyReceiver())
			const oldUserBalance = await context.symmioToken.balanceOf(context.signers.user1)

			await symmVesting.connect(context.signers.user1).claimLockedToken(context.symmioToken, 1000)

			const newPenaltyContractBalance = await context.symmioToken.balanceOf(await symmVesting.lockedClaimPenaltyReceiver())
			const newUserBalance = await context.symmioToken.balanceOf(context.signers.user1)

			await expect (newPenaltyContractBalance).to.be.equal(oldPenaltyContractBalance + BigInt(500))
			await expect (newUserBalance).to.be.equal(oldUserBalance + BigInt(500))
		})

		it("Should allow user to claim locked token by percentage", async () => {
			const oldPenaltyContractBalance = await context.symmioToken.balanceOf(await symmVesting.lockedClaimPenaltyReceiver())
			const oldUserBalance = await context.symmioToken.balanceOf(context.signers.user1)

			await symmVesting.connect(context.signers.user1).claimLockedTokenByPercentage(context.symmioToken, e(0.5))

			const newPenaltyContractBalance = await context.symmioToken.balanceOf(await symmVesting.lockedClaimPenaltyReceiver())
			const newUserBalance = await context.symmioToken.balanceOf(context.signers.user1)

			await expect (newPenaltyContractBalance).to.be.equal(oldPenaltyContractBalance + BigInt(250))
			await expect (newUserBalance).to.be.equal(oldUserBalance + BigInt(250))
		})

		it("Should allow admin to claim locked token for user", async()=>{
			const oldPenaltyContractBalance = await context.symmioToken.balanceOf(await symmVesting.lockedClaimPenaltyReceiver())
			const oldUserBalance = await context.symmioToken.balanceOf(context.signers.user1)

			await symmVesting.connect(admin).claimLockedTokenFor(symmToken, user1, 1000)

			const newPenaltyContractBalance = await context.symmioToken.balanceOf(await symmVesting.lockedClaimPenaltyReceiver())
			const newUserBalance = await context.symmioToken.balanceOf(context.signers.user1)

			await expect (newPenaltyContractBalance).to.be.equal(oldPenaltyContractBalance + BigInt(500))
			await expect (newUserBalance).to.be.equal(oldUserBalance + BigInt(500))
		})

		it("Should allow admin to claim locked token for user by percentage", async () => {
			const oldPenaltyContractBalance = await context.symmioToken.balanceOf(await symmVesting.lockedClaimPenaltyReceiver())
			const oldUserBalance = await context.symmioToken.balanceOf(context.signers.user1)

			await symmVesting.connect(context.signers.admin).claimLockedTokenForByPercentage(context.symmioToken, user1, e(0.5))

			const newPenaltyContractBalance = await context.symmioToken.balanceOf(await symmVesting.lockedClaimPenaltyReceiver())
			const newUserBalance = await context.symmioToken.balanceOf(context.signers.user1)

			await expect (newPenaltyContractBalance).to.be.equal(oldPenaltyContractBalance + BigInt(250))
			await expect (newUserBalance).to.be.equal(oldUserBalance + BigInt(250))
		})

		it("Should reset claimed amount to zero after claiming", async () => {
			await symmVesting.connect(context.signers.user1).claimLockedToken(context.symmioToken, 1000)

			const plan = await symmVesting.vestingPlans(context.symmioToken, await context.signers.user1.getAddress())
			await expect (plan.claimedAmount).to.be.equal(0)
		})


	})

	describe("resetVestingPlans", () => {
		beforeEach(async () => {
			await context.symmioToken.connect(context.signers.admin).mint(await symmVesting, 5000)

			const users = [await context.signers.user1.getAddress()]
			const amounts = ["1000"]
			const now = new Date()
			const startTime = Math.floor(now.getTime() / 1000)

			now.setMonth(now.getMonth() + 9)
			const endTime = Math.floor(now.getTime() / 1000)

			await symmVesting.setupVestingPlans(await context.symmioToken.getAddress(), startTime, endTime, users, amounts)
		})

		it("Should reset vesting plan successfully", async () => {
			const user = await context.signers.user1.getAddress()
			const newAmount = BigInt(1500)
			const token = await context.symmioToken.getAddress()

			const planBefore = await symmVesting.vestingPlans(token, user)
			const totalVestedBefore = await symmVesting.totalVested(token)

			await expect(symmVesting.connect(context.signers.admin).resetVestingPlans(token, [user], [newAmount]))
				.to.emit(symmVesting, "VestingPlanReset")
				.withArgs(token, user, newAmount)

			const planAfter = await symmVesting.vestingPlans(token, user)
			const totalVestedAfter = await symmVesting.totalVested(token)

			await expect (planAfter.amount).to.equal(newAmount)
			await expect (planAfter.claimedAmount).to.equal(0)
			await expect (totalVestedAfter).to.equal(totalVestedBefore - planBefore.amount + newAmount)
		})

		it("Should fail if users and amounts arrays have different lengths", async () => {
			const user = await context.signers.user1.getAddress()
			const token = await context.symmioToken.getAddress()

			await expect(symmVesting.connect(context.signers.admin).resetVestingPlans(token, [user], [])).to.be.revertedWithCustomError(
				symmVesting,
				"MismatchArrays",
			)
		})


		it("Should claim unlocked tokens before resetting", async () => {
			const user = await context.signers.user1.getAddress()
			const token = await context.symmioToken.getAddress()
			const newAmount = 1200

			await expect(symmVesting.connect(context.signers.admin).resetVestingPlans(token, [user], [newAmount])).to.not.be.reverted

			const planAfter = await symmVesting.vestingPlans(token, user)
			await expect (planAfter.claimedAmount).to.equal(0)
		})
	})

	describe("modifiers",  ()=>{
		it('should allow PAUSER_ROLE to pause and unpase the contract', async()=>{
			await symmVesting.connect(admin).pause()
			await expect(await symmVesting.paused()).to.be.true

			await expect(symmVesting.connect(admin).resetVestingPlans(
				await symmToken.getAddress(), [await user1.getAddress()], [e(1)]
			)).to.be.revertedWithCustomError(symmVesting, "EnforcedPause")

			await expect(symmVesting.connect(admin).setupVestingPlans(
				await symmToken.getAddress(),
				Math.floor(Date.now() / 1000),
				Math.floor(Date.now() / 1000)+3600,
				[await user1.getAddress()],
				[e(1)]
			)).to.be.revertedWithCustomError(symmVesting, "EnforcedPause")

			await expect(symmVesting.connect(admin).setupVestingPlans(
				await symmToken.getAddress(),
				Math.floor(Date.now() / 1000),
				Math.floor(Date.now() / 1000)+3600,
				[await user1.getAddress()],
				[e(1)]
			)).to.be.revertedWithCustomError(symmVesting, "EnforcedPause")

			await expect(symmVesting.connect(admin).claimUnlockedToken(
				await symmToken.getAddress()
			)).to.be.revertedWithCustomError(symmVesting, "EnforcedPause")

			await expect(symmVesting.connect(admin).claimUnlockedTokenFor(
				await symmToken.getAddress(),
				await user1.getAddress()
			)).to.be.revertedWithCustomError(symmVesting, "EnforcedPause")

			await expect(symmVesting.connect(admin).claimLockedToken(
				await symmToken.getAddress(),
				await user1.getAddress()
			)).to.be.revertedWithCustomError(symmVesting, "EnforcedPause")

			await expect(symmVesting.connect(admin).claimLockedTokenByPercentage(
				await symmToken.getAddress(),
				e(5e-1)
			)).to.be.revertedWithCustomError(symmVesting, "EnforcedPause")

			await expect(symmVesting.connect(admin).claimLockedTokenFor(
				await symmToken.getAddress(),
				await user1.getAddress(),
				e(10)
			)).to.be.revertedWithCustomError(symmVesting, "EnforcedPause")

			await expect(symmVesting.connect(admin).claimLockedTokenForByPercentage(
				await symmToken.getAddress(),
				await user1.getAddress(),
				e(10)
			)).to.be.revertedWithCustomError(symmVesting, "EnforcedPause")

			await expect(symmVesting.connect(user1).addLiquidity(
				e(1),
				0,
				0
			)).to.be.revertedWithCustomError(symmVesting, "EnforcedPause")

			await symmVesting.connect(admin).unpause()
			await expect(await symmVesting.paused()).to.be.false

			await expect(symmVesting.connect(admin).resetVestingPlans(
				await symmToken.getAddress(), [await user1.getAddress()], [e(1)]
			)).to.be.ok

			await expect(symmVesting.connect(admin).setupVestingPlans(
				await symmToken.getAddress(),
				Math.floor(Date.now() / 1000),
				Math.floor(Date.now() / 1000)+3600,
				[await user1.getAddress()],
				[e(1)]
			)).to.be.ok

			await expect(symmVesting.connect(admin).setupVestingPlans(
				await symmToken.getAddress(),
				Math.floor(Date.now() / 1000),
				Math.floor(Date.now() / 1000)+3600,
				[await user1.getAddress()],
				[e(1)]
			)).to.be.ok

			await expect(symmVesting.connect(admin).claimUnlockedToken(
				await symmToken.getAddress()
			)).to.be.ok

			await expect(symmVesting.connect(admin).claimUnlockedTokenFor(
				await symmToken.getAddress(),
				await user1.getAddress()
			)).to.be.ok

			await expect(symmVesting.connect(admin).claimLockedToken(
				await symmToken.getAddress(),
				await user1.getAddress()
			)).to.be.ok

			await expect(symmVesting.connect(admin).claimLockedTokenByPercentage(
				await symmToken.getAddress(),
				e(5e-1)
			)).to.be.ok

			await expect(symmVesting.connect(admin).claimLockedTokenFor(
				await symmToken.getAddress(),
				await user1.getAddress(),
				e(10)
			)).to.be.ok

			await expect(symmVesting.connect(admin).claimLockedTokenForByPercentage(
				await symmToken.getAddress(),
				await user1.getAddress(),
				e(10)
			)).to.be.ok

			await expect(symmVesting.connect(user1).addLiquidity(
				e(1),
				0,
				0
			)).to.be.ok
		})

		it('should revert when initialize method is from nonInitializer/constructor method',
			async()=>{
			const zeroAddress = "0x0000000000000000000000000000000000000000";
			await expect(symmVesting.connect(admin).initialize(admin, admin,
				zeroAddress, zeroAddress, zeroAddress, zeroAddress, zeroAddress, zeroAddress, zeroAddress))
				.to.be.reverted
			const adminAdress = await admin.getAddress()
			await expect(symmVesting.connect(admin).__vesting_init(adminAdress, adminAdress, adminAdress))
				.to.be.reverted
		})

		it('should fail when zero is passed as address to symmVesting initialize method', async()=>{
			const VestingPlanOps = await ethers.getContractFactory("VestingPlanOps")
			const vestingPlanOps = await VestingPlanOps.deploy()
			await vestingPlanOps.waitForDeployment()

			const zeroAddress = "0x0000000000000000000000000000000000000000";
			const nonZeroAddress = "0x0000000000000000000000000000000000000001";

			const VestingFactory = await ethers.getContractFactory("SymmVesting", {
				libraries: {
					VestingPlanOps: await vestingPlanOps.getAddress(),
				},
			})

			await expect( upgrades.deployProxy(VestingFactory, [zeroAddress, nonZeroAddress,
				nonZeroAddress, nonZeroAddress, nonZeroAddress, nonZeroAddress, nonZeroAddress, nonZeroAddress, nonZeroAddress ], {
				unsafeAllow: ["external-library-linking"],
				initializer: "initialize",
			})).to.be.revertedWithCustomError(symmVesting, "ZeroAddress")

			await expect( upgrades.deployProxy(VestingFactory, [nonZeroAddress, zeroAddress,
				nonZeroAddress, nonZeroAddress, nonZeroAddress, nonZeroAddress, nonZeroAddress, nonZeroAddress, nonZeroAddress ], {
				unsafeAllow: ["external-library-linking"],
				initializer: "initialize",
			})).to.be.revertedWithCustomError(symmVesting, "ZeroAddress")

			await expect( upgrades.deployProxy(VestingFactory, [nonZeroAddress, nonZeroAddress,
				zeroAddress, nonZeroAddress, nonZeroAddress, nonZeroAddress, nonZeroAddress, nonZeroAddress, nonZeroAddress ], {
				unsafeAllow: ["external-library-linking"],
				initializer: "initialize",
			})).to.be.revertedWithCustomError(symmVesting, "ZeroAddress")

			await expect( upgrades.deployProxy(VestingFactory, [nonZeroAddress, nonZeroAddress,
				nonZeroAddress, zeroAddress, nonZeroAddress, nonZeroAddress, nonZeroAddress, nonZeroAddress, nonZeroAddress ], {
				unsafeAllow: ["external-library-linking"],
				initializer: "initialize",
			})).to.be.revertedWithCustomError(symmVesting, "ZeroAddress")

			await expect( upgrades.deployProxy(VestingFactory, [nonZeroAddress, nonZeroAddress,
				nonZeroAddress, nonZeroAddress, zeroAddress, nonZeroAddress, nonZeroAddress, nonZeroAddress, nonZeroAddress ], {
				unsafeAllow: ["external-library-linking"],
				initializer: "initialize",
			})).to.be.revertedWithCustomError(symmVesting, "ZeroAddress")

			await expect( upgrades.deployProxy(VestingFactory, [nonZeroAddress, nonZeroAddress,
				nonZeroAddress, nonZeroAddress, nonZeroAddress, zeroAddress, nonZeroAddress, nonZeroAddress, nonZeroAddress ], {
				unsafeAllow: ["external-library-linking"],
				initializer: "initialize",
			})).to.be.revertedWithCustomError(symmVesting, "ZeroAddress")

			await expect( upgrades.deployProxy(VestingFactory, [nonZeroAddress, nonZeroAddress,
				nonZeroAddress, nonZeroAddress, nonZeroAddress, nonZeroAddress, zeroAddress, nonZeroAddress, nonZeroAddress ], {
				unsafeAllow: ["external-library-linking"],
				initializer: "initialize",
			})).to.be.revertedWithCustomError(symmVesting, "ZeroAddress")

			await expect( upgrades.deployProxy(VestingFactory, [nonZeroAddress, nonZeroAddress,
				nonZeroAddress, nonZeroAddress, nonZeroAddress, nonZeroAddress, nonZeroAddress, zeroAddress, nonZeroAddress ], {
				unsafeAllow: ["external-library-linking"],
				initializer: "initialize",
			})).to.be.revertedWithCustomError(symmVesting, "ZeroAddress")

			await expect( upgrades.deployProxy(VestingFactory, [nonZeroAddress, nonZeroAddress,
				nonZeroAddress, nonZeroAddress, nonZeroAddress, nonZeroAddress, nonZeroAddress, nonZeroAddress, zeroAddress ], {
				unsafeAllow: ["external-library-linking"],
				initializer: "initialize",
			})).to.be.revertedWithCustomError(symmVesting, "ZeroAddress")
		})

		it('should fail when zero is passed as address to vesting initialize method', async()=>{
			const VestingPlanOps = await ethers.getContractFactory("VestingPlanOps")
			const vestingPlanOps = await VestingPlanOps.deploy()
			await vestingPlanOps.waitForDeployment()

			const zeroAddress = "0x0000000000000000000000000000000000000000";
			const nonZeroAddress = "0x0000000000000000000000000000000000000001";

			const Vesting = await ethers.getContractFactory("SymmVesting", {
				libraries: {
					VestingPlanOps: await vestingPlanOps.getAddress(),
				},
			})
			await expect( upgrades.deployProxy(Vesting, [zeroAddress, nonZeroAddress, nonZeroAddress], {
				unsafeAllow: ["external-library-linking"],
				initializer: "__vesting_init",
			})).to.be.revertedWithCustomError(symmVesting, "NotInitializing")

			await expect( upgrades.deployProxy(Vesting, [nonZeroAddress, nonZeroAddress, zeroAddress], {
				unsafeAllow: ["external-library-linking"],
				initializer: "__vesting_init",
			})).to.be.revertedWithCustomError(symmVesting, "NotInitializing")
		})
	})
	describe('Role management', ()=>{
		it('should allow calling methods just to the ones who have the required role', async()=>{
			await expect(symmVesting.connect(user1).pause()).to.be.revertedWithCustomError(symmVesting, "AccessControlUnauthorizedAccount")
			await expect(symmVesting.connect(user1).unpause()).to.be.revertedWithCustomError(symmVesting, "AccessControlUnauthorizedAccount")
			await expect(symmVesting.connect(user1).resetVestingPlans(symmToken, [await user1.getAddress()], [e(1)]))
				.to.be.revertedWithCustomError(symmVesting, "AccessControlUnauthorizedAccount")
			await expect(symmVesting.connect(user1).setupVestingPlans(
				await symmToken.getAddress(),
				Math.floor(Date.now() / 1000),
				Math.floor(Date.now() / 1000)+3600,
				[await user1.getAddress()],
				[e(1)]
			)).to.be.revertedWithCustomError(symmVesting, "AccessControlUnauthorizedAccount")
			await expect(symmVesting.connect(user1).claimUnlockedTokenFor(symmToken, user1))
				.to.be.revertedWithCustomError(symmVesting, "AccessControlUnauthorizedAccount")
			await expect(symmVesting.connect(user1).claimLockedTokenFor(symmToken, user1, e(1)))
				.to.be.revertedWithCustomError(symmVesting, "AccessControlUnauthorizedAccount")
			await expect(symmVesting.connect(user1).claimLockedTokenForByPercentage(symmToken, user1, e(0.5)))
				.to.be.revertedWithCustomError(symmVesting, "AccessControlUnauthorizedAccount")
		})
	})
}
