import { loadFixture } from "@nomicfoundation/hardhat-network-helpers"
import { expect } from "chai"
import { SymmAllocationClaimer, Symmio } from "../typechain-types"
import { initializeFixture, RunContext } from "./Initialize.fixture"
import { e } from "../utils"
import { AddressLike, ZeroAddress } from "ethers"

export function shouldBehaveLikeSymmAllocationClaimer() {
	let context: RunContext
	let symmClaim: SymmAllocationClaimer
	let symmToken: Symmio

	const ROLES = {
		DEFAULT_ADMIN: "DEFAULT_ADMIN_ROLE",
		BALANCER: "SETTER_ROLE",
		PAUSER: "PAUSER_ROLE",
		SETTER: "MINTER_ROLE",
		UNPAUSER: "UNPAUSER_ROLE",
	} as const

	beforeEach(async () => {
		context = await loadFixture(initializeFixture)
		symmClaim = context.claimSymm
		symmToken = context.symmioToken
	})

	// Deployment tests
	describe("Deployment", () => {
		it("Should deploy the contract successfully", async () => {
			expect(await symmClaim.getAddress()).to.be.properAddress
		})

		it("Should correctly grant roles during deployment", async () => {
			for (const role of Object.values(ROLES)) {
				const hasRole = await symmClaim.hasRole(await symmClaim[role](), await context.signers.admin.getAddress())
				expect(hasRole).to.be.true
			}
		})
	})

	// Configuration tests
	describe("Configuration", () => {
		it("Should correctly pause and unpause the contract", async () => {
			expect(await symmClaim.paused()).to.be.false
			await symmClaim.connect(context.signers.admin).pause()
			expect(await symmClaim.paused()).to.be.true
			await symmClaim.connect(context.signers.admin).unpause()
			expect(await symmClaim.paused()).to.be.false
		})
	})

	// Allocation setting tests
	describe("Setting Allocations", () => {
		it("Should set batch allocations for users correctly", async () => {
			const users = [context.signers.user1.address, context.signers.user2.address]
			const allocations = [e(1000), e(2000)] // Allocations in 18 decimals

			await symmClaim.connect(context.signers.setter).setBatchAllocations(users, allocations)

			expect(await symmClaim.userAllocations(context.signers.user1.address)).to.equal(e(1000))
			expect(await symmClaim.userAllocations(context.signers.user2.address)).to.equal(e(2000))
			expect(await symmClaim.totalAllocation()).to.equal(e(3000))
		})

		it("Should revert if a non-setter attempts to set allocations", async () => {
			const users = [context.signers.user1.address, context.signers.user2.address]
			const allocations = [e(1000), e(2000)]

			await expect(symmClaim.connect(context.signers.user1).setBatchAllocations(users, allocations)).to.be.revertedWithCustomError(
				symmClaim,
				"AccessControlUnauthorizedAccount",
			)
		})

		it("Should revert if the total allocation exceeds the maximum issuable tokens", async () => {
			const users = [context.signers.user1.address]
			const allocations = [e(401000000)] // Exceeds MAX_ISSUABLE_TOKEN

			await expect(symmClaim.connect(context.signers.setter).setBatchAllocations(users, allocations)).to.be.revertedWithCustomError(
				symmClaim,
				"TotalAllocationExceedsMax",
			)
		})

		it("Should revert if user and allocation array lengths do not match", async () => {
			const users = [context.signers.user1.address, context.signers.user2.address]
			const allocations = [e(1000)]

			await expect(symmClaim.connect(context.signers.setter).setBatchAllocations(users, allocations)).to.be.revertedWithCustomError(
				symmClaim,
				"ArrayLengthMismatch",
			)
		})

		it("Should revert if any account in the batch is a zero address", async () => {
			const users = [context.signers.user1.address, ZeroAddress]
			const allocations = [e(1000), e(1000)]

			await expect(symmClaim.connect(context.signers.setter).setBatchAllocations(users, allocations)).to.be.revertedWithCustomError(
				symmClaim,
				"ZeroAddress",
			)
		})

		it("Should revert if input arrays are empty", async () => {
			const users: AddressLike[] = []
			const allocations: bigint[] = []

			await expect(symmClaim.connect(context.signers.setter).setBatchAllocations(users, allocations)).to.be.revertedWithCustomError(
				symmClaim,
				"EmptyArrays",
			)
		})
	})

	// Claiming allocations tests
	describe("Claiming Allocations", () => {
		beforeEach(async () => {
			const users = [context.signers.user1.address]
			const allocations = [e(1000)]
			await symmClaim.connect(context.signers.setter).setBatchAllocations(users, allocations)
		})

		it("Should allow users to claim their allocations", async () => {
			const user1InitialBalance = await symmToken.balanceOf(context.signers.user1.address)
			await symmClaim.connect(context.signers.user1).claim()

			const user1FinalBalance = await symmToken.balanceOf(context.signers.user1.address)
			expect(user1FinalBalance - user1InitialBalance).to.equal(e(500)) // Adjusted for mintFactor

			expect(await symmClaim.userAllocations(context.signers.user1.address)).to.equal(e(0))
			expect(await symmClaim.totalAllocation()).to.equal(e(1000))
		})

		it("Should revert if a user with no allocation attempts to claim", async () => {
			await expect(symmClaim.connect(context.signers.user2).claim()).to.be.revertedWithCustomError(symmClaim, "UserHasNoAllocation")
		})
	})

	// Admin claims tests
	describe("Admin Claims", () => {
		it("Should allow the admin to claim tokens for the foundation", async () => {
			const users = [context.signers.user1.address]
			const allocations = [e(1000)]
			await symmClaim.connect(context.signers.setter).setBatchAllocations(users, allocations)
			await symmClaim.connect(context.signers.user1).claim()

			await symmClaim.connect(context.signers.admin).adminClaim(e(500)) // Admin claims tokens
			const foundationBalance = await symmToken.balanceOf(context.signers.symmioFoundation.address)
			expect(foundationBalance).to.equal(e(500))
		})

		it("Should revert if the admin attempts to claim more tokens than available", async () => {
			await expect(symmClaim.connect(context.signers.admin).adminClaim(e(1000))).to.be.revertedWithCustomError(
				symmClaim,
				"AdminClaimAmountExceedsAvailable",
			)
		})
	})

	// Symmio Foundation address tests
	describe("Set Symmio Foundation Address", () => {
		it("Should allow the setter to update the Symmio Foundation address", async () => {
			const newAddress = await context.signers.user2.getAddress()

			await symmClaim.connect(context.signers.setter).setSymmioFoundationAddress(newAddress)

			expect(await symmClaim.symmioFoundationAddress()).to.equal(newAddress)
		})

		it("Should revert if a non-setter attempts to update the Symmio Foundation address", async () => {
			const newAddress = await context.signers.user2.getAddress()

			await expect(symmClaim.connect(context.signers.user1).setSymmioFoundationAddress(newAddress)).to.be.revertedWithCustomError(
				symmClaim,
				"AccessControlUnauthorizedAccount",
			)
		})

		it("Should revert if the new Symmio Foundation address is the zero address", async () => {
			await expect(symmClaim.connect(context.signers.setter).setSymmioFoundationAddress(ZeroAddress)).to.be.revertedWithCustomError(
				symmClaim,
				"ZeroAddress",
			)
		})
	})
}
