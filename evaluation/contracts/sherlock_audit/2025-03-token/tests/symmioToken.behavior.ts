import { loadFixture } from "@nomicfoundation/hardhat-network-helpers"
import { expect } from "chai"
import { Symmio } from "../typechain-types"
import { e } from "../utils"
import { initializeFixture, RunContext } from "./Initialize.fixture"

export function shouldBehaveLikeSymmioToken() {
	let context: RunContext
	let symmToken: Symmio

	beforeEach(async () => {
		context = await loadFixture(initializeFixture)
		symmToken = context.symmioToken
	})

	describe("Deployment", () => {
		it("should have the correct name and symbol", async () => {
			expect(await symmToken.name()).to.equal("SYMMIO")
			expect(await symmToken.symbol()).to.equal("SYMM")
		})

		it("should grant the admin role to the deployer", async () => {
			const hasRole = await symmToken.hasRole(await symmToken.DEFAULT_ADMIN_ROLE(), context.signers.admin.address)
			expect(hasRole).to.be.true
		})
	})

	describe("Minting", () => {
		it("should allow minter to mint tokens", async () => {
			const amount = e(1000)
			await symmToken.connect(context.signers.admin).grantRole(await symmToken.MINTER_ROLE(), context.signers.admin.address)
			await symmToken.connect(context.signers.admin).mint(context.signers.user1.address, amount)

			const user1Balance = await symmToken.balanceOf(context.signers.user1.address)
			expect(user1Balance).to.equal(amount)
		})

		it("should revert when non-minters try to mint tokens", async () => {
			const amount = e(1000) // 1000 tokens
			await expect(symmToken.connect(context.signers.user1).mint(context.signers.user1.address, amount)).to.be.revertedWithCustomError(
				symmToken,
				"AccessControlUnauthorizedAccount",
			)
		})
	})

	describe("Burning", () => {
		it("should allow users to burn their tokens", async () => {
			const amount = e(500) // 500 tokens
			await symmToken.connect(context.signers.admin).grantRole(await symmToken.MINTER_ROLE(), context.signers.admin.address)
			await symmToken.connect(context.signers.admin).mint(context.signers.user1.address, amount)

			const user1BalanceBefore = await symmToken.balanceOf(context.signers.user1.address)
			expect(user1BalanceBefore).to.equal(amount)

			await symmToken.connect(context.signers.user1).burn(amount)

			const user1BalanceAfter = await symmToken.balanceOf(context.signers.user1.address)
			expect(user1BalanceAfter).to.equal(0) // Should be 0 after burning all tokens
		})

		it("should revert when users try to burn more tokens than they have", async () => {
			await expect(symmToken.connect(context.signers.user1).burn(e(1000))).to.be.revertedWithCustomError(symmToken, "ERC20InsufficientBalance")
		})
	})

	describe("Transfers", () => {
		it("should allow users to transfer tokens", async () => {
			const amount = e(1000) // 1000 tokens
			await symmToken.connect(context.signers.admin).grantRole(await symmToken.MINTER_ROLE(), context.signers.admin.address)
			await symmToken.connect(context.signers.admin).mint(context.signers.user1.address, amount)

			const user1BalanceBefore = await symmToken.balanceOf(context.signers.user1.address)
			const user2BalanceBefore = await symmToken.balanceOf(context.signers.user2.address)

			expect(user1BalanceBefore).to.equal(amount)

			await symmToken.connect(context.signers.user1).transfer(context.signers.user2.address, amount)

			const user1BalanceAfter = await symmToken.balanceOf(context.signers.user1.address)
			const user2BalanceAfter = await symmToken.balanceOf(context.signers.user2.address)

			expect(user1BalanceAfter).to.equal(0) // User1 balance should be 0 after transfer
			expect(user2BalanceAfter).to.equal(amount) // User2 should have received the tokens
		})

		it("should revert when transferring more than balance", async () => {
			await expect(symmToken.connect(context.signers.user1).transfer(context.signers.user2.address, e(1000))).to.be.revertedWithCustomError(
				symmToken,
				"ERC20InsufficientBalance",
			)
		})
	})
}
