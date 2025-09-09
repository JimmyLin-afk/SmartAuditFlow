import { setBalance } from "@nomicfoundation/hardhat-network-helpers"
import { expect } from "chai"
import { Signer } from "ethers"
import { ethers, network } from "hardhat"
import { AirdropHelper, Symmio } from "../typechain-types"

export function shouldBehaveLikeAirdropHelper(): void {
	// Contract instances
	let airdropHelper: AirdropHelper
	let symmToken: Symmio

	// Signers
	let owner: Signer
	let user1: Signer
	let user2: Signer
	let user3: Signer

	// Addresses
	const ADDRESSES = {
		SYMM: "0x800822d361335b4d5F352Dac293cA4128b5B605f",
		OWNER: "0x39E4cdd23Ef0994D38da526da0D2CDfb5b1624f3",
	}

	const MINTER_ROLE = "0x9f2df0fed2c77648de5860a4cc508cd0818c85b8b8a1ab4ceeef8d981c8956a6"
	const INITIAL_BALANCE = ethers.parseEther("1000000")

	async function impersonateAccount(address: string): Promise<Signer> {
		await network.provider.request({
			method: "hardhat_impersonateAccount",
			params: [address],
		})
		return ethers.getImpersonatedSigner(address)
	}

	beforeEach(async () => {
		;[user1, user2, user3] = await ethers.getSigners()

		// Setup owner account
		owner = await impersonateAccount(ADDRESSES.OWNER)
		await setBalance(await owner.getAddress(), INITIAL_BALANCE)

		// Setup contracts
		const TokenFactory = await ethers.getContractFactory("Symmio")
		symmToken = TokenFactory.attach(ADDRESSES.SYMM) as Symmio

		const AirdropHelperFactory = await ethers.getContractFactory("AirdropHelper")
		airdropHelper = await AirdropHelperFactory.connect(owner).deploy()
		await airdropHelper.waitForDeployment()

		// Grant minter role and mint tokens
		await symmToken.connect(owner).grantRole(MINTER_ROLE, await owner.getAddress())
		await symmToken.connect(owner).mint(await airdropHelper.getAddress(), ethers.parseEther("1000"))
	})

	describe("Configuration", () => {
		it("should successfully configure an airdrop", async () => {
			const recipients = [await user1.getAddress(), await user2.getAddress(), await user3.getAddress()]
			const amounts = [ethers.parseEther("10"), ethers.parseEther("20"), ethers.parseEther("30")]

			await expect(airdropHelper.connect(owner).configureAirdrop(recipients, amounts))
				.to.emit(airdropHelper, "AirdropConfigured")
				.withArgs(ethers.parseEther("60"), 3)

			const config = await airdropHelper.getAirdropConfig()
			expect(config.recipients).to.deep.equal(recipients)
			expect(config.amounts.map(a => a.toString())).to.deep.equal(amounts.map(a => a.toString()))
			expect(config.processedIndex).to.equal(0)
		})

		it("should revert when configuring with mismatched arrays", async () => {
			const recipients = [await user1.getAddress()]
			const amounts = [ethers.parseEther("10"), ethers.parseEther("20")]

			await expect(airdropHelper.connect(owner).configureAirdrop(recipients, amounts)).to.be.revertedWithCustomError(
				airdropHelper,
				"ArrayLengthMismatch",
			)
		})

		it("should revert when configuring with empty arrays", async () => {
			await expect(airdropHelper.connect(owner).configureAirdrop([], [])).to.be.revertedWithCustomError(airdropHelper, "EmptyArrays")
		})

		it("should revert when configuring with zero address", async () => {
			await expect(airdropHelper.connect(owner).configureAirdrop([ethers.ZeroAddress], [ethers.parseEther("10")])).to.be.revertedWithCustomError(
				airdropHelper,
				"InvalidRecipient",
			)
		})
	})

	describe("Airdrop Execution", () => {
		beforeEach(async () => {
			// Clear any existing config
			await airdropHelper.connect(owner).clearAirdropConfig()

			// Setup new config for each test
			const recipients = [await user1.getAddress(), await user2.getAddress(), await user3.getAddress()]
			const amounts = [ethers.parseEther("10"), ethers.parseEther("20"), ethers.parseEther("30")]
			await airdropHelper.connect(owner).configureAirdrop(recipients, amounts)
		})

		it("should execute airdrop batch successfully", async () => {
			const initialBalance1 = await symmToken.balanceOf(await user1.getAddress())
			const initialBalance2 = await symmToken.balanceOf(await user2.getAddress())
			const initialBalance3 = await symmToken.balanceOf(await user3.getAddress())

			await expect(airdropHelper.connect(owner).transferAirdrops(2))
				.to.emit(airdropHelper, "AirdropBatchExecuted")
				.withArgs(0, 2, ethers.parseEther("30"))

			const finalBalance1 = await symmToken.balanceOf(await user1.getAddress())
			const finalBalance2 = await symmToken.balanceOf(await user2.getAddress())
			const finalBalance3 = await symmToken.balanceOf(await user3.getAddress())

			expect(finalBalance1 - initialBalance1).to.equal(ethers.parseEther("10"))
			expect(finalBalance2 - initialBalance2).to.equal(ethers.parseEther("20"))
			expect(finalBalance3 - initialBalance3).to.equal(0)
		})

		it("should complete full airdrop successfully", async () => {
			const initialBalance1 = await symmToken.balanceOf(await user1.getAddress())
			const initialBalance2 = await symmToken.balanceOf(await user2.getAddress())
			const initialBalance3 = await symmToken.balanceOf(await user3.getAddress())

			await expect(airdropHelper.connect(owner).transferAirdrops(3)).to.emit(airdropHelper, "AirdropCompleted").withArgs(ethers.parseEther("60"), 3)

			const finalBalance1 = await symmToken.balanceOf(await user1.getAddress())
			const finalBalance2 = await symmToken.balanceOf(await user2.getAddress())
			const finalBalance3 = await symmToken.balanceOf(await user3.getAddress())

			expect(finalBalance1 - initialBalance1).to.equal(ethers.parseEther("10"))
			expect(finalBalance2 - initialBalance2).to.equal(ethers.parseEther("20"))
			expect(finalBalance3 - initialBalance3).to.equal(ethers.parseEther("30"))

			const config = await airdropHelper.getAirdropConfig()
			expect(config.recipients).to.have.length(0)
			expect(config.amounts).to.have.length(0)
			expect(config.processedIndex).to.equal(0)
		})

		it("should complete full airdrop successfully on batch size higher than size", async () => {
			const initialBalance1 = await symmToken.balanceOf(await user1.getAddress())
			const initialBalance2 = await symmToken.balanceOf(await user2.getAddress())
			const initialBalance3 = await symmToken.balanceOf(await user3.getAddress())

			await expect(airdropHelper.connect(owner).transferAirdrops(5)).to.emit(airdropHelper, "AirdropCompleted").withArgs(ethers.parseEther("60"), 3)

			const finalBalance1 = await symmToken.balanceOf(await user1.getAddress())
			const finalBalance2 = await symmToken.balanceOf(await user2.getAddress())
			const finalBalance3 = await symmToken.balanceOf(await user3.getAddress())

			expect(finalBalance1 - initialBalance1).to.equal(ethers.parseEther("10"))
			expect(finalBalance2 - initialBalance2).to.equal(ethers.parseEther("20"))
			expect(finalBalance3 - initialBalance3).to.equal(ethers.parseEther("30"))

			const config = await airdropHelper.getAirdropConfig()
			expect(config.recipients).to.have.length(0)
			expect(config.amounts).to.have.length(0)
			expect(config.processedIndex).to.equal(0)
		})

		it("should complete full airdrop successfully on multiple batches but complete", async () => {
			const initialBalance1 = await symmToken.balanceOf(await user1.getAddress())
			const initialBalance2 = await symmToken.balanceOf(await user2.getAddress())
			const initialBalance3 = await symmToken.balanceOf(await user3.getAddress())

			await airdropHelper.connect(owner).transferAirdrops(1)
			await expect(airdropHelper.connect(owner).transferAirdrops(2)).to.emit(airdropHelper, "AirdropCompleted").withArgs(ethers.parseEther("60"), 3)

			const finalBalance1 = await symmToken.balanceOf(await user1.getAddress())
			const finalBalance2 = await symmToken.balanceOf(await user2.getAddress())
			const finalBalance3 = await symmToken.balanceOf(await user3.getAddress())

			expect(finalBalance1 - initialBalance1).to.equal(ethers.parseEther("10"))
			expect(finalBalance2 - initialBalance2).to.equal(ethers.parseEther("20"))
			expect(finalBalance3 - initialBalance3).to.equal(ethers.parseEther("30"))

			const config = await airdropHelper.getAirdropConfig()
			expect(config.recipients).to.have.length(0)
			expect(config.amounts).to.have.length(0)
			expect(config.processedIndex).to.equal(0)
		})

		it("should complete full airdrop successfully on multiple batches but not complete", async () => {
			const initialBalance1 = await symmToken.balanceOf(await user1.getAddress())
			const initialBalance2 = await symmToken.balanceOf(await user2.getAddress())
			const initialBalance3 = await symmToken.balanceOf(await user3.getAddress())

			await airdropHelper.connect(owner).transferAirdrops(1)
			await airdropHelper.connect(owner).transferAirdrops(1)

			const finalBalance1 = await symmToken.balanceOf(await user1.getAddress())
			const finalBalance2 = await symmToken.balanceOf(await user2.getAddress())
			const finalBalance3 = await symmToken.balanceOf(await user3.getAddress())

			expect(finalBalance1 - initialBalance1).to.equal(ethers.parseEther("10"))
			expect(finalBalance2 - initialBalance2).to.equal(ethers.parseEther("20"))
			expect(finalBalance3 - initialBalance3).to.equal(0)

			const config = await airdropHelper.getAirdropConfig()
			expect(config.recipients).to.have.length(3)
			expect(config.amounts).to.have.length(3)
			expect(config.processedIndex).to.equal(2)
		})

		it("should revert with invalid batch size", async () => {
			await expect(airdropHelper.connect(owner).transferAirdrops(0)).to.be.revertedWithCustomError(airdropHelper, "InvalidBatchSize")
		})
	})

	describe("Admin Functions", () => {
		it("should clear airdrop configuration", async () => {
			const recipients = [await user1.getAddress(), await user2.getAddress(), await user3.getAddress()]
			const amounts = [ethers.parseEther("10"), ethers.parseEther("20"), ethers.parseEther("30")]
			await airdropHelper.connect(owner).configureAirdrop(recipients, amounts)

			await airdropHelper.connect(owner).clearAirdropConfig()

			const config = await airdropHelper.getAirdropConfig()
			expect(config.recipients).to.have.length(0)
			expect(config.amounts).to.have.length(0)
			expect(config.processedIndex).to.equal(0)
		})

		it("should rescue accidentally sent tokens", async () => {
			await expect(airdropHelper.connect(owner).rescueFunds(ADDRESSES.SYMM))
				.to.emit(airdropHelper, "FundsRescued")
				.withArgs(ADDRESSES.SYMM, ethers.parseEther("1000"))
		})

		it("should revert rescuing tokens with zero address", async () => {
			await expect(airdropHelper.connect(owner).rescueFunds(ethers.ZeroAddress)).to.be.revertedWithCustomError(airdropHelper, "ZeroAddress")
		})
	})

	describe("View Functions", () => {
		it("should return correct remaining airdrops count", async () => {
			await airdropHelper.connect(owner).clearAirdropConfig()

			const recipients = [await user1.getAddress(), await user2.getAddress(), await user3.getAddress()]
			const amounts = [ethers.parseEther("10"), ethers.parseEther("20"), ethers.parseEther("30")]

			await airdropHelper.connect(owner).configureAirdrop(recipients, amounts)
			expect(await airdropHelper.getRemainingAirdrops()).to.equal(3)

			await airdropHelper.connect(owner).transferAirdrops(2)
			expect(await airdropHelper.getRemainingAirdrops()).to.equal(1)
		})
	})
}
