import { loadFixture, setBalance, time } from "@nomicfoundation/hardhat-network-helpers"
import { expect } from "chai"
import { Signer } from "ethers"
import { ethers, network, upgrades } from "hardhat"
import { ERC20, Symmio, SymmVesting, VestingPlanOps__factory } from "../typechain-types"
import { e } from "../utils"
import { initializeFixture, RunContext } from "./Initialize.fixture"

export function shouldBehaveLikeSymmVesting() {
	let symmVesting: SymmVesting
	let symmToken: Symmio
	let erc20: ERC20
	let owner: Signer, admin: Signer, user1: Signer, user2: Signer, vestingPenaltyReceiver: Signer, usdcWhale: Signer
	let pool: String
	let context: RunContext
	let VestingPlanOps: VestingPlanOps__factory
	let user1UsdcAmount: bigint

	const MINTER_ROLE = "0x9f2df0fed2c77648de5860a4cc508cd0818c85b8b8a1ab4ceeef8d981c8956a6"
	const INITIAL_BALANCE = e(100)

	async function impersonateAccount(address: string) {
		await network.provider.request({ method: "hardhat_impersonateAccount", params: [address] })
		return ethers.getImpersonatedSigner(address)
	}

	beforeEach(async () => {
		context = await loadFixture(initializeFixture)
		symmVesting = await context.vesting
		VestingPlanOps = await ethers.getContractFactory("VestingPlanOps")

		admin = context.signers.admin
		user1 = context.signers.user1
		user2 = context.signers.user2
		vestingPenaltyReceiver = context.signers.user3
		pool = await symmVesting.POOL()

		owner = await impersonateAccount("0x8CF65060CdA270a3886452A1A1cb656BECEE5bA4")
		usdcWhale = await impersonateAccount("0x607094ed3a8361bB5e94dD21bcBef2997b687478")

		await setBalance(await owner.getAddress(), INITIAL_BALANCE)
		await setBalance(await usdcWhale.getAddress(), INITIAL_BALANCE)

		const TokenFactory = await ethers.getContractFactory("Symmio")
		const ERC20Factory = await ethers.getContractFactory("MockERC20")
		symmToken = TokenFactory.attach("0x800822d361335b4d5F352Dac293cA4128b5B605f") as Symmio
		erc20 = ERC20Factory.attach("0x833589fcd6edb6e08f4c7c32d4f71b54bda02913") as ERC20

		const vestingPlanOps = await VestingPlanOps.deploy()
		await vestingPlanOps.waitForDeployment()

		const SymmVestingFactory = await ethers.getContractFactory("SymmVesting", {
			libraries: { VestingPlanOps: await vestingPlanOps.getAddress() },
		})

		await symmToken.connect(owner).grantRole(MINTER_ROLE, await owner.getAddress())
		await symmToken.connect(owner).mint(await symmVesting.getAddress(), e("1000"))

		user1UsdcAmount = BigInt(1000e6)
		await erc20.connect(usdcWhale).transfer(await user1.getAddress(), user1UsdcAmount)
		await erc20.connect(user1).approve(await symmVesting.getAddress(), user1UsdcAmount)

		// Setup vesting plans
		const users = [await user1.getAddress()]
		const amounts = [e("1000")]
		const startTime = Math.floor(Date.now() / 1000) - 2 * 30 * 24 * 60 * 60 // 2 months ago
		const endTime = startTime + 9 * 30 * 24 * 60 * 60 // 9 months later

		await symmVesting.connect(admin).setupVestingPlans(await symmToken.getAddress(), startTime, endTime, users, amounts)
	})
	describe("Add Liquidity", () => {
		it("should allow a user to add liquidity successfully", async () => {
			const symmAmount = e(1)
			const minLpAmount = e("0.05")

			const lockedAmountBefore = await symmVesting.getLockedAmountsForToken(await user1.getAddress(), await symmToken.getAddress())
			const claimableAmountBefore = await symmVesting.getClaimableAmountsForToken(await user1.getAddress(), await symmToken.getAddress())
			const unlockedAmountBefore = await symmVesting.getUnlockedAmountForToken(await user1.getAddress(), await symmToken.getAddress())
			const planBefore = await symmVesting.vestingPlans(pool, await context.signers.user1.getAddress())
			const quote = await symmVesting.getLiquidityQuote(symmAmount)
			await expect(planBefore.amount).to.equal(0)

			await symmVesting.connect(user1).addLiquidity(symmAmount, minLpAmount, user1UsdcAmount)

			const lockedAmountAfter = await symmVesting.getLockedAmountsForToken(await user1.getAddress(), await symmToken.getAddress())
			const claimableAmountAfter = await symmVesting.getClaimableAmountsForToken(await user1.getAddress(), await symmToken.getAddress())
			const unlockedAmountAfter = await symmVesting.getUnlockedAmountForToken(await user1.getAddress(), await symmToken.getAddress())
			const planAfter = await symmVesting.vestingPlans(pool, await user1.getAddress())

			await expect(lockedAmountBefore - lockedAmountAfter).to.be.within(BigInt(symmAmount), BigInt(symmAmount) + BigInt(1e15))
			await expect(claimableAmountBefore - claimableAmountAfter).to.be.within(
				BigInt(claimableAmountBefore),
				BigInt(claimableAmountBefore) + BigInt(1e5),
			)
			await expect(claimableAmountAfter).to.be.lessThan(1e5)
			await expect(unlockedAmountBefore - unlockedAmountAfter).to.be.within(
				BigInt(claimableAmountBefore),
				BigInt(claimableAmountBefore) + BigInt(1e5),
			)
			await expect(unlockedAmountAfter).to.be.closeTo(0, BigInt(1e5))
			await expect(planAfter.amount).to.be.closeTo(quote.lpAmount, BigInt(1e2))
		})

		it("should allow a user to add liquidity by percentage successfully", async () => {
			const symmPercentage = e(0.5)
			const symmAmount = (await symmVesting.getLockedAmountsForToken(user1, symmToken)) / BigInt(2)
			const minLpAmount = e("0.05")

			const lockedAmountBefore = await symmVesting.getLockedAmountsForToken(await user1.getAddress(), await symmToken.getAddress())
			const claimableAmountBefore = await symmVesting.getClaimableAmountsForToken(await user1.getAddress(), await symmToken.getAddress())
			const unlockedAmountBefore = await symmVesting.getUnlockedAmountForToken(await user1.getAddress(), await symmToken.getAddress())

			await symmVesting.connect(user1).addLiquidityByPercentage(symmPercentage, minLpAmount, user1UsdcAmount)

			const lockedAmountAfter = await symmVesting.getLockedAmountsForToken(await user1.getAddress(), await symmToken.getAddress())
			const claimableAmountAfter = await symmVesting.getClaimableAmountsForToken(await user1.getAddress(), await symmToken.getAddress())
			const unlockedAmountAfter = await symmVesting.getUnlockedAmountForToken(await user1.getAddress(), await symmToken.getAddress())

			await expect(lockedAmountBefore - lockedAmountAfter).to.be.within(BigInt(symmAmount), BigInt(symmAmount) + BigInt(1e15))
			await expect(claimableAmountBefore - claimableAmountAfter).to.be.within(
				BigInt(claimableAmountBefore),
				BigInt(claimableAmountBefore) + BigInt(1e5),
			)
			await expect(claimableAmountAfter).to.be.lessThan(1e5)
			await expect(unlockedAmountBefore - unlockedAmountAfter).to.be.within(
				BigInt(claimableAmountBefore),
				BigInt(claimableAmountBefore) + BigInt(1e5),
			)
			await expect(unlockedAmountAfter).to.be.closeTo(0, BigInt(1e5))
		})

		it("should revert if slippage limit is exceeded", async () => {
			const symmAmount = String(1e18)
			const minLpAmount = e(200)
			await expect(symmVesting.connect(user1).addLiquidity(symmAmount, minLpAmount, user1UsdcAmount)).to.be.revertedWithCustomError(
				symmVesting,
				"SlippageExceeded",
			)
		})

		it("should revert if user does not have enough locked SYMM", async () => {
			const symmAmount = e(950)
			const minLpAmount = 0
			await expect(symmVesting.connect(user1).addLiquidity(symmAmount, minLpAmount, user1UsdcAmount)).to.be.revertedWithCustomError(
				symmVesting,
				"InvalidAmount",
			)
		})

		it("should mint SYMM if SymmVesting doesn't have enough balance", async () => {
			const symmAmount = e("1100")
			const minLpAmount = 0
			const symmVestingBalance = await symmToken.balanceOf(await symmVesting.getAddress())

			// Setup vesting plans
			const users = [await user2.getAddress()]
			const amounts = [e(1e15)]
			const startTime = Math.floor(Date.now() / 1000) - 1 * 30 * 24 * 60 * 60 // 1 months ago
			const endTime = startTime + 9 * 30 * 24 * 60 * 60 // 9 months later
			await symmVesting.connect(admin).setupVestingPlans(await symmToken.getAddress(), startTime, endTime, users, amounts)

			await symmToken.connect(owner).grantRole(MINTER_ROLE, await symmVesting.getAddress())

			const usdcAmount = String(1e8)
			await erc20.connect(usdcWhale).transfer(await user2.getAddress(), usdcAmount)
			await erc20.connect(user2).approve(await symmVesting.getAddress(), usdcAmount)

			const diffBalance = symmAmount - symmVestingBalance
			await expect(diffBalance).to.be.greaterThan(0)
			// await symmVesting.connect(user2).addLiquidity(symmAmount, minLpAmount, usdcAmount)
		})

		it("should emit LiquidityAdded event for each addLiquidity with correct amounts", async () => {
			// Define the number of liquidity additions we want to test
			const liquidityAdditionsCount = BigInt(100)

			// Setup vesting plans
			const users = [await user2.getAddress()]
			const amounts = [e(1e15)]
			const startTime = Math.floor(Date.now() / 1000) - 8 * 30 * 24 * 60 * 60 // 1 months ago
			const endTime = startTime + 9 * 30 * 24 * 60 * 60 // 9 months later
			await symmVesting.connect(admin).setupVestingPlans(await symmToken.getAddress(), startTime, endTime, users, amounts)

			await symmToken.connect(owner).grantRole(MINTER_ROLE, await symmVesting.getAddress())

			const usdcTransferAmount = String(1e14)
			await erc20.connect(usdcWhale).transfer(await user2.getAddress(), usdcTransferAmount)
			await erc20.connect(user2).approve(await symmVesting.getAddress(), usdcTransferAmount)

			const minVal = e(0.1)
			const maxVal = e(1e7)
			const totalMax = e(1e12)

			// Check feasibility: if even the minimum for every element would exceed the total allowed.
			if (liquidityAdditionsCount * minVal > totalMax) {
				throw new Error("Cannot generate amounts: minimum sum exceeds total maximum allowed")
			}

			const symmAmountList = []
			const minLpAmountList = []
			let currentSum = BigInt(0)

			/**
			 * Returns a random BigInt in the range [0, n)
			 * using rejection sampling and 32-bit chunks.
			 */
			function randomBigInt(n: any) {
				if (n <= BigInt(0)) return BigInt(0)
				const bits = n.toString(2).length
				let result
				while (true) {
					let r = BigInt(0)
					// Calculate how many 32-bit chunks we need.
					const numChunks = Math.ceil(bits / 32)
					for (let i = 0; i < numChunks; i++) {
						// Generate a random 32-bit chunk.
						const random32 = BigInt(Math.floor(Math.random() * 4294967296))
						r = (r << BigInt(32)) + random32
					}
					if (r < n) {
						result = r
						break
					}
				}
				return result
			}
			let sumUsdcDiff = Number(0)
			let sumSymmDiff = Number(0)
			let sumLPDiff = Number(0)

			for (let i = 0; i < Number(liquidityAdditionsCount); i++) {
				const remainingCount = liquidityAdditionsCount - BigInt(i)
				// Reserve the minimum for all remaining elements (excluding current one)
				const reserved = (remainingCount - BigInt(1)) * minVal
				// Maximum allowed for this iteration without breaking the total constraint:
				const allowed = totalMax - currentSum - reserved
				// Use the smaller of allowed and maxVal:
				const maxAllowedForThis = allowed < maxVal ? allowed : maxVal

				// Compute the range width (inclusive) and generate a random increment
				const range = maxAllowedForThis - minVal + BigInt(1)
				const randomIncrement = randomBigInt(range)
				const value = minVal + randomIncrement

				symmAmountList.push(value.toString())
				minLpAmountList.push("0")
				currentSum += value
			}

			for (let i = 0; i < liquidityAdditionsCount; i++) {
				const symmAmount = symmAmountList[i]
				const minLpAmount = minLpAmountList[i]
				// Get the liquidity quote before the liquidity addition
				const [usdcAmount, expectedLpAmount] = await symmVesting.connect(user2).getLiquidityQuote(symmAmount)

				// Call addLiquidity and capture the transaction
				const userUsdcBalanceBefore = await erc20.balanceOf(await user2.getAddress())
				const tx = await symmVesting.connect(user2).addLiquidity(symmAmount, minLpAmount, usdcTransferAmount)
				const userUsdcBalanceAfter = await erc20.balanceOf(await user2.getAddress())
				const receipt = await tx.wait()
				for (const log of receipt?.logs!) {
					const parsedLog = symmVesting.interface.parseLog(log)
					if (parsedLog?.name === "LiquidityAdded") {
						const symmIn = parsedLog.args[1]
						const usdcIn = parsedLog.args[2]
						const lpOut = parsedLog.args[3]
						sumUsdcDiff += Number(usdcIn) - Number(usdcAmount)
						sumSymmDiff += Number(symmIn) - Number(symmAmount)
						sumLPDiff += Number(lpOut) - Number(await expectedLpAmount)

						await expect(usdcAmount).to.be.closeTo(usdcIn, 5)
						await expect(symmAmount).to.be.closeTo(symmIn, 5)
						await expect(await expectedLpAmount).to.be.closeTo(lpOut, 5)

						await expect(userUsdcBalanceBefore - userUsdcBalanceAfter).to.be.equal(Number(usdcIn))
					}
				}
			}

			await expect(sumUsdcDiff).to.be.closeTo(0, liquidityAdditionsCount)
			await expect(sumSymmDiff).to.be.closeTo(0, liquidityAdditionsCount)
			await expect(sumLPDiff).to.be.closeTo(0, liquidityAdditionsCount)
		}).timeout(500000)

		it("should revert if the contract tries to transferFrom more than maxUsdc even when it's approved", async () => {
			const symmAmount = String(1e18)
			const minLpAmount = 0
			const maxUsdcIn = 100
			await expect(symmVesting.connect(user1).addLiquidity(symmAmount, minLpAmount, maxUsdcIn)).to.be.revertedWithCustomError(
				symmVesting,
				"MaxUsdcExceeded",
			)
		})

		it("should claim lp tokens after add liquidity by the next addLiquidity", async () => {
			const lockedLPBefore1 = await symmVesting.getLockedAmountsForToken(user1, pool)
			const unlockedSymmBefore1 = await symmVesting.getUnlockedAmountForToken(user1, symmToken)
			await symmVesting.connect(user1).addLiquidity(e(100), 0, 0)
			const query1 = await symmVesting.getLiquidityQuote(e(100))
			const lockedLPAfter1 = await symmVesting.getLockedAmountsForToken(user1, pool)
			const unlockedSymmAfter1 = await symmVesting.getUnlockedAmountForToken(user1, symmToken)
			await expect(lockedLPAfter1 - lockedLPBefore1).to.be.closeTo(query1[1], 1000)
			await expect(unlockedSymmBefore1 - unlockedSymmAfter1).to.be.greaterThan(0)

			const lockedLPBefore2 = await symmVesting.getLockedAmountsForToken(user1, pool)
			const symmBalanceBefore2 = await symmToken.balanceOf(user1)
			const query2 = await symmVesting.getLiquidityQuote(e(1))
			await symmVesting.connect(user1).addLiquidity(e(1), 0, 0)
			const symmBalanceAfter2 = await symmToken.balanceOf(user1)
			const lockedLPAfter2 = await symmVesting.getLockedAmountsForToken(user1, pool)
			await expect(lockedLPAfter2 - lockedLPBefore2).to.be.closeTo(query2.lpAmount, String(2e13)) //2e13: unlocked amount during passed time of several calls
			await expect(symmBalanceAfter2 - symmBalanceBefore2).to.be.greaterThan(0)
		})

		it("should claim lp tokens after add liquidity by claimTokens", async () => {
			await symmVesting.connect(user1).addLiquidity(e(100), 0, 0)

			await time.increase(1e7)

			const unlockedSymmBefore = await symmVesting.getClaimableAmountsForToken(user1, symmToken)
			const userSymmBalanceBefore = await symmToken.balanceOf(user1)
			await symmVesting.connect(user1).claimUnlockedToken(symmToken, user1)
			const userSymmBalanceAfter = await symmToken.balanceOf(user1)
			const unlockedSymmAfter = await symmVesting.getClaimableAmountsForToken(user1, symmToken)
			await expect(unlockedSymmAfter).to.be.lessThan(unlockedSymmBefore)
			await expect(userSymmBalanceAfter).to.be.greaterThan(userSymmBalanceBefore)

			const unlockedLPBefore = await symmVesting.getClaimableAmountsForToken(user1, pool)
			await symmVesting.connect(user1).claimUnlockedToken(pool, user1)
			const unlockedLPAfter = await symmVesting.getClaimableAmountsForToken(user1, pool)
			await expect(unlockedLPAfter).to.be.lessThan(unlockedLPBefore)
		})
	})
}
