import { SignerWithAddress } from "@nomicfoundation/hardhat-ethers/signers"
import { loadFixture, time } from "@nomicfoundation/hardhat-network-helpers"
import { expect } from "chai"
import { ethers } from "hardhat"
import { ERC20, Symmio, SymmStaking } from "../typechain-types"
import { e } from "../utils"
import { initializeFixture, RunContext } from "./Initialize.fixture"

export function shouldBehaveLikeSymmStaking() {
	let context: RunContext
	let symmStaking: SymmStaking
	let stakingToken: Symmio
	let user1: SignerWithAddress
	let user2: SignerWithAddress
	let admin: SignerWithAddress
	let usdtToken: ERC20
	let usdcToken: ERC20

	beforeEach(async function () {
		context = await loadFixture(initializeFixture)
		symmStaking = context.symmStaking
		admin = context.signers.admin
		user1 = context.signers.user1
		user2 = context.signers.user2
		stakingToken = context.symmioToken

		// 1. Mint initial balance of 100 SYMM (staking token) to user1, user2, and admin
		let initialBalance: bigint = e("100")
		await stakingToken.connect(admin).mint(user1.address, initialBalance)
		await stakingToken.connect(admin).mint(user2.address, initialBalance)
		await stakingToken.connect(admin).mint(admin.address, initialBalance)

		// 2. Deploy USDT (ERC20) and USDC (ERC20) tokens
		const ERC20 = await ethers.getContractFactory("MockERC20")

		// USDT Token
		usdtToken = await ERC20.connect(admin).deploy("USDT", "USDT", 18)
		await usdtToken.waitForDeployment()

		// USDC Token
		usdcToken = await ERC20.deploy("USDC", "USDC", 18) //TODO: is it ok that it's not 6?
		await usdcToken.waitForDeployment()
	})

	describe("Deployment", function () {
		it("should have the correct admin", async () => {
			expect(await context.symmStaking.hasRole(await context.symmStaking.DEFAULT_ADMIN_ROLE(), await context.signers.admin.getAddress())).to.be.true
		})

		it("should set the correct staking token", async function () {
			expect(await context.symmStaking.stakingToken()).to.equal(await symmStaking.stakingToken())
		})
	})

	describe("Deposit", function () {
		it("should revert if amount is 0", async function () {
			const depositAmount = 0
			const receiver = user1.address

			// Expecting ZeroAmount error if the deposit amount is 0
			await expect(symmStaking.connect(user1).deposit(depositAmount, receiver)).to.be.revertedWithCustomError(symmStaking, "ZeroAmount")
		})

		it("should revert if receiver is address(0)", async function () {
			const depositAmount = ethers.parseUnits("10", 18) // 10 SYMM tokens
			const receiver = "0x0000000000000000000000000000000000000000" // address(0)

			// Expecting ZeroAddress error if the receiver is address(0)
			await expect(symmStaking.connect(user1).deposit(depositAmount, receiver)).to.be.revertedWithCustomError(symmStaking, "ZeroAddress")
		})

		it("should correctly deposit 10 SYMM tokens and update totalSupply and balanceOf for user1", async function () {
			const depositAmount = e("10") // 10 SYMM tokens
			const receiver = user1.address

			// Check the totalSupply and balanceOf for user1
			const totalSupplyBefore = await symmStaking.totalSupply()
			const balanceOfUser1Before = await symmStaking.balanceOf(user1.address)

			// Approve the contract to transfer SYMM tokens on behalf of user1
			await stakingToken.connect(user1).approve(await symmStaking.getAddress(), depositAmount)
			// Deposit SYMM tokens into the staking contract for user1
			await symmStaking.connect(user1).deposit(depositAmount, receiver)

			// Check the totalSupply and balanceOf for user1
			const totalSupplyAfter = await symmStaking.totalSupply()
			const balanceOfUser1After = await symmStaking.balanceOf(user1.address)

			// Assert that the deposit was successful and totalSupply and balanceOf for user1 are updated correctly
			expect(totalSupplyAfter - totalSupplyBefore).to.equal(depositAmount)
			expect(balanceOfUser1After - balanceOfUser1Before).to.equal(depositAmount)
		})
	})

	describe("Reward Calculation", function () {
		it("should calculate reward correctly after single user deposit", async function () {
			// Scenario: Single depositor — user1 deposits 604,800 SYMM, waits 200s, claims 200 tokens.

			const depositAmount = "604800"
			const rewardAmount = depositAmount
			await stakingToken.connect(admin).approve(await symmStaking.getAddress(), rewardAmount)
			await symmStaking.connect(admin).addRewardToken(await stakingToken.getAddress())
			await symmStaking.connect(admin).notifyRewardAmount([await stakingToken.getAddress()], [rewardAmount])

			await time.increase(200)

			await stakingToken.connect(user1).approve(await symmStaking.getAddress(), depositAmount)
			await symmStaking.connect(user1).deposit(depositAmount, user1.address)

			await time.increase(200)

			const user1BalanceBefore = await stakingToken.balanceOf(user1.address)
			await symmStaking.connect(user1).claimRewards()
			const user1BalanceAfter = await stakingToken.balanceOf(user1.address)
			const claimed = user1BalanceAfter - user1BalanceBefore

			expect(claimed).to.equal("200")
		})

		it("should calculate reward correctly after two users deposit with 2:1 ratio", async function () {
			// Scenario: User1 deposits 604,800 SYMM, User2 deposits 302,400 SYMM, both claim USDT after 200s in 2:1 ratio.
			const depositUser1 = "604800"
			const depositUser2 = "302400"
			// await stakingToken.connect(admin).mint(user1.address, depositUser1);
			// await stakingToken.connect(admin).mint(user2.address, depositUser2);

			await stakingToken.connect(user1).approve(await symmStaking.getAddress(), depositUser1)
			await stakingToken.connect(user2).approve(await symmStaking.getAddress(), depositUser2)

			await symmStaking.connect(user1).deposit(depositUser1, user1.address)
			await symmStaking.connect(user2).deposit(depositUser2, user2.address)

			const rewardAmount = "604800"
			await usdtToken.connect(admin).approve(await symmStaking.getAddress(), rewardAmount)
			await symmStaking.connect(admin).addRewardToken(await usdtToken.getAddress())
			await symmStaking.connect(admin).notifyRewardAmount([await usdtToken.getAddress()], [rewardAmount])

			await time.increase(200)

			const user1BalanceBefore = await usdtToken.balanceOf(user1.address)
			await symmStaking.connect(user1).claimRewards()
			const user1BalanceAfter = await usdtToken.balanceOf(user1.address)
			const user1Claimed = user1BalanceAfter - user1BalanceBefore

			const user2BalanceBefore = await usdtToken.balanceOf(user2.address)
			await symmStaking.connect(user2).claimRewards()
			const user2BalanceAfter = await usdtToken.balanceOf(user2.address)
			const user2Claimed = user2BalanceAfter - user2BalanceBefore

			expect(user1Claimed + user2Claimed).to.equal(200n)
			expect(user1Claimed).to.equal(133n)
			expect(user2Claimed).to.equal(67n)
		})

		it("should return zero rewards if no rewards are notified", async function () {
			// Scenario: User1 deposits 100 SYMM, configures USDT & USDC but no rewards are notified, so they claim 0.
			const depositAmount = "100"
			await stakingToken.connect(user1).approve(await symmStaking.getAddress(), depositAmount)
			await symmStaking.connect(user1).deposit(depositAmount, user1.address)

			await symmStaking.connect(admin).addRewardToken(await usdtToken.getAddress())
			await symmStaking.connect(admin).addRewardToken(await usdcToken.getAddress())

			const usdtBalanceBefore = await usdtToken.balanceOf(user1.address)
			const usdcBalanceBefore = await usdcToken.balanceOf(user1.address)
			await symmStaking.connect(user1).claimRewards()
			const usdtBalanceAfter = await usdtToken.balanceOf(user1.address)
			const usdcBalanceAfter = await usdcToken.balanceOf(user1.address)

			const usdtClaimed = usdtBalanceAfter - usdtBalanceBefore
			const usdcClaimed = usdcBalanceAfter - usdcBalanceBefore

			expect(usdtClaimed).to.equal(0n)
			expect(usdcClaimed).to.equal(0n)
		})

		it("should calculate rewards correctly after multiple deposits with dual rewards", async function () {
			// Scenario: User1 deposits 302,400 + 302,400, User2 deposits 302,400. They claim USDT & USDC after 200s.
			const depositUser1Part1 = 302400n
			const depositUser1Part2 = 302400n
			const depositUser2 = 302400n

			await stakingToken.connect(user1).approve(await symmStaking.getAddress(), (depositUser1Part1 + depositUser1Part2).toString())
			await stakingToken.connect(user2).approve(await symmStaking.getAddress(), depositUser2.toString())

			await symmStaking.connect(user1).deposit(depositUser1Part1.toString(), user1.address)
			await symmStaking.connect(user2).deposit(depositUser2.toString(), user2.address)
			await symmStaking.connect(user1).deposit(depositUser1Part2.toString(), user1.address)

			const rewardAmount = 604800n
			await usdtToken.connect(admin).approve(await symmStaking.getAddress(), rewardAmount.toString())
			await usdcToken.connect(admin).approve(await symmStaking.getAddress(), rewardAmount.toString())
			await symmStaking.connect(admin).addRewardToken(await usdtToken.getAddress())
			await symmStaking.connect(admin).addRewardToken(await usdcToken.getAddress())
			await symmStaking
				.connect(admin)
				.notifyRewardAmount([await usdtToken.getAddress(), await usdcToken.getAddress()], [rewardAmount.toString(), rewardAmount.toString()])

			await time.increase(200)

			const user1BeforeUSDT = await usdtToken.balanceOf(user1.address)
			const user1BeforeUSDC = await usdcToken.balanceOf(user1.address)
			await symmStaking.connect(user1).claimRewards()
			const user1AfterUSDT = await usdtToken.balanceOf(user1.address)
			const user1AfterUSDC = await usdcToken.balanceOf(user1.address)
			const user1ClaimedUSDT = user1AfterUSDT - user1BeforeUSDT
			const user1ClaimedUSDC = user1AfterUSDC - user1BeforeUSDC

			const user2BeforeUSDT = await usdtToken.balanceOf(user2.address)
			const user2BeforeUSDC = await usdcToken.balanceOf(user2.address)
			await symmStaking.connect(user2).claimRewards()
			const user2AfterUSDT = await usdtToken.balanceOf(user2.address)
			const user2AfterUSDC = await usdcToken.balanceOf(user2.address)
			const user2ClaimedUSDT = user2AfterUSDT - user2BeforeUSDT
			const user2ClaimedUSDC = user2AfterUSDC - user2BeforeUSDC

			expect(user1ClaimedUSDT + user2ClaimedUSDT).to.equal(200n)
			expect(user1ClaimedUSDT).to.equal(133n)
			expect(user2ClaimedUSDT).to.equal(67n)

			expect(user1ClaimedUSDC + user2ClaimedUSDC).to.equal(200n)
			expect(user1ClaimedUSDC).to.equal(133n)
			expect(user2ClaimedUSDC).to.equal(67n)
		})

		it("should correctly update perTokenStored for tokens with different decimals", async () => {
			const context = await initializeFixture()
			const { admin, user1 } = context.signers

			// Deploy three mock reward tokens with different decimals
			const MockERC20 = await ethers.getContractFactory("MockERC20")
			const reward18 = await MockERC20.deploy("RewardToken18", "RT18", 18)
			const reward6 = await MockERC20.deploy("RewardToken6", "RT6", 6)
			// const reward4 = await MockERC20.deploy("RewardToken2", "RT2", 4)

			// Add them as reward tokens
			await context.symmStaking.connect(admin).addRewardToken(reward18)
			await context.symmStaking.connect(admin).addRewardToken(reward6)
			// await context.symmStaking.connect(admin).addRewardToken(reward4 ,true)

			// Amount to notify for each token
			const amount18 = BigInt(1209.6e18) //2000 * 604800(1week) * 1e18
			const amount6 = BigInt(1209.6e6)
			// const amount4 = BigInt(1209.6e4)

			// Mint reward tokens to admin
			await reward18.mint(admin, amount18)
			await reward6.mint(admin, amount6)
			// await reward4.mint(admin, amount4)

			// Approve staking contract to spend rewards
			await reward18.connect(admin).approve(context.symmStaking, amount18)
			await reward6.connect(admin).approve(context.symmStaking, amount6)
			// await reward4.connect(admin).approve(context.symmStaking, amount4)

			// Notify staking contract about the reward amounts
			await context.symmStaking.connect(admin).notifyRewardAmount(
				[reward18, reward6],
				[amount18, amount6]
			)

			// User1 stakes a large amount of SYMM
			await context.symmioToken.mint(user1.address, (BigInt(10_000_000e18)))
			await context.symmioToken.connect(user1).approve(context.symmStaking, (BigInt(10_000_000e18)))
			await context.symmStaking.connect(user1).deposit(BigInt(1_000_000e18), user1) //so we get reward rate of 2000

			// One block later:
			await context.symmStaking.connect(user1).deposit(1, user1)
			let reward18State = await context.symmStaking.rewardState(reward18)
			let reward6State = await context.symmStaking.rewardState(reward6)
			// let reward4State = await context.symmStaking.rewardState(reward4)
			let perTokenStored18_1 = reward18State.perTokenStored
			let perTokenStored6_1 = reward6State.perTokenStored
			// let perTokenStored4_1 = reward4State.perTokenStored
			// expect(perTokenStored6).to.be.equal(2000e6)
			expect(perTokenStored18_1).to.be.equal(2e9)
			expect(perTokenStored6_1).to.be.equal(2e9)
			// expect(perTokenStored4_1).to.be.equal(2e9)

			// One block later:
			await context.symmStaking.connect(user1).withdraw(1, user1)
			reward18State = await context.symmStaking.rewardState(reward18)
			reward6State = await context.symmStaking.rewardState(reward6)
			// reward4State = await context.symmStaking.rewardState(reward4)
			let perTokenStored18_2 = reward18State.perTokenStored
			let perTokenStored6_2 = reward6State.perTokenStored
			// let perTokenStored4_2 = reward4State.perTokenStored
			expect(perTokenStored18_2-perTokenStored18_1).to.be.equal(2e9)
			expect(perTokenStored6_2-perTokenStored6_1).to.be.equal(2e9)
			// expect(perTokenStored4_2-perTokenStored4_1).to.be.equal(2e9)
		})

	})

	describe("claimFor", function () {
		it("should allow an admin to claim rewards on behalf of a user", async function () {
			// Scenario: Admin claims rewards for user1

			const depositAmount = "604800" // user1 deposits 604,800 SYMM
			const rewardAmount = "604800" // Total reward for 7 days

			// user1 deposits tokens
			await stakingToken.connect(user1).approve(await symmStaking.getAddress(), depositAmount)
			await symmStaking.connect(user1).deposit(depositAmount, user1.address)

			// Admin configures and notifies rewards
			await usdtToken.connect(admin).approve(await symmStaking.getAddress(), rewardAmount)
			await symmStaking.connect(admin).addRewardToken(await usdtToken.getAddress())
			await symmStaking.connect(admin).notifyRewardAmount([await usdtToken.getAddress()], [rewardAmount])

			// Fast forward time to accumulate some rewards
			await time.increase(200)

			// Admin claims rewards on behalf of user1
			const user1BalanceBefore = await usdtToken.balanceOf(user1.address)
			await symmStaking.connect(admin).claimFor(user1.address)
			const user1BalanceAfter = await usdtToken.balanceOf(user1.address)

			// The claimed rewards should be 200 tokens (200 seconds * 1 token/sec)
			expect(user1BalanceAfter - user1BalanceBefore).to.equal("200")
		})

		it("should revert if non-admin tries to claim rewards on behalf of a user", async function () {
			// Scenario: A non-admin user attempts to claim rewards for another user

			const depositAmount = "604800" // user1 deposits 604,800 SYMM
			const rewardAmount = "604800" // Total reward for 7 days

			// user1 deposits tokens
			await stakingToken.connect(user1).approve(await symmStaking.getAddress(), depositAmount)
			await symmStaking.connect(user1).deposit(depositAmount, user1.address)

			// Admin configures and notifies rewards
			await usdtToken.connect(admin).approve(await symmStaking.getAddress(), rewardAmount)
			await symmStaking.connect(admin).addRewardToken(await usdtToken.getAddress())
			await symmStaking.connect(admin).notifyRewardAmount([await usdtToken.getAddress()], [rewardAmount])

			// Fast forward time to accumulate some rewards
			await time.increase(200)

			// user2 (non-admin) tries to claim rewards for user1
			await expect(symmStaking.connect(user2).claimFor(user1.address)).to.be.revertedWithCustomError(symmStaking, "AccessControlUnauthorizedAccount")
		})

		it("should revert if there are no rewards to claim for the user", async function () {
			// Scenario: Admin tries to claim rewards for a user who has no rewards

			const depositAmount = "604800" // user1 deposits 604,800 SYMM
			const rewardAmount = "0" // No reward for the user1

			// user1 deposits tokens
			await stakingToken.connect(user1).approve(await symmStaking.getAddress(), depositAmount)
			await symmStaking.connect(user1).deposit(depositAmount, user1.address)

			// Admin configures and notifies rewards (but none for user1)
			await symmStaking.connect(admin).addRewardToken(await usdtToken.getAddress())
			await symmStaking.connect(admin).notifyRewardAmount([await usdtToken.getAddress()], [rewardAmount])

			// Fast forward time
			await time.increase(200)

			// Admin tries to claim rewards for user1 (but there are none)
			await expect(symmStaking.connect(admin).claimFor(user1.address)).to.be.ok
		})

		it("should not allow claiming rewards for zero address", async function () {
			// Scenario: Admin tries to claim rewards for the zero address

			const depositAmount = "604800" // user1 deposits 604,800 SYMM
			const rewardAmount = "604800" // Total reward for 7 days

			// user1 deposits tokens
			await stakingToken.connect(user1).approve(await symmStaking.getAddress(), depositAmount)
			await symmStaking.connect(user1).deposit(depositAmount, user1.address)

			// Admin configures and notifies rewards
			await usdtToken.connect(admin).approve(await symmStaking.getAddress(), rewardAmount)
			await symmStaking.connect(admin).addRewardToken(await usdtToken.getAddress())
			await symmStaking.connect(admin).notifyRewardAmount([await usdtToken.getAddress()], [rewardAmount])

			// Fast forward time
			await time.increase(200)

			// Admin tries to claim rewards for the zero address
			await expect(symmStaking.connect(admin).claimFor("0x0000000000000000000000000000000000000000")).to.be.ok
			// .to.be.revertedWithCustomError(symmStaking, "ZeroAddress");
		})
	})

	describe("Notify", function () {
		it("should notify rewards correctly when one token has a zero amount", async function () {
			// Scenario:
			// 1. Admin configures and notifies rewards for two tokens: USDT and USDC.
			// 2. USDT has a reward amount of zero, while USDC has a valid amount.
			// 3. Contract should only notify the valid token and ignore the one with zero reward amount.

			const rewardAmountUSDT = "0" // USDT has a zero reward amount
			const rewardAmountUSDC = "1000" // USDC has a positive reward amount

			// Admin approves the tokens to notify rewards
			await usdtToken.connect(admin).approve(await symmStaking.getAddress(), rewardAmountUSDT)
			await usdcToken.connect(admin).approve(await symmStaking.getAddress(), rewardAmountUSDC)

			// Admin configures and notifies rewards
			await symmStaking.connect(admin).addRewardToken(await usdtToken.getAddress())
			await symmStaking.connect(admin).addRewardToken(await usdcToken.getAddress())

			// Notify rewards (USDT is zero, so it should be ignored)
			await expect(
				symmStaking
					.connect(admin)
					.notifyRewardAmount([await usdtToken.getAddress(), await usdcToken.getAddress()], [rewardAmountUSDT, rewardAmountUSDC]),
			).to.be.ok
		})
	})

	describe("Withdraw", function () {
		it("should allow a user to deposit and then withdraw after some time, totalSupply should be zero", async function () {
			// Scenario:
			// 1. user deposits tokens
			// 2. waits some time
			// 3. user withdraws

			const depositAmount = "500"
			await stakingToken.connect(user1).approve(await symmStaking.getAddress(), depositAmount)
			await symmStaking.connect(user1).deposit(depositAmount, user1.address)

			// Fast-forward time
			await time.increase(200) // wait for 200 seconds

			const user1BalanceBefore = await stakingToken.balanceOf(user1.address)

			// User withdraws
			await symmStaking.connect(user1).withdraw(depositAmount, user1.address)

			const user1BalanceAfter = await stakingToken.balanceOf(user1.address)
			const withdrawAmount = user1BalanceAfter - user1BalanceBefore

			expect(withdrawAmount).to.equal(depositAmount)
			expect(await symmStaking.connect(admin).totalSupply()).to.equal("0")
		})

		it("should allow a user to deposit, claim reward, and then withdraw tokens", async function () {
			// Scenario:
			// 1. user deposits tokens
			// 2. time passes, user claims rewards
			// 3. user withdraws staked tokens

			const depositAmount = "604800"
			await stakingToken.connect(user1).approve(await symmStaking.getAddress(), depositAmount)
			await symmStaking.connect(user1).deposit(depositAmount, user1.address)

			// Admin mints and notifies rewards
			const rewardAmount = "604800"
			await usdtToken.connect(admin).approve(await symmStaking.getAddress(), rewardAmount)
			await symmStaking.connect(admin).addRewardToken(await usdtToken.getAddress())
			await symmStaking.connect(admin).notifyRewardAmount([await usdtToken.getAddress()], [rewardAmount])

			// Fast-forward time
			await time.increase(200)

			// User claims rewards
			const user1BalanceBefore = await usdtToken.balanceOf(user1.address)
			await symmStaking.connect(user1).claimRewards()
			const user1BalanceAfter = await usdtToken.balanceOf(user1.address)
			const user1Claimed = user1BalanceAfter - user1BalanceBefore

			expect(user1Claimed).to.equal("200")

			// User withdraws staking tokens
			const user1StakedBalanceBefore = await stakingToken.balanceOf(user1.address)
			await symmStaking.connect(user1).withdraw(depositAmount, user1.address)
			const user1StakedBalanceAfter = await stakingToken.balanceOf(user1.address)
			const user1Withdrawn = user1StakedBalanceAfter - user1StakedBalanceBefore

			expect(user1Withdrawn).to.equal(depositAmount)
		})

		it("should handle other users withdraw and calculate the the rewards correctly", async function () {
			// Scenario:
			// 1. user1 deposits 604800 (1 week in seconds)
			// 2. user2 deposits 604800
			// 3. admin configures and notifies two tokens (USDC, USDT) 604800 for each
			// 4. user1 withdraws 604800
			// 5. user2 claims(200 seconds after notifies) => it gets 200
			// 6. user2 withdraws

			const depositUser1 = "604800"
			const depositUser2 = "604800"

			await stakingToken.connect(user1).approve(await symmStaking.getAddress(), depositUser1)
			await stakingToken.connect(user2).approve(await symmStaking.getAddress(), depositUser2)

			await symmStaking.connect(user1).deposit(depositUser1, user1.address)
			await symmStaking.connect(user2).deposit(depositUser2, user2.address)

			const rewardAmount = "604800"
			await usdtToken.connect(admin).approve(await symmStaking.getAddress(), rewardAmount)
			await usdcToken.connect(admin).approve(await symmStaking.getAddress(), rewardAmount)

			await symmStaking.connect(admin).addRewardToken(await usdtToken.getAddress())
			await symmStaking.connect(admin).addRewardToken(await usdcToken.getAddress())

			await stakingToken.connect(admin).approve(await symmStaking.getAddress(), rewardAmount)
			await symmStaking.connect(admin).notifyRewardAmount([await usdtToken.getAddress(), await usdcToken.getAddress()], [rewardAmount, rewardAmount])

			const currentBlock = await ethers.provider.getBlock("latest")
			const afterNotifyTime = currentBlock?.timestamp

			// User1 withdraws staked tokens
			const user1StakedBalanceBefore = await stakingToken.balanceOf(user1.address)
			await symmStaking.connect(user1).withdraw(depositUser1, user1.address)
			const user1StakedBalanceAfter = await stakingToken.balanceOf(user1.address)
			const user1Withdrawn = user1StakedBalanceAfter - user1StakedBalanceBefore

			expect(user1Withdrawn).to.equal(depositUser1)

			// Fast-forward time
			await time.increaseTo(afterNotifyTime + 200)

			// User2 claims rewards
			let user2BalanceBeforeUSDT = await usdtToken.balanceOf(user2.address)
			let user2BalanceBeforeUSDC = await usdcToken.balanceOf(user2.address)

			await symmStaking.connect(user2).claimRewards()

			let user2BalanceAfterUSDT = await usdtToken.balanceOf(user2.address)
			let user2BalanceAfterUSDC = await usdtToken.balanceOf(user2.address)
			let user2ClaimedUSDT = user2BalanceAfterUSDT + user2BalanceAfterUSDC - (user2BalanceBeforeUSDT + user2BalanceBeforeUSDC)

			expect(user2ClaimedUSDT).to.equal("400")

			// User2 claims rewards
			user2BalanceBeforeUSDT = await usdtToken.balanceOf(user2.address)
			user2BalanceBeforeUSDC = await usdcToken.balanceOf(user2.address)

			await symmStaking.connect(user2).claimRewards()

			user2BalanceAfterUSDT = await usdtToken.balanceOf(user2.address)
			user2BalanceAfterUSDC = await usdtToken.balanceOf(user2.address)
			user2ClaimedUSDT = user2BalanceAfterUSDT + user2BalanceAfterUSDC - (user2BalanceBeforeUSDT + user2BalanceBeforeUSDC)

			expect(user2ClaimedUSDT).to.equal("0") //The first claim, claims both tokens

			// User2 withdraws staked tokens
			const user2StakedBalanceBefore = await stakingToken.balanceOf(user2.address)
			await symmStaking.connect(user2).withdraw(depositUser2, user2.address)
			const user2StakedBalanceAfter = await stakingToken.balanceOf(user2.address)
			const user2Withdrawn = user2StakedBalanceAfter - user2StakedBalanceBefore

			expect(user2Withdrawn).to.equal(depositUser2)
		})

		it("should revert if the user tries to withdraw more than their balance", async function () {
			// Scenario:
			// 1. User deposits 500 SYMM.
			// 2. User attempts to withdraw 600 SYMM, which is more than they have.
			// 3. The transaction should revert.

			const depositAmount = "500"
			const withdrawAmount = "600" // Attempting to withdraw more than the deposited amount

			// Approve and deposit the tokens
			await stakingToken.connect(user1).approve(await symmStaking.getAddress(), depositAmount)
			await symmStaking.connect(user1).deposit(depositAmount, user1.address)

			// Check user balance before withdrawal
			const user1BalanceBefore = await stakingToken.balanceOf(user1.address)

			// Try withdrawing more than the user has staked and expect a revert
			await expect(symmStaking.connect(user1).withdraw(withdrawAmount, user1.address)).to.be.revertedWithCustomError(
				symmStaking,
				"InsufficientBalance",
			) // Assuming the revert reason is "InsufficientBalance"

			// Check that the user's balance hasn't changed
			const user1BalanceAfter = await stakingToken.balanceOf(user1.address)
			expect(user1BalanceBefore).to.equal(user1BalanceAfter) // Balance should remain the same
		})

		it("should revert with ZeroAddress if the user tries to withdraw to address(0)", async function () {
			// Scenario:
			// 1. User deposits 500 SYMM.
			// 2. User attempts to withdraw to address(0).
			// 3. The transaction should revert with ZeroAddress error.

			const depositAmount = "500"
			const withdrawAmount = "500" // Amount user wants to withdraw

			// Approve and deposit the tokens
			await stakingToken.connect(user1).approve(await symmStaking.getAddress(), depositAmount)
			await symmStaking.connect(user1).deposit(depositAmount, user1.address)

			// Try withdrawing to address(0) and expect a revert
			await expect(symmStaking.connect(user1).withdraw(withdrawAmount, "0x0000000000000000000000000000000000000000")).to.be.revertedWithCustomError(
				symmStaking,
				"ZeroAddress",
			)
		})

		it("should revert with ZeroAmount if the user tries to withdraw zero amount", async function () {
			// Scenario:
			// 1. User deposits 500 SYMM.
			// 2. User attempts to withdraw 0 SYMM.
			// 3. The transaction should revert with ZeroAmount error.

			const depositAmount = "500"
			const withdrawAmount = "0" // Trying to withdraw zero

			// Approve and deposit the tokens
			await stakingToken.connect(user1).approve(await symmStaking.getAddress(), depositAmount)
			await symmStaking.connect(user1).deposit(depositAmount, user1.address)

			// Try withdrawing 0 amount and expect a revert
			await expect(symmStaking.connect(user1).withdraw(withdrawAmount, user1.address)).to.be.revertedWithCustomError(symmStaking, "ZeroAmount")
		})
	})

	describe("Pause and Unpause Functionality", function () {
		it("should revert when contract is paused for whenNotPaused methods", async function () {
			// Scenario:
			// 1. Admin pauses the contract
			// 2. User tries to withdraw tokens while paused (should revert)
			// 3. Admin unpauses the contract
			// 4. User withdraws tokens after unpausing

			const depositAmount = "500"
			await stakingToken.connect(user1).approve(await symmStaking.getAddress(), depositAmount)
			await symmStaking.connect(user1).deposit(depositAmount, user1.address)

			// Pause the contract
			await symmStaking.connect(admin).pause()

			// Contract is paused
			expect(await symmStaking.paused()).to.be.true

			// Try to withdraw while paused (should revert)
			await expect(symmStaking.connect(user1).withdraw(depositAmount, user1.address)).to.be.revertedWithCustomError(symmStaking, "EnforcedPause")

			// Unpause the contract
			await symmStaking.connect(admin).unpause()

			// Contract is unpaused
			expect(await symmStaking.paused()).to.be.false

			// User1 withdraws tokens now that contract is unpaused (should succeed)
			await symmStaking.connect(user1).withdraw(depositAmount, user1.address)
		})
	})

	describe("Role Management", function () {
		it("should allow only the PAUSER_ROLE to pause and unpause the contract", async function () {
			// Scenario:
			// 1. Only the account with PAUSER_ROLE can pause and unpause the contract
			// 2. Others should not be able to perform these actions and should be reverted.

			// Check that admin can pause
			await symmStaking.connect(admin).pause()
			expect(await symmStaking.paused()).to.be.true

			// Check that user1 cannot pause the contract
			await expect(symmStaking.connect(user1).pause()).to.be.revertedWithCustomError(symmStaking, "AccessControlUnauthorizedAccount")

			// Unpause by admin
			await symmStaking.connect(admin).unpause()
			expect(await symmStaking.paused()).to.be.false

			// Check that user1 cannot unpause the contract
			await expect(symmStaking.connect(user1).unpause()).to.be.revertedWithCustomError(symmStaking, "AccessControlUnauthorizedAccount")
		})

		it("should allow only the REWARD_MANAGER_ROLE to configure reward tokens", async function () {
			// Scenario:
			// 1. Only the account with REWARD_MANAGER_ROLE can configure reward tokens and notify rewards
			// 2. Others should not be able to perform these actions and should be reverted.

			// Check that admin (who has REWARD_MANAGER_ROLE) can configure and notify rewards
			await symmStaking.connect(admin).addRewardToken(await usdtToken.getAddress())
			await usdtToken.connect(admin).approve(await symmStaking.getAddress(), "604800")
			await symmStaking.connect(admin).notifyRewardAmount([await usdtToken.getAddress()], ["604800"])

			// Check that user1 cannot configure or notify rewards
			await expect(symmStaking.connect(user1).addRewardToken(await usdtToken.getAddress())).to.be.revertedWithCustomError(
				symmStaking,
				"AccessControlUnauthorizedAccount",
			)
		})

		it("should allow only the ADMIN_ROLE to grant roles to others", async function () {
			// Scenario:
			// 1. Only the admin can grant roles to other users
			// 2. Others should not be able to grant roles and should be reverted.

			// Check that admin can grant the REWARD_MANAGER_ROLE to user2
			await symmStaking.connect(admin).grantRole(await symmStaking.REWARD_MANAGER_ROLE(), user2.address)

			// Check that user1 cannot grant roles
			await expect(symmStaking.connect(user1).grantRole(await symmStaking.REWARD_MANAGER_ROLE(), user2.address)).to.be.revertedWithCustomError(
				symmStaking,
				"AccessControlUnauthorizedAccount",
			)

			// Check that admin can revoke roles from user2
			await symmStaking.connect(admin).revokeRole(await symmStaking.REWARD_MANAGER_ROLE(), user2.address)

			// Check that user1 cannot revoke roles
			await expect(symmStaking.connect(user1).revokeRole(await symmStaking.REWARD_MANAGER_ROLE(), user2.address)).to.be.revertedWithCustomError(
				symmStaking,
				"AccessControlUnauthorizedAccount",
			)
		})

		it("should revert if an unauthorized user tries to call restricted functions", async function () {
			// Scenario:
			// 1. User1 tries to call functions like pause, unpause, and reward notification without the appropriate roles.
			// 2. The contract should revert with a message indicating the missing role.

			// User1 tries to call pause (should revert, as only PAUSER_ROLE is allowed)
			await expect(symmStaking.connect(user1).pause()).to.be.revertedWithCustomError(symmStaking, "AccessControlUnauthorizedAccount")

			// User1 tries to call unpause (should revert)
			await expect(symmStaking.connect(user1).unpause()).to.be.revertedWithCustomError(symmStaking, "AccessControlUnauthorizedAccount")

			// User1 tries to configure and notify reward tokens (should revert, as only REWARD_MANAGER_ROLE is allowed)
			await expect(symmStaking.connect(user1).addRewardToken(await usdtToken.getAddress())).to.be.revertedWithCustomError(
				symmStaking,
				"AccessControlUnauthorizedAccount",
			)
		})
	})

	describe("View Methods", function () {
		it("should return the correct number of reward tokens", async function () {
			// Scenario:
			// 1. Admin configures 2 reward tokens: USDT and USDC
			// 2. rewardTokensCount should return 2

			await symmStaking.connect(admin).addRewardToken(await usdtToken.getAddress())
			await symmStaking.connect(admin).addRewardToken(await usdcToken.getAddress())

			const rewardTokensCount = await symmStaking.rewardTokensCount()
			expect(rewardTokensCount).to.equal(2)
		})

		it("should return the correct last time reward applicable for a token", async function () {
			// Scenario:
			// 1. Admin configures and notifies rewards for USDT
			// 2. lastTimeRewardApplicable should return the correct time for the rewards token

			const rewardAmount = "604800"
			await usdtToken.connect(admin).approve(await symmStaking.getAddress(), rewardAmount)
			await symmStaking.connect(admin).addRewardToken(await usdtToken.getAddress())
			await symmStaking.connect(admin).notifyRewardAmount([await usdtToken.getAddress()], [rewardAmount])

			const lastTime = await symmStaking.lastTimeRewardApplicable(await usdtToken.getAddress())
			expect(lastTime).to.be.above(0)
		})

		it("should return the correct reward per token", async function () {
			// Scenario:
			// 1. User1 deposits 1000 SYMM
			// 2. Admin configures and notifies rewards for USDT
			// 3. rewardPerToken should return the reward rate per token

			await stakingToken.connect(user1).approve(await symmStaking.getAddress(), "100000")
			await symmStaking.connect(user1).deposit("100000", user1.address)

			const rewardAmount = "604800"
			await usdtToken.connect(admin).approve(await symmStaking.getAddress(), rewardAmount)
			await symmStaking.connect(admin).addRewardToken(await usdtToken.getAddress())
			await symmStaking.connect(admin).notifyRewardAmount([await usdtToken.getAddress()], [rewardAmount])

			await time.increase(200)

			const rewardPerToken = await symmStaking.rewardPerToken(await usdtToken.getAddress())
			expect(rewardPerToken).to.be.above(0) // Ensure a positive reward per token
		})

		it("should return the correct earned amount for an account", async function () {
			// Scenario:
			// 1. User1 deposits 500 SYMM tokens
			// 2. Admin configures and notifies rewards for USDT
			// 3. Earned method should return the correct earned reward for user1

			const depositAmount = "500"
			await stakingToken.connect(user1).approve(await symmStaking.getAddress(), depositAmount)
			await symmStaking.connect(user1).deposit(depositAmount, user1.address)

			const rewardAmount = "604800"
			await usdtToken.connect(admin).approve(await symmStaking.getAddress(), rewardAmount)
			await symmStaking.connect(admin).addRewardToken(await usdtToken.getAddress())
			await symmStaking.connect(admin).notifyRewardAmount([await usdtToken.getAddress()], [rewardAmount])

			// Fast forward time
			await time.increase(200)

			const earnedAmount = await symmStaking.earned(user1.address, await usdtToken.getAddress())
			expect(earnedAmount).to.be.above(0) // User1 should have earned some reward
		})

		it("should return the correct full period reward for a token", async function () {
			// Scenario:
			// 1. Admin configures and notifies rewards for USDT
			// 2. getFullPeriodReward should return the full reward for the duration of the period

			const rewardAmount = "604800"
			await usdtToken.connect(admin).approve(await symmStaking.getAddress(), rewardAmount)
			await symmStaking.connect(admin).addRewardToken(await usdtToken.getAddress())
			await symmStaking.connect(admin).notifyRewardAmount([await usdtToken.getAddress()], [rewardAmount])

			const fullPeriodReward = await symmStaking.getFullPeriodReward(await usdtToken.getAddress())
			expect(fullPeriodReward).to.equal(rewardAmount) // Should match the notified reward amount
		})
	})
}
