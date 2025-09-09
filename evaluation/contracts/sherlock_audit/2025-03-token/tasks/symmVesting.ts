import { task } from "hardhat/config"
import { HardhatRuntimeEnvironment } from "hardhat/types"

task("deploy:Vesting", "Deploys the Vesting contract")
	.addParam("admin", "The admin of the Vesting contract")
	.addParam("lockedClaimPenaltyReceiver", "Address that receives the penalty")
	.addParam("pool", "Address of the pool")
	.addParam("router", "Address of the router")
	.addParam("permit2", "Address of the permit2")
	.addParam("vault", "Address of the vault")
	.addParam("symm", "Address of symm token")
	.addParam("usdc", "Address of usdc token")
	.addParam("symmLp", "Address of symmLp token")
	.setAction(async ({ admin, lockedClaimPenaltyReceiver, pool, router, permit2, vault, symm, usdc, symmLp }, { ethers, upgrades }: HardhatRuntimeEnvironment) => {
		console.log("deploy:Vesting")

		const VestingPlanOps = await ethers.getContractFactory("VestingPlanOps")
		const vestingPlanOps = await VestingPlanOps.deploy()
		await vestingPlanOps.waitForDeployment()

		const VestingFactory = await ethers.getContractFactory("SymmVesting", {
			libraries: {
				VestingPlanOps: await vestingPlanOps.getAddress(),
			},
		})
		const vestingContract = await upgrades.deployProxy(VestingFactory, [admin, lockedClaimPenaltyReceiver, pool, router, permit2, vault, symm, usdc, symmLp ], {
			unsafeAllow: ["external-library-linking"],
			initializer: "initialize",
		})
		await vestingContract.waitForDeployment()

		console.log(`SymmVesting Contract deployed at: ${await vestingContract.getAddress()}`)
		return vestingContract
	})
