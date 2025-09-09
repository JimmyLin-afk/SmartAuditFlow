import { task } from "hardhat/config";
import { HardhatRuntimeEnvironment } from "hardhat/types";

task("deploy:SymmStaking", "Deploys the SymmStaking contract")
	.addParam("admin", "The admin of the SymmStaking contract")
	.addParam("stakingToken", "The address of the staking token")
	.setAction(async ({admin, stakingToken}, { ethers, upgrades }: HardhatRuntimeEnvironment) => {
		console.log("deploy:SymmStaking");

		const SymmStakingFactory = await ethers.getContractFactory("SymmStaking");
		const symmStakingContract = await upgrades.deployProxy(SymmStakingFactory, [admin, stakingToken], { initializer: "initialize" });
		await symmStakingContract.waitForDeployment();

		console.log(`SymmStaking Contract deployed at: ${await symmStakingContract.getAddress()}`);
		return symmStakingContract;
	});
