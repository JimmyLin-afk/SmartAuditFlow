import { SignerWithAddress } from "@nomicfoundation/hardhat-ethers/signers"
import { task } from "hardhat/config"
import { HardhatRuntimeEnvironment } from "hardhat/types"

task("deploy:SymmAllocationClaimer", "Deploys the SymmAllocationClaimer")
	.addParam("admin", "The admin address of the SymmAllocationClaimer contract")
	.addParam("setter", "The setter address of the SymmAllocationClaimer contract")
	.addParam("token", "The address of symm token")
	.addParam("symmioFoundation", "The address of symmioFoundation")
	.addParam("mintFactor", "The mint factor")
	.setAction(async ({ admin, setter, token, symmioFoundation, mintFactor }, { ethers }: HardhatRuntimeEnvironment) => {
		console.log("deploy:SymmAllocationClaimer")

		const signers: SignerWithAddress[] = await ethers.getSigners()
		const owner: SignerWithAddress = signers[0]

		const SymmAllocationClaimerFactory = await ethers.getContractFactory("SymmAllocationClaimer")
		const symmAllocationClaimer = await SymmAllocationClaimerFactory.connect(owner).deploy(admin, setter, token, symmioFoundation, mintFactor)
		await symmAllocationClaimer.waitForDeployment()

		console.log(`SymmAllocationClaimer deployed: ${await symmAllocationClaimer.getAddress()}`)

		return symmAllocationClaimer
	})
