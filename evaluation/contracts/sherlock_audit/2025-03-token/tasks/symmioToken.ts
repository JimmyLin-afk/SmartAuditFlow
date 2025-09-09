import { SignerWithAddress } from "@nomicfoundation/hardhat-ethers/signers"
import { task } from "hardhat/config"
import { HardhatRuntimeEnvironment } from "hardhat/types"

task("deploy:SymmioToken", "Deploys the Symmio token")
	.addParam("name", "The name of the Symmio token contract")
	.addParam("symbol", "The symbol of the Symmio token contract")
	.addParam("admin", "The admin address of the Symmio token contract")
	.setAction(async ({ name, symbol, admin }, { ethers }: HardhatRuntimeEnvironment) => {
		console.log("deploy:SymmioToken")

		const signers: SignerWithAddress[] = await ethers.getSigners()
		const owner: SignerWithAddress = signers[0]

		const SymmioTokenFactory = await ethers.getContractFactory("Symmio")
		const symmioToken = await SymmioTokenFactory.connect(owner).deploy(name, symbol, admin)
		await symmioToken.waitForDeployment()

		console.log(`Symmio Token deployed: ${await symmioToken.getAddress()}`)

		return symmioToken
	})
