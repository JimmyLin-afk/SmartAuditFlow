import { ethers, run } from "hardhat"

async function main() {
	const contractName = ""
	const factory = await ethers.getContractFactory(contractName)
	const contract = await factory.deploy()

	await contract.waitForDeployment()

	console.log(`${contractName} deployed: ${await contract.getAddress()}`)

	await run("verify:verify", {
		address: await contract.getAddress(),
		constructorArguments: [],
	})
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch(error => {
	console.error(error)
	process.exitCode = 1
})
