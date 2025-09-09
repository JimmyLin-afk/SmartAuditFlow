import { formatEther, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { getBlast } from "./utils";
import { CONFIG } from "./config";

async function main() {
  const config = CONFIG;
  const {config: blastConfig} = await getBlast();

  const longPoolAddress = config.longPool;
  const shortPoolAddress = config.shortPool;

  const longPool = await hre.viem.getContractAt("WasabiLongPool", longPoolAddress, blastConfig);
  const perpManagerAddress = await longPool.read.owner();
  const existingAddressProviderAddress = await longPool.read.addressProvider();
  const existingAddressProvider = await hre.viem.getContractAt("AddressProvider", existingAddressProviderAddress, blastConfig);
  const weth = await existingAddressProvider.read.getWethAddress();

  console.log("1. Deploying WETH WasabiVault...");
  const contractName = "BlastWasabiVault";
  const BlastWasabiVault = await hre.ethers.getContractFactory(contractName);
  const address = 
      await hre.upgrades.deployProxy(
        BlastWasabiVault,
          [longPoolAddress, shortPoolAddress, existingAddressProviderAddress, perpManagerAddress, weth, 'Spicy WETH Vault', 'sWETH'],
          { kind: 'uups'}
      )
      .then(c => c.waitForDeployment())
      .then(c => c.getAddress()).then(getAddress);
  const vault = await hre.viem.getContractAt(contractName, address, blastConfig);
  console.log(`BlastWasabiVault deployed to ${address}`);

  console.log("2. Setting vault in pool...");
  await longPool.write.addVault([vault.address],);
  console.log(`Vault ${address} added to pool ${longPoolAddress}`);
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
