import { formatEther, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { CONFIG } from "./config";

async function main() {
  const longPoolAddress = CONFIG.longPool;
  const shortPoolAddress = CONFIG.shortPool;

  const longPool = await hre.viem.getContractAt("WasabiLongPool", longPoolAddress);
  const perpManagerAddress = await longPool.read.owner();
  const existingAddressProviderAddress = await longPool.read.addressProvider();
  const existingAddressProvider = await hre.viem.getContractAt("AddressProvider", existingAddressProviderAddress);
  const weth = await existingAddressProvider.read.getWethAddress();

  console.log("1. Deploying WETH WasabiVault...");
  const contractName = "WasabiVault";
  const WasabiVault = await hre.ethers.getContractFactory(contractName);
  const address = 
      await hre.upgrades.deployProxy(
          WasabiVault,
          [longPoolAddress, shortPoolAddress, existingAddressProviderAddress, perpManagerAddress, weth, 'Wasabi WETH Vault', 'wWETH'],
          { kind: 'uups'}
      )
      .then(c => c.waitForDeployment())
      .then(c => c.getAddress()).then(getAddress);
  const vault = await hre.viem.getContractAt(contractName, address);
  console.log(`WasabiVault deployed to ${address}`);

  console.log("2. Setting vault in pool...");
  await longPool.write.addVault([vault.address]);
  console.log(`Vault ${address} added to pool ${longPoolAddress}`);
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
