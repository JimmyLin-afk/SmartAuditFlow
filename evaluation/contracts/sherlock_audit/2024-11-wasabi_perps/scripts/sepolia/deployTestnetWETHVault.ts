import { formatEther, parseEther, getAddress } from "viem";
import hre from "hardhat";
import {CONFIG} from "./config";

async function main() {
  const deployer = "0x129320410d1F827597Befcb01Dc7a037c7fbA6d5";
  const longPoolAddress = CONFIG.longPool;
  const shortPoolAddress = CONFIG.shortPool;
  const longPool = await hre.viem.getContractAt("WasabiLongPool", longPoolAddress);
  const shortPool = await hre.viem.getContractAt("WasabiShortPool", shortPoolAddress);
  const addressProviderAddress = CONFIG.addressProvider;
  const perpManagerAddress = CONFIG.perpManager;
  const wethAddress = CONFIG.weth;
  const weth = await hre.viem.getContractAt("WETH9", wethAddress);

  console.log("1. Deploying WETH WasabiVault...");
  const contractName = "WasabiVault";
  const WasabiVault = await hre.ethers.getContractFactory(contractName);
  const address = 
      await hre.upgrades.deployProxy(
          WasabiVault,
          [longPoolAddress, shortPoolAddress, addressProviderAddress, perpManagerAddress, wethAddress, 'Spicy WETH Vault', 'sWETH'],
          { kind: 'uups'}
      )
      .then(c => c.waitForDeployment())
      .then(c => c.getAddress()).then(getAddress);
  const vault = await hre.viem.getContractAt(contractName, address);
  console.log(`WasabiVault deployed to ${address}`);

  console.log("2. Adding vault in long pool...");
  await longPool.write.addVault([vault.address]);
  console.log(`Vault ${address} added to pool ${longPoolAddress}`);

  console.log("3. Adding vault in short pool...");
  await shortPool.write.addVault([vault.address]);
  console.log(`Vault ${address} added to pool ${shortPoolAddress}`);

  console.log("4. Depositing WETH to vault...");
  const depositAmount = parseEther("25");
  await weth.write.approve([address, depositAmount]);
  await vault.write.deposit([depositAmount, deployer]);
  console.log(`Deposited ${formatEther(depositAmount)} ETH to vault ${address}`);
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
