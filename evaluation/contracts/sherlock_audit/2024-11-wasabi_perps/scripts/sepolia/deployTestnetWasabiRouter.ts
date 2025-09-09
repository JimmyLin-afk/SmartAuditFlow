import { formatEther, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import {CONFIG} from "./config";

async function main() {
  const longPoolAddress = CONFIG.longPool;
  const shortPoolAddress = CONFIG.shortPool;
  const perpManagerAddress = CONFIG.perpManager;
  const addressProviderAddress = CONFIG.addressProvider;
  const wethAddress = CONFIG.weth;
  const swapRouterAddress = CONFIG.swapRouter;
  const swapFeeReceiver = CONFIG.swapFeeReceiver;
  const swapFeeBips = 25n;

  console.log("1. Deploying WasabiRouter...");
  const WasabiRouter = await hre.ethers.getContractFactory("WasabiRouter");
  const routerAddress = 
      await hre.upgrades.deployProxy(
          WasabiRouter,
          [longPoolAddress, shortPoolAddress, wethAddress, perpManagerAddress, swapRouterAddress, swapFeeReceiver, swapFeeBips],
          { kind: 'uups' }
      )
      .then(c => c.waitForDeployment())
      .then(c => c.getAddress()).then(getAddress);
  
  console.log(`WasabiRouter deployed to ${routerAddress}`);

  await delay(10_000);
  await verifyContract(routerAddress, []);

  console.log("2. Setting WasabiRouter in AddressProvider...");

  const addressProvider = await hre.viem.getContractAt("AddressProvider", addressProviderAddress);
  await addressProvider.write.setWasabiRouter([routerAddress]);
}

function delay(ms: number) {
  return new Promise( resolve => setTimeout(resolve, ms) );
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});