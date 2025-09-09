import { formatEther, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";

async function main() {
  const config = CONFIG;

  const longPoolAddress = config.longPool;
  const shortPoolAddress = config.shortPool;
  const addressProviderAddress = config.addressProvider;

  const longPool = await hre.viem.getContractAt("WasabiLongPool", longPoolAddress);
  const existingAddressProvider = await hre.viem.getContractAt("AddressProvider", addressProviderAddress);
  const perpManagerAddress = await longPool.read.owner();
  const debtControllerAddress = await existingAddressProvider.read.debtController();
  const wberaAddress = config.weth;
  const swapRouter = config.swapRouter;
  const feeReceiver = "0x5C629f8C0B5368F523C85bFe79d2A8EFB64fB0c8";
  const withdrawFeeReceiver = "0x97165754beA07D70Ab27C2A9E02728c79ED80d64";
  const liquidationFeeReceiver = "0x5C629f8C0B5368F523C85bFe79d2A8EFB64fB0c8";
  const feeBips = 25n;

  console.log("1. Deploying WasabiRouter...");
  const WasabiRouter = await hre.ethers.getContractFactory("WasabiRouter");
  const routerAddress = 
      await hre.upgrades.deployProxy(
          WasabiRouter,
          [longPoolAddress, shortPoolAddress, wberaAddress, perpManagerAddress, swapRouter, withdrawFeeReceiver, feeBips],
          { kind: 'uups' }
      )
      .then(c => c.waitForDeployment())
      .then(c => c.getAddress()).then(getAddress);
  
  console.log(`WasabiRouter deployed to ${routerAddress}`);

  await delay(10_000);
  await verifyContract(routerAddress, []);

  // console.log("2. Deploying new AddressProvider...");

  // const addressProvider = 
  //   await hre.viem.deployContract(
  //       "AddressProvider",
  //       [debtControllerAddress, routerAddress, feeReceiver, wethAddress, liquidationFeeReceiver]);
  // console.log(`AddressProvider deployed to ${addressProvider.address}`);

  // await delay(10_000);
  // await verifyContract(addressProvider.address, [debtControllerAddress, routerAddress, feeReceiver, wethAddress, liquidationFeeReceiver]);

  console.log("2. Setting WasabiRouter in AddressProvider...");

  await existingAddressProvider.write.setWasabiRouter([routerAddress]);
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