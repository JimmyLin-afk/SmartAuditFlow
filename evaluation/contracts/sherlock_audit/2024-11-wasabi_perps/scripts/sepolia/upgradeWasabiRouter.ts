import { formatEther, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";

async function main() {
  const feeBips = 25n;
  const WasabiRouter = await hre.ethers.getContractFactory("WasabiRouter");

  // await hre.upgrades.forceImport(
  //   CONFIG.wasabiRouter,
  //   WasabiRouter
  // )
  
  console.log("1. Upgrading WasabiRouter...");
  const address =
    await hre.upgrades.upgradeProxy(
      CONFIG.wasabiRouter,
      WasabiRouter,
      {
        redeployImplementation: 'always',
      }
    )
    .then(c => c.waitForDeployment())
    .then(c => c.getAddress()).then(getAddress);
  console.log(`WasabiRouter upgraded to ${address}`);

  await verifyContract(address);

  // const wasabiRouter = await hre.viem.getContractAt("WasabiRouter", address);
  // console.log("2. Setting WETH...");
  // await wasabiRouter.write.setWETH([CONFIG.weth]);

  // console.log("3. Setting swapRouter...");
  // await wasabiRouter.write.setSwapRouter([CONFIG.swapRouter]);

  // console.log("4. Setting feeReceiver...");
  // await wasabiRouter.write.setFeeReceiver([CONFIG.swapFeeReceiver]);

  // console.log("5. Setting withdrawFeeBips...");
  // await wasabiRouter.write.setWithdrawFeeBips([feeBips]);

  // console.log("Finished setting up WasabiRouter");
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
