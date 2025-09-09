import { zeroAddress, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";

async function main() {
  console.log("1. Deploying DynamicSwap...");
  const dynamicSwap = await hre.viem.deployContract("DynamicSwap");
  console.log(`DynamicSwap deployed to ${dynamicSwap.address}`);

  await verifyContract(dynamicSwap.address);
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
