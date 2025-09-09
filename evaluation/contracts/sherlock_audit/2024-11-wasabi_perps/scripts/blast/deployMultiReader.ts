import { zeroAddress, parseEther, getAddress, defineChain } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { getBlast } from "./utils";

async function main() {
  const {
    config
  } = await getBlast();

  console.log("1. Deploying MultiReader...");
  const MultiReader = 
    await hre.viem.deployContract("MultiReader");
  console.log(`MultiReader deployed to ${MultiReader.address}`);

  await delay(10_000);
  await verifyContract(MultiReader.address);
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
