import { toFunctionSelector, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";

async function main() {
  const config = CONFIG;

  const longPoolAddress = config.longPool;
  const longPool = await hre.viem.getContractAt("WasabiLongPool", longPoolAddress);
  const perpManagerAddress = await longPool.read.owner();

  console.log("1. Deploying ExactOutSwapper...");
  const ExactOutSwapper = await hre.ethers.getContractFactory("ExactOutSwapper");
  const swapperAddress =
      await hre.upgrades.deployProxy(
          ExactOutSwapper,
          [perpManagerAddress],
          { kind: 'uups' }
      )
      .then(c => c.waitForDeployment())
      .then(c => c.getAddress()).then(getAddress);
  console.log(`ExactOutSwapper deployed to ${swapperAddress}`);

  await delay(10_000);
  await verifyContract(swapperAddress, []);
  console.log("Done")
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