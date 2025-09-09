import { formatEther, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { getBlast, getBlastSepolia } from "./utils";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";

async function main() {

  console.log("1. Upgrading BlastShortPool...");
  const BlastShortPool = await hre.ethers.getContractFactory("BlastShortPool");
  const address =
    await hre.upgrades.upgradeProxy(
      CONFIG.shortPool,
      BlastShortPool,
      {
        redeployImplementation: 'always',
      }
    )
    .then(c => c.waitForDeployment())
    .then(c => c.getAddress()).then(getAddress);
  console.log(`BlastShortPool upgraded to ${address}`);

  await delay(5_000);
  await verifyContract(address);
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
