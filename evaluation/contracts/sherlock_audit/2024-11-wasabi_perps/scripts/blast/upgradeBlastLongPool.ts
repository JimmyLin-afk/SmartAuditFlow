import { formatEther, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { getBlast, getBlastSepolia } from "./utils";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";

async function main() {
  const longPoolAddress = CONFIG.longPool;

  console.log("1. Upgrading BlastLongPool...");
  const BlastLongPool = await hre.ethers.getContractFactory("BlastLongPool");
  const address =
    await hre.upgrades.upgradeProxy(
      longPoolAddress,
      BlastLongPool,
      {
        redeployImplementation: 'always',
      }
    )
    .then(c => c.waitForDeployment())
    .then(c => c.getAddress()).then(getAddress);
  console.log(`BlastLongPool upgraded to ${address}`);

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
