import { formatEther, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";
import { delay } from "../utils";

async function main() {
  const BlastRouter = await hre.ethers.getContractFactory("BlastRouter");
  
  console.log("1. Upgrading BlastRouter...");
  const address =
    await hre.upgrades.upgradeProxy(
      CONFIG.wasabiRouter,
      BlastRouter
    )
    .then(c => c.waitForDeployment())
    .then(c => c.getAddress()).then(getAddress);
  console.log(`BlastRouter upgraded to ${address}`);

  await delay(5000); // wait for the contract to be deployed

  await verifyContract(address);

  console.log("Finished setting up BlastRouter");
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
