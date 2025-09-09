import { formatEther, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";
import { delay } from "../utils";

async function main() {
  const PerpManager = await hre.ethers.getContractFactory("PerpManager");
  
  console.log("1. Upgrading PerpManager...");
  const address =
    await hre.upgrades.upgradeProxy(
      CONFIG.perpManager,
      PerpManager
    )
    .then(c => c.waitForDeployment())
    .then(c => c.getAddress()).then(getAddress);
  console.log(`PerpManager upgraded to ${address}`);

  await delay(10_000);

  await verifyContract(address);

  console.log("Finished setting up PerpManager");
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
