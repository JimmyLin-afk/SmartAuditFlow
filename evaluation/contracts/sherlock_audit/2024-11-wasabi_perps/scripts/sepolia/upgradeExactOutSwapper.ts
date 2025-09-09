import { formatEther, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";
import { delay } from "../utils";

async function main() {
  const swapperAddress = CONFIG.exactOutSwapper;
  const ExactOutSwapper = await hre.ethers.getContractFactory("ExactOutSwapper");

  console.log("1. Upgrading ExactOutSwapper...");
  const address =
    await hre.upgrades.upgradeProxy(
      swapperAddress,
      ExactOutSwapper
    )
    .then(c => c.waitForDeployment())
    .then(c => c.getAddress()).then(getAddress);
  
  await verifyContract(address);

  await delay(10_000);

  const implAddress = getAddress(await hre.upgrades.erc1967.getImplementationAddress(address));
  console.log(`ExactOutSwapper upgraded to ${implAddress}`);
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});