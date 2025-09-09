import { formatEther, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";

async function main() {
  const currentAddress = getAddress(CONFIG.shortPool);
  const WasabiShortPool = await hre.ethers.getContractFactory("WasabiShortPool");

  await hre.upgrades.forceImport(
    CONFIG.shortPool,
    WasabiShortPool
  )

  console.log("1. Upgrading WasabiShortPool...");
  const address =
    await hre.upgrades.upgradeProxy(
      currentAddress,
      WasabiShortPool,
      {
        redeployImplementation: 'always',
        // call: {
        //   fn: "migrateFees",
        //   args: [
        //     ["0x6400c43e5dd1f713fd623d92dc64831dd12d3298", "0x92ea09e6f1cc933baac19cd6414b64a9d84cc135"], 
        //     [2750000000000000n, 10620994n],
        //     [693263597109332504n, 2280378565n]
        //   ]
        // }
      }
    )
    .then(c => c.waitForDeployment())
    .then(c => c.getAddress()).then(getAddress);
  console.log(`WasabiShortPool upgraded to ${address}`);

  await delay(10_000);
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
