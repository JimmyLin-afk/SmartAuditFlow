import { formatEther, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";
import { delay } from "../utils";

async function main() {
  const currentAddress = getAddress(CONFIG.longPool);
  const WasabiLongPool = await hre.ethers.getContractFactory("WasabiLongPool");

  // await hre.upgrades.forceImport(
  //   CONFIG.longPool,
  //   WasabiLongPool
  // )
  
  console.log("1. Upgrading WasabiLongPool...");
  const address =
    await hre.upgrades.upgradeProxy(
      currentAddress,
      WasabiLongPool,
      {
        redeployImplementation: 'always',
        // call: {
        //   fn: "migrateFees",
        //   args: [
        //     ["0x6400c43e5dd1f713fd623d92dc64831dd12d3298", "0x92ea09e6f1cc933baac19cd6414b64a9d84cc135"], 
        //     [12300000263809366n, 226470503n],
        //     [12941153986961836488n, 3291553456n]
        //   ]
        // }
      }
    )
    .then(c => c.waitForDeployment())
    .then(c => c.getAddress()).then(getAddress);
  console.log(`WasabiLongPool upgraded to ${address}`);

  await delay(10000);
  await verifyContract(address);
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
