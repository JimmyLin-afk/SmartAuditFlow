import { formatEther, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";

async function main() {
  const BeraLongPool = await hre.ethers.getContractFactory("BeraLongPool");
  
  console.log("1. Upgrading BeraLongPool...");
  const address =
    await hre.upgrades.upgradeProxy(
      CONFIG.longPool,
      BeraLongPool,
      {
        redeployImplementation: 'always',
      }
    )
    .then(c => c.waitForDeployment())
    .then(c => c.getAddress()).then(getAddress);
  console.log(`BeraLongPool upgraded to ${address}`);

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
