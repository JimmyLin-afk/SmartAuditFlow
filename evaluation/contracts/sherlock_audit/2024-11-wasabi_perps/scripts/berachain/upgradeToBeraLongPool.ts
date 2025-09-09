import { getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";

async function main() {
  const currentAddress = getAddress(CONFIG.longPool);
  const BeraLongPool = await hre.ethers.getContractFactory("BeraLongPool");
  
  console.log("1. Upgrading WasabiLongPool to BeraLongPool...");
  const address =
    await hre.upgrades.upgradeProxy(
      currentAddress,
      BeraLongPool,
      {
        redeployImplementation: "always",
        call: {
          fn: "setAddressProvider",
          args: [CONFIG.addressProvider],
        }
      }
    )
    .then(c => c.waitForDeployment())
    .then(c => c.getAddress()).then(getAddress);

  await delay(10_000);
  const implAddress = getAddress(await hre.upgrades.erc1967.getImplementationAddress(address));
  console.log(`WasabiLongPool upgraded to BeraLongPool at ${implAddress}`);

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
