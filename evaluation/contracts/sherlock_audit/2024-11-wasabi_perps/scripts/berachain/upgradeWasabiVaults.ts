import { formatEther, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import BeraVaults from "./berachainVaults.json";

async function main() {

  console.log("1. Upgrading BeraVaults...");
  const BeraVault = await hre.ethers.getContractFactory("BeraVault");

  for (let i = 0; i < BeraVaults.length; i++) {
    const vault = BeraVaults[i];
    console.log(`  Upgrading BeraVault ${vault.name} [${vault.address}]...`);
    const address =
      await hre.upgrades.upgradeProxy(
          vault.address,
          BeraVault,
          {
            call: {
              fn: "setInterestFeeBips",
              args: [1000]
            }
          }
      )
      .then(c => c.waitForDeployment())
      .then(c => c.getAddress()).then(getAddress);
    const implAddress = getAddress(await hre.upgrades.erc1967.getImplementationAddress(address));
    console.log(`${i + 1}/${BeraVaults.length} - BeraVault ${vault.name} upgraded to ${implAddress}`);

    await delay(5_000);
    await verifyContract(address);
    await delay(5_000);
  }
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
