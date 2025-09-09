import { formatEther, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import BlastVaults from "./blastVaults.json";
import { CONFIG } from "./config";

async function main() {

  console.log("1. Upgrading BlastVaults...");
  const BlastVault = await hre.ethers.getContractFactory("BlastVault");

  for (let i = 0; i < BlastVaults.length; i++) {
    const vault = BlastVaults[i];
    console.log(`  Upgrading BlastVault ${vault.name} [https://blastscan.io/address/${vault.address}]...`);

    const address =
      await hre.upgrades.upgradeProxy(
          vault.address,
          BlastVault,
          {
            call: {
              fn: "setInterestFeeBips",
              args: [1000]
            }
          }
      )
      .then(c => c.waitForDeployment())
      .then(c => c.getAddress())
      .then(getAddress);

    await delay(5_000);

    const implAddress = getAddress(await hre.upgrades.erc1967.getImplementationAddress(address));
    console.log(`BlastVault ${vault.name} upgraded to ${implAddress}`);

    await delay(5_000);
    await verifyContract(address);
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
