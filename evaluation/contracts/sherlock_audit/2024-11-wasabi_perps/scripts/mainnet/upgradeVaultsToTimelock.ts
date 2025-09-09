import { formatEther, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";

import TimelockVaults from "./mainnetTimelockVaults.json";

async function main() {
  console.log("1. Upgrading vaults...");
  const TimelockWasabiVault = await hre.ethers.getContractFactory("TimelockWasabiVault");
  // const WasabiVault = await hre.ethers.getContractFactory("WasabiVault");

  const cooldownDuration = 0n; // BigInt(14 * 24 * 60 * 60); // 14 days in seconds

  for (let i = 0; i < TimelockVaults.length; i++) {
    const vault = TimelockVaults[i];
    console.log(`   Upgrading ${vault.name} to TimelockWasabiVault...`);
    // await hre.upgrades.forceImport(vault.address, WasabiVault);
    const address =
      await hre.upgrades.upgradeProxy(
          vault.address,
          TimelockWasabiVault,
          {
            call: {
              fn: "setCooldownDuration",
              args: [cooldownDuration]
            }
          }
      )
      .then(c => c.waitForDeployment())
      .then(c => c.getAddress()).then(getAddress);
    
    await delay(5_000);
    await verifyContract(address);

    const implAddress = await hre.upgrades.erc1967.getImplementationAddress(vault.address);
    console.log(`   ${vault.name} upgraded to ${implAddress}`);
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