import { getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import BaseVaults from "./baseVaults.json";

async function main() {

  console.log("1. Upgrading Vaults...");
  const WasabiVault = await hre.ethers.getContractFactory("WasabiVault");
  const TimelockWasabiVault = await hre.ethers.getContractFactory("TimelockWasabiVault");

  for (let i = 0; i < BaseVaults.length; i++) {
    const vault = BaseVaults[i];
    console.log(`[${i + 1}/${BaseVaults.length}] - Upgrading WasabiVault ${vault.name}...`);
    const timelockWasabiVault = await hre.viem.getContractAt("TimelockWasabiVault", getAddress(vault.address));
    let isTimelockVault = false;
    let contractFactory = WasabiVault;
    try {
      const duration = await timelockWasabiVault.read.getCooldownDuration();
      console.log(`  ${vault.name} is a timelock vault: duration: ${duration}`);
      isTimelockVault = true;
      contractFactory = TimelockWasabiVault;
    } catch (error) {
      // console.log(`  ${vault.name} is not a timelock vault`);
    }

    try {
      const feebps = await timelockWasabiVault.read.interestFeeBips();
      console.log(`  ${vault.name} interest fee bips: ${feebps} | Skipping upgrade`);
      continue;
    } catch (error) {
    }

    let address =
      await hre.upgrades.upgradeProxy(
          vault.address,
          contractFactory,
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
    console.log(`[${i + 1}/${BaseVaults.length}] - WasabiVault ${vault.name} upgraded to ${implAddress}`);

    await delay(5_000);
    await verifyContract(getAddress(address));
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
