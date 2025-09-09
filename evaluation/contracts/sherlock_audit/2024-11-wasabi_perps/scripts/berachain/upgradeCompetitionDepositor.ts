import { toFunctionSelector, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";
import { VAULT_ADMIN_ROLE } from "../../test/utils/constants";

async function main() {
  const config = CONFIG;

  const depositor = "0xdee9f998293BbAEe7214f91e45701a317Bb55F18";
  console.log("1. Upgrading CappedVaultCompetitionDepositor...");
  const CappedVaultCompetitionDepositor = await hre.ethers.getContractFactory("CappedVaultCompetitionDepositor");
  
  const address =
    await hre.upgrades.upgradeProxy(
        depositor,
        CappedVaultCompetitionDepositor
    )
    .then(c => c.waitForDeployment())
    .then(c => c.getAddress()).then(getAddress);

  await delay(10_000);
  await verifyContract(address, []);

  console.log("Done")
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