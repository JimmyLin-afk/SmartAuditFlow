import { toFunctionSelector, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";
import { VAULT_ADMIN_ROLE } from "../../test/utils/constants";

async function main() {
  const config = CONFIG;

  const longPoolAddress = config.longPool;
  const longPool = await hre.viem.getContractAt("WasabiLongPool", longPoolAddress);
  const perpManagerAddress = await longPool.read.owner();
  const vaultAddress = "0xd948212f077e552533158becbc1882c1b19c40fe";

  console.log("1. Deploying CappedVaultCompetitionDepositor...");
  const CappedVaultCompetitionDepositor = await hre.ethers.getContractFactory("CappedVaultCompetitionDepositor");
  const competitionDepositorAddress =
    await hre.upgrades.deployProxy(
      CappedVaultCompetitionDepositor,
      [ vaultAddress, perpManagerAddress ],
      { kind: 'uups' }
    )
    .then(c => c.waitForDeployment())
    .then(c => c.getAddress()).then(getAddress);
  console.log(`CappedVaultCompetitionDepositor deployed to ${competitionDepositorAddress}`);

  await delay(10_000);
  await verifyContract(competitionDepositorAddress, []);

  console.log("2. Granting VAULT_ADMIN role to competition depositor...");
  const perpManager = await hre.viem.getContractAt("PerpManager", perpManagerAddress);
  await perpManager.write.grantRole([VAULT_ADMIN_ROLE, competitionDepositorAddress, 0]);
  await delay(10_000);

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