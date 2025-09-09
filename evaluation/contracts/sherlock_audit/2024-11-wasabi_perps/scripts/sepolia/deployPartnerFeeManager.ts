import { getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";

async function main() {
  const config = CONFIG;

  const longPoolAddress = config.longPool;
  const shortPoolAddress = config.shortPool;
  const perpManagerAddress = config.perpManager;

  console.log("1. Deploying PartnerFeeManager...");
  const PartnerFeeManager = await hre.ethers.getContractFactory("PartnerFeeManager");
  const partnerFeeManagerAddress = 
    await hre.upgrades.deployProxy(
        PartnerFeeManager, 
        [perpManagerAddress, longPoolAddress, shortPoolAddress], 
        { kind: 'uups' }
    )
    .then(c => c.waitForDeployment())
    .then(c => c.getAddress()).then(getAddress);
  console.log(`PartnerFeeManager deployed to ${partnerFeeManagerAddress}`);

  await delay(10_000);
  await verifyContract(partnerFeeManagerAddress, []);
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
