import { getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { delay } from "../utils";

async function main() {
  const PartnerFeeManager = await hre.ethers.getContractFactory("PartnerFeeManager");
  const partnerFeeManagerAddress = "0xf0714a3e2f4d78ffc1b400bfdacb7b8869bfe1be";
  
  console.log("1. Upgrading PartnerFeeManager...");
  const address =
    await hre.upgrades.upgradeProxy(
      partnerFeeManagerAddress,
      PartnerFeeManager
    )
    .then(c => c.waitForDeployment())
    .then(c => c.getAddress()).then(getAddress);
  console.log(`PartnerFeeManager upgraded to ${address}`);

  await delay(5_000);

  await verifyContract(address);

  console.log("Finished setting up PerpMaPartnerFeeManagernager");
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
