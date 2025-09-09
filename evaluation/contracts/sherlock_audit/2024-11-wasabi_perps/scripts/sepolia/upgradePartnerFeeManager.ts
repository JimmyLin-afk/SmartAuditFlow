import { formatEther, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";
import { delay } from "../utils";

async function main() {
    const feeManagerAddress = "0x4C076139584801E0268D4292e0bd0AA3a69cef16";
    const PartnerFeeManager = await hre.ethers.getContractFactory("PartnerFeeManager");

    console.log("1. Upgrading PartnerFeeManager...");
    const address =
    await hre.upgrades.upgradeProxy(
      feeManagerAddress,
      PartnerFeeManager
    )
    .then(c => c.waitForDeployment())
    .then(c => c.getAddress()).then(getAddress);
  
  await verifyContract(address);

  await delay(10_000);

  const implAddress = getAddress(await hre.upgrades.erc1967.getImplementationAddress(address));
  console.log(`PartnerFeeManager upgraded to ${implAddress}`);
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});