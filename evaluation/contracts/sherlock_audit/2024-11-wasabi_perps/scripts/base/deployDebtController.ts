import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";

async function main() {
  console.log("1. Deploying DebtController...");
  const debtController = 
    await hre.viem.deployContract("DebtController", [300, 1010]);
  console.log(`DebtController deployed to ${debtController.address}`);

  await delay(5_000);
  await verifyContract(debtController.address, [300, 1010]);
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
