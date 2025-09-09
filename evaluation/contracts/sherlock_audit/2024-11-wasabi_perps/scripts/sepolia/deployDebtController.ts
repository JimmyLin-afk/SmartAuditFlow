import { zeroAddress, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { LIQUIDATOR_ROLE, ORDER_SIGNER_ROLE } from "../../test/utils/constants";

async function main() {
  const addressProviderAddress = "0xc0c2DA35262e088472AC25fD75d922a14952426A";

  const maxApy = 300n; // 300% APY
  const maxLeverage = 500n; // 5x Leverage

  console.log("1. Deploying DebtController...");
  const debtController =
    await hre.viem.deployContract(  
      "DebtController",
      [maxApy, maxLeverage]);
  console.log(`DebtController deployed to ${debtController.address}`);

  await delay(10_000);
  await verifyContract(debtController.address, [maxApy, maxLeverage]);

  console.log("2. Setting in AddressProvider...");
  const addressProvider = await hre.viem.getContractAt(
    "AddressProvider",
    addressProviderAddress
  );
  await addressProvider.write.setDebtController([debtController.address]);
  console.log("DebtController set in AddressProvider");
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
