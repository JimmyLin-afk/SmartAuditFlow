import { zeroAddress, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";

async function main() {
  const config = CONFIG;

  const longPoolAddress = config.longPool;
  const shortPoolAddress = config.shortPool;
  const longPool = await hre.viem.getContractAt("WasabiLongPool", longPoolAddress);
  const shortPool = await hre.viem.getContractAt("WasabiShortPool", shortPoolAddress);

  const debtControllerAddress = "0x954383b1E0D3017A00aC6895174e4e907Cb7f925";
  const feeReceiver = "0x129320410d1F827597Befcb01Dc7a037c7fbA6d5";
  const partnerFeeManagerAddress = "0x4C076139584801E0268D4292e0bd0AA3a69cef16";

  console.log("1. Deploying new AddressProvider...");
  const addressProvider = await hre.viem.deployContract("AddressProvider", [
    debtControllerAddress,
    config.wasabiRouter,
    feeReceiver,
    config.weth,
    feeReceiver,
    zeroAddress,
    partnerFeeManagerAddress
  ]);

  console.log(`AddressProvider deployed to ${addressProvider.address}`);

  await delay(10_000);
  await verifyContract(addressProvider.address, [debtControllerAddress, config.wasabiRouter, feeReceiver, config.weth, feeReceiver, zeroAddress, partnerFeeManagerAddress]);

  console.log("2. Setting AddressProvider for WasabiLongPool...");
  await longPool.write.setAddressProvider([addressProvider.address]);

  console.log("3. Setting AddressProvider for WasabiShortPool...");
  await shortPool.write.setAddressProvider([addressProvider.address]);

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