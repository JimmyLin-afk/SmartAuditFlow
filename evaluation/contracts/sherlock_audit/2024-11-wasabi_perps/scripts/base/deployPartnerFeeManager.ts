import { getAddress, zeroAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";

async function main() {
  const config = CONFIG;

  const longPoolAddress = config.longPool;
  const shortPoolAddress = config.shortPool;
  const debtControllerAddress = "0xcc9b0312968388D34393dfda57e62Aa1F2cF895d";
  const feeReceiver = "0x5C629f8C0B5368F523C85bFe79d2A8EFB64fB0c8";
  const liquidationFeeReceiver = "0xF6336dd76300524Ef382FA9FC861305A37b929b6";

  // console.log("1. Deploying PartnerFeeManager...");
  // const PartnerFeeManager = await hre.ethers.getContractFactory("PartnerFeeManager");
  // const partnerFeeManagerAddress = 
  //   await hre.upgrades.deployProxy(
  //       PartnerFeeManager, 
  //       [perpManagerAddress, longPoolAddress, shortPoolAddress], 
  //       { kind: 'uups' }
  //   )
  //   .then(c => c.waitForDeployment())
  //   .then(c => c.getAddress()).then(getAddress);
  // console.log(`PartnerFeeManager deployed to ${partnerFeeManagerAddress}`);

  // await delay(10_000);
  // await verifyContract(partnerFeeManagerAddress, []);
  const partnerFeeManagerAddress = "0xf0714a3e2f4d78ffc1b400bfdacb7b8869bfe1be";

  console.log("2. Deploying new AddressProvider...");
  const addressProvider = 
    await hre.viem.deployContract(
        "AddressProvider", 
        [debtControllerAddress, config.wasabiRouter, feeReceiver, config.weth, liquidationFeeReceiver, zeroAddress, partnerFeeManagerAddress]
    );
  console.log(`AddressProvider deployed to ${addressProvider.address}`);

  await delay(10_000);
  await verifyContract(addressProvider.address, [debtControllerAddress, config.wasabiRouter, feeReceiver, config.weth, feeReceiver, zeroAddress, partnerFeeManagerAddress]);

  await delay(10_000);
  console.log("3. Upgrading WasabiLongPool...");
  const WasabiLongPool = await hre.ethers.getContractFactory("WasabiLongPool");
  await hre.upgrades.upgradeProxy(
    longPoolAddress,
    WasabiLongPool,
    {
      redeployImplementation: 'always',
      call: {
        fn: "migrateFees",
        args: [
          addressProvider.address,
          ["0x4200000000000000000000000000000000000006", "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"], 
          [], // Run balance checker script to get these values
          []  // Run balance checker script to get these values
        ]
      }
    }
  );

  await delay(10_000);
  await verifyContract(longPoolAddress);

  await delay(10_000);
  console.log("4. Upgrading WasabiShortPool...");
  const WasabiShortPool = await hre.ethers.getContractFactory("WasabiShortPool");
  await hre.upgrades.upgradeProxy(
    shortPoolAddress,
    WasabiShortPool,
    {
      redeployImplementation: 'always',
      call: {
        fn: "migrateFees",
        args: [
          addressProvider.address,
          ["0x4200000000000000000000000000000000000006", "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"], 
          [], // Run balance checker script to get these values
          []  // Run balance checker script to get these values
        ]
      }
    }
  );

  await delay(10_000);
  await verifyContract(shortPoolAddress);

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
