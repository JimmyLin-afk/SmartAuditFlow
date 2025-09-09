import { getAddress, zeroAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";

async function main() {
  const config = CONFIG;

  const longPoolAddress = config.longPool;
  const shortPoolAddress = config.shortPool;
  const perpManagerAddress = config.perpManager;
  const debtControllerAddress = "0x93F287F2c3FD9FdE49373afd3E6E679D7A9350F9";
  const feeReceiver = "0x5C629f8C0B5368F523C85bFe79d2A8EFB64fB0c8";
  const liquidationFeeReceiver = "0xF6336dd76300524Ef382FA9FC861305A37b929b6";
  const stakingAccountFactory = "0x1E5C9AA12B37393BCdECbBEE0892830561c15d1a";

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

  // console.log("2. Deploying new AddressProvider...");
  // const addressProvider = 
  //   await hre.viem.deployContract(
  //       "AddressProvider", 
  //       [debtControllerAddress, config.wasabiRouter, feeReceiver, config.weth, liquidationFeeReceiver, stakingAccountFactory, partnerFeeManagerAddress]
  //   );
  // console.log(`AddressProvider deployed to ${addressProvider.address}`);

  // await delay(10_000);
  // await verifyContract(addressProvider.address, [debtControllerAddress, config.wasabiRouter, feeReceiver, config.weth, feeReceiver, zeroAddress, partnerFeeManagerAddress]);

  // await delay(10_000);
  // console.log("3. Upgrading BeraLongPool...");
  // const BeraLongPool = await hre.ethers.getContractFactory("BeraLongPool");
  // await hre.upgrades.upgradeProxy(
  //   longPoolAddress,
  //   BeraLongPool,
  //   {
  //     redeployImplementation: 'always',
  //     call: {
  //       fn: "migrateFees",
  //       args: [
  //         addressProvider.address,
  //         ["0x6969696969696969696969696969696969696969", "0xFCBD14DC51f0A4d49d5E53C2E0950e0bC26d0Dce"], 
  //         [], // Run balance checker script to get these values
  //         []  // Run balance checker script to get these values
  //       ]
  //     }
  //   }
  // );

  // await delay(10_000);
  // await verifyContract(longPoolAddress);

  // await delay(10_000);
  // console.log("4. Upgrading WasabiShortPool...");
  // const WasabiShortPool = await hre.ethers.getContractFactory("WasabiShortPool");
  // await hre.upgrades.upgradeProxy(
  //   shortPoolAddress,
  //   WasabiShortPool,
  //   {
  //     redeployImplementation: 'always',
  //     call: {
  //       fn: "migrateFees",
  //       args: [
  //         addressProvider.address,
  //         ["0x6969696969696969696969696969696969696969", "0xFCBD14DC51f0A4d49d5E53C2E0950e0bC26d0Dce"], 
  //         [], // Run balance checker script to get these values
  //         []  // Run balance checker script to get these values
  //       ]
  //     }
  //   }
  // );

  // await delay(10_000);
  // await verifyContract(shortPoolAddress);

  // console.log("Done")
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
