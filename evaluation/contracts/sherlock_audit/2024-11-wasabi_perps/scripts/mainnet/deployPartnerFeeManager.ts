import { getAddress, zeroAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";

async function main() {
  const config = CONFIG;

  const longPoolAddress = config.longPool;
  const shortPoolAddress = config.shortPool;
  const perpManagerAddress = config.perpManager;
  const debtControllerAddress = "0xfBc94e5Fc406509C9a6D0635c047E1cd601246d1";
  const feeReceiver = "0x5C629f8C0B5368F523C85bFe79d2A8EFB64fB0c8";
  const stakingAccountFactory = "0x0000000000000000000000000000000000000000";

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
  //       [debtControllerAddress, config.wasabiRouter, feeReceiver, config.weth, feeReceiver, zeroAddress, partnerFeeManagerAddress]
  //   );
  // console.log(`AddressProvider deployed to ${addressProvider.address}`);

  // await delay(10_000);
  // await verifyContract(addressProvider.address, [debtControllerAddress, config.wasabiRouter, feeReceiver, config.weth, feeReceiver, zeroAddress, partnerFeeManagerAddress]);

  // await delay(10_000);
  // console.log("3. Upgrading WasabiLongPool...");
  // const WasabiLongPool = await hre.ethers.getContractFactory("WasabiLongPool");
  // await hre.upgrades.upgradeProxy(
  //   longPoolAddress,
  //   WasabiLongPool,
  //   {
  //     redeployImplementation: 'always',
  //     call: {
  //       fn: "migrateFees",
  //       args: [
  //         addressProvider.address,
  //         ["0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"], 
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
  //         ["0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"], 
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
