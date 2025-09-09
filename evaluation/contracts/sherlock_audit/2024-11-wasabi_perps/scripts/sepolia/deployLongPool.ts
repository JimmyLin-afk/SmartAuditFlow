import { zeroAddress, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { LIQUIDATOR_ROLE, ORDER_SIGNER_ROLE } from "../../test/utils/constants";

async function main() {
  // const feeReceiver = "0x129320410d1F827597Befcb01Dc7a037c7fbA6d5";
  // const wethAddress = "0x6400c43e5dd1f713fd623d92dc64831dd12d3298";

  // const maxApy = 300n; // 300% APY
  // const maxLeverage = 500n; // 5x Leverage

  // console.log("1. Deploying DebtController...");
  // const debtController =
  //   await hre.viem.deployContract(  
  //     "DebtController",
  //     [maxApy, maxLeverage]);
  // console.log(`DebtController deployed to ${debtController.address}`);

  // await delay(10_000);
  // await verifyContract(debtController.address, [maxApy, maxLeverage]);

  // console.log("2. Deploying AddressProvider...");
  // const addressProvider = 
  //   await hre.viem.deployContract(
  //       "AddressProvider",
  //       [debtController.address, feeReceiver, wethAddress, feeReceiver]);
  // console.log(`AddressProvider deployed to ${addressProvider.address}`);

  // await delay(10_000);
  // await verifyContract(addressProvider.address, [debtController.address, feeReceiver, wethAddress, feeReceiver]);


  // console.log("3. Deploying Perp Manager...");
  // const PerpManager = await hre.ethers.getContractFactory("PerpManager");
  // const perpManagerAddress = 
  //     await hre.upgrades.deployProxy(
  //         PerpManager,
  //         [],
  //         { kind: 'uups'})
  //     .then(c => c.waitForDeployment())
  //     .then(c => c.getAddress())
  //     .then(getAddress);
  // console.log(`PerpManager deployed to ${perpManagerAddress}`);

  // console.log("3a. Verifying PerpManager...");
  // await delay(10_000);
  // await verifyContract(perpManagerAddress, []);
  // console.log("PerpManager verified");

  // const perpManager = await hre.viem.getContractAt("PerpManager", perpManagerAddress);
  // console.log("3. Grant LIQUIDATOR role...");
  // await perpManager.write.grantRole([LIQUIDATOR_ROLE, feeReceiver, 0]);
  // console.log("LIQUIDATOR role granted");
  
  // console.log("4. Grant ORDER_SIGNER_ROLE role...");
  // await perpManager.write.grantRole([ORDER_SIGNER_ROLE, feeReceiver, 0]);
  // console.log("ORDER_SIGNER_ROLE role granted");

  const addressProvider = await hre.viem.getContractAt("AddressProvider", "0xc0c2DA35262e088472AC25fD75d922a14952426A");
  const perpManagerAddress = "0x2C00dbf8F9996fD41547F67222FB5774E94c72A7";
  console.log("4. Deploying WasabiLongPool...");
  const WasabiLongPool = await hre.ethers.getContractFactory("WasabiLongPool");
  const address = 
      await hre.upgrades.deployProxy(
          WasabiLongPool,
          [addressProvider.address, perpManagerAddress],
          { kind: 'uups'})
      .then(c => c.waitForDeployment())
      .then(c => c.getAddress()).then(getAddress);
  console.log(`WasabiLongPool deployed to ${address}`);

  await delay(10_000);
  await verifyContract(address);
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
