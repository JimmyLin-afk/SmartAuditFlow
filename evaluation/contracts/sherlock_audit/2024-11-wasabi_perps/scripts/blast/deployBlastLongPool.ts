import { zeroAddress, parseEther, getAddress, defineChain } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { getBlastSepolia } from "./utils";

async function main() {
  
  const {
    config, feeReceiver, wethAddress, perpManager
  } = await getBlastSepolia();

  const maxApy = 300n; // 300% APY
  const maxLeverage = 500n; // 5x Leverage

  console.log("1. Deploying DebtController...");
  const debtController =
    await hre.viem.deployContract(
      "DebtController",
      [maxApy, maxLeverage],
      config);
  console.log(`DebtController deployed to ${debtController.address}`);

  await delay(10_000);
  await verifyContract(debtController.address, [maxApy, maxLeverage]);

  console.log("2. Deploying AddressProvider...");
  const addressProvider = 
    await hre.viem.deployContract(
      "AddressProvider",
      [debtController.address, feeReceiver, wethAddress],
      config);
  console.log(`AddressProvider deployed to ${addressProvider.address}`);

  await delay(10_000);
  await verifyContract(addressProvider.address, [debtController.address, feeReceiver, wethAddress]);

  console.log("4. Deploying BlastLongPool...");
  const BlastLongPool = await hre.ethers.getContractFactory("BlastLongPool");
  const address = 
      await hre.upgrades.deployProxy(
        BlastLongPool,
        [addressProvider.address, perpManager],
        { kind: 'uups', redeployImplementation: "always"})
      .then(c => c.waitForDeployment())
      .then(c => c.getAddress()).then(getAddress);
  console.log(`BlastLongPool deployed to ${address}`);

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
