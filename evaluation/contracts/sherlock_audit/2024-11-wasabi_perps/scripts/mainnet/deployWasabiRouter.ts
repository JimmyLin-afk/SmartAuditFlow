import { formatEther, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";

async function main() {
  const longPoolAddress = "0x8e0edfd6d15f858adbb41677b82ab64797d5afc0";
  const shortPoolAddress = "0x0fdc7b5ce282763d5372a44b01db65e14830d8ff";
  const perpManagerAddress = "0xc0b01a4f4A4459D5A7E13C2E8566CDe93A010e7D";
  const debtControllerAddress = "0xFe7B8F5722ac881242e16beBc8Ea0B28c3EE60C2";
  const wethAddress = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2";
  const swapRouter = "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45";
  const feeReceiver = "0x5C629f8C0B5368F523C85bFe79d2A8EFB64fB0c8";
  const withdrawFeeReceiver = "0x97165754beA07D70Ab27C2A9E02728c79ED80d64";
  const liquidationFeeReceiver = "0xF6336dd76300524Ef382FA9FC861305A37b929b6";
  const feeBips = 25n;

  console.log("1. Deploying WasabiRouter...");
  const WasabiRouter = await hre.ethers.getContractFactory("WasabiRouter");
  const routerAddress = 
      await hre.upgrades.deployProxy(
          WasabiRouter,
          [longPoolAddress, shortPoolAddress, wethAddress, perpManagerAddress, swapRouter, withdrawFeeReceiver, feeBips],
          { kind: 'uups' }
      )
      .then(c => c.waitForDeployment())
      .then(c => c.getAddress()).then(getAddress);
  
  console.log(`WasabiRouter dedployed to ${routerAddress}`);

  await delay(10_000);
  await verifyContract(routerAddress, []);

  console.log("2. Deploying new AddressProvider...");

  const addressProvider = 
    await hre.viem.deployContract(
        "AddressProvider",
        [debtControllerAddress, routerAddress, feeReceiver, wethAddress, liquidationFeeReceiver]);
  console.log(`AddressProvider deployed to ${addressProvider.address}`);

  await delay(10_000);
  await verifyContract(addressProvider.address, [debtControllerAddress, routerAddress, feeReceiver, wethAddress, liquidationFeeReceiver]);
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