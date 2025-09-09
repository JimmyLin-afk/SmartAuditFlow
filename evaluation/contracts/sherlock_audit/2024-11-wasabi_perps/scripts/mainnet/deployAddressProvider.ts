import { zeroAddress, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";

async function main() {
  const config = CONFIG;

  const debtControllerAddress = "0xfBc94e5Fc406509C9a6D0635c047E1cd601246d1";
  const feeReceiver = "0x5C629f8C0B5368F523C85bFe79d2A8EFB64fB0c8";
  const liquidationFeeReceiver = "0xF6336dd76300524Ef382FA9FC861305A37b929b6";

  console.log("1. Deploying AddressProvider...");
  const addressProvider = 
    await hre.viem.deployContract(
        "AddressProvider", 
        [
          debtControllerAddress,
          config.wasabiRouter,
          feeReceiver,
          config.weth,
          liquidationFeeReceiver,
          zeroAddress, // stakingAccountFactory
          config.partnerFeeManager
        ]
    );
  console.log(`AddressProvider deployed to ${addressProvider.address}`);

  await delay(10_000);
  await verifyContract(addressProvider.address, 
    [
      debtControllerAddress,
      config.wasabiRouter,
      feeReceiver,
      config.weth,
      liquidationFeeReceiver,
      zeroAddress, // stakingAccountFactory
      config.partnerFeeManager
    ]
  );
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
