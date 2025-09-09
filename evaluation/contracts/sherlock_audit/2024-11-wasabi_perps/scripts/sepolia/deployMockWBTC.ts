import { formatEther, parseUnits, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";

async function main() {
  const deployer = "0x129320410d1F827597Befcb01Dc7a037c7fbA6d5";
  
  console.log("1. Deploying WBTC...");
  const WBTC = await hre.viem.deployContract("WBTC");
  console.log(`WBTC deployed to ${WBTC.address}`);

  console.log("2. Minting WBTC...");
  await WBTC.write.mint([deployer, parseUnits("100", 6)]);

  await verifyContract(WBTC.address);
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
