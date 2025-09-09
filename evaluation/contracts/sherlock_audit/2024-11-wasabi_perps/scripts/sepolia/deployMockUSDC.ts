import { formatEther, parseUnits, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";

async function main() {
  const deployer = "0x129320410d1F827597Befcb01Dc7a037c7fbA6d5";
  
  console.log("1. Deploying USDC...");
  const USDC = await hre.viem.deployContract("USDC");
  console.log(`USDC deployed to ${USDC.address}`);

  console.log("2. Minting USDC...");
  await USDC.write.mint([deployer, parseUnits("2000000", 6)]);

  await verifyContract(USDC.address);
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
