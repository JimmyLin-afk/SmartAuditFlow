import { toFunctionSelector, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";

async function main() {
  const config = CONFIG;

  const longPoolAddress = config.longPool;
  const swapRouterAddress = config.swapRouter;
  const openOceanSwapRouterAddress = "0x6352a56caadc4f1e25cd6c75970fa768a3304e64";
  const swapFunctionSelectors = [
    toFunctionSelector("function swapExactTokensForTokens(uint256,uint256,address[],address)"),
    toFunctionSelector("function exactInputSingle((address,address,uint24,address,uint256,uint256,uint160))"),
    toFunctionSelector("function exactInput((bytes,address,uint256,uint256))"),
    toFunctionSelector(
      "function swap(address,(address,address,address,address,uint256,uint256,uint256,uint256,address,bytes),(uint256,uint256,uint256,bytes)[])"
    )
  ]

  const longPool = await hre.viem.getContractAt("WasabiLongPool", longPoolAddress);
  const perpManagerAddress = await longPool.read.owner();

  console.log("1. Deploying ExactOutSwapper...");
  const ExactOutSwapper = await hre.ethers.getContractFactory("ExactOutSwapper");
  const swapperAddress =
      await hre.upgrades.deployProxy(
          ExactOutSwapper,
          [perpManagerAddress],
          { kind: 'uups' }
      )
      .then(c => c.waitForDeployment())
      .then(c => c.getAddress()).then(getAddress);
  console.log(`ExactOutSwapper deployed to ${swapperAddress}`);

  await delay(10_000);
  await verifyContract(swapperAddress, []);
  const swapper = await hre.viem.getContractAt("ExactOutSwapper", swapperAddress);

  console.log("2. Whitelisting Uniswap swap router...");

  await swapper.write.setWhitelistedAddress([swapRouterAddress, true]);

  await delay(10_000);

  console.log("3. Whitelisting OpenOcean swap router...");

  await swapper.write.setWhitelistedAddress([openOceanSwapRouterAddress, true]);

  await delay(10_000);

  console.log("4. Whitelisting swap function selectors...");
  await swapper.write.setWhitelistedFunctionSelectors([swapFunctionSelectors, true]);

  await delay(10_000);
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