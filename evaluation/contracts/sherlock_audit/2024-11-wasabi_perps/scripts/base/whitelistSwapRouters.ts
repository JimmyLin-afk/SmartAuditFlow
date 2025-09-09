import { toFunctionSelector, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";

async function main() {
  const config = CONFIG;

  const exactOutSwapper = config.exactOutSwapper;
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

  const swapper = await hre.viem.getContractAt("ExactOutSwapper", exactOutSwapper);

  console.log("1. Whitelisting Uniswap swap router...");

  await swapper.write.setWhitelistedAddress([swapRouterAddress, true]);

  await delay(10_000);

  console.log("2. Whitelisting OpenOcean swap router...");

  await swapper.write.setWhitelistedAddress([openOceanSwapRouterAddress, true]);

  await delay(10_000);

  console.log("3. Whitelisting swap function selectors...");
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