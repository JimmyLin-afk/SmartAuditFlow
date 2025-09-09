import { formatEther, parseEther, getAddress } from "viem";
import hre from "hardhat";

import PerpTokens from "./mainnetTokens.json";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";

// ALWAYS RUN ImportMarket script before this one
async function main() {
  const config = CONFIG;

  const longPool = await hre.viem.getContractAt("WasabiLongPool", config.longPool);
  const shortPool = await hre.viem.getContractAt("WasabiShortPool", config.shortPool);
  const perpManagerAddress = await longPool.read.owner();
  const cooldownDuration = BigInt(14 * 24 * 60 * 60); // 14 days in seconds

  for (let i = 0; i < PerpTokens.length; i++) {
    const token = PerpTokens[i];
    console.log(`[${i + 1}/${PerpTokens.length}] Deploying Vault For ${token.address}...`);
    console.log(`------------ 1. Deploying ${token.name} WasabiVault...`);
    const contractName = "TimelockWasabiVault";
    const TimelockWasabiVault = await hre.ethers.getContractFactory(contractName);
    const name = `Spicy ${token.symbol} Vault`;
    const address =
        await hre.upgrades.deployProxy(
            TimelockWasabiVault,
            [config.longPool, config.shortPool, config.addressProvider, perpManagerAddress, getAddress(token.address), name, `s${token.symbol}`, cooldownDuration],
            {
              kind: 'uups',
              initializer: 'initialize(address,address,address,address,address,string,string,uint256)',
           })
            .then(c => c.waitForDeployment())
            .then(c => c.getAddress()).then(getAddress);
    console.log(`------------------------ ${name} deployed to ${address}`);

    await delay(10_000);

    console.log("------------ 2. Verifying contract...");
    await verifyContract(address);
    console.log(`------------------------ Contract ${address} verified`);

    await delay(10_000);

    console.log("------------ 3. Setting vault in short pool...");
    await shortPool.write.addVault([address]);
    console.log(`------------------------ Vault ${address} added to pool ${config.shortPool}`);
    console.log("------------ Finished");

    await delay(10_000);
  }
}

function delay(ms: number) {
  return new Promise( resolve => setTimeout(resolve, ms) );
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
// To run: 
// npx hardhat run scripts/deployShortPoolVaults.ts --network mainnet
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
