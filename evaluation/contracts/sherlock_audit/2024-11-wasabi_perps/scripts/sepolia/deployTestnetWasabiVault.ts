import { formatEther, parseEther, getAddress } from "viem";
import hre from "hardhat";

import PerpTokens from "./sepoliaPerpTokens.json";
import { verifyContract } from "../../utils/verifyContract";
import { CONFIG } from "./config";

interface PerpToken {
  address: string;
  symbol: string;
  name: string;
  decimals: number;
  image_url: string;
  nft_address: string;
  pair_address: string;
  mainnet_address: string;
}

async function main() {
  const longPoolAddress = CONFIG.longPool;
  const shortPoolAddress = CONFIG.shortPool;
  const shortPool = await hre.viem.getContractAt("WasabiShortPool", shortPoolAddress);
  const addressProvider = CONFIG.addressProvider;
  const perpManagerAddress = CONFIG.perpManager;

  for (let i = 0; i < PerpTokens.length; i++) {
    const token = PerpTokens[i];
    console.log(`[${i + 1}/${PerpTokens.length}] Deploying Vault For ${token.address}...`);
    console.log(`------------ 1. Deploying ${token.name} WasabiVault...`);
    const contractName = "WasabiVault";
    const WasabiVault = await hre.ethers.getContractFactory(contractName);
    const name = `Spicy ${token.symbol} Vault`;
    const address =
        await hre.upgrades.deployProxy(
            WasabiVault,
            [longPoolAddress, shortPoolAddress, addressProvider, perpManagerAddress, token.address, name, `s${token.symbol}`],
            { kind: 'uups'})
            .then(c => c.waitForDeployment())
            .then(c => c.getAddress()).then(getAddress);
    console.log(`------------------------ ${name} deployed to ${address}`);

    await delay(10_000);

    console.log("------------ 2. Adding vault in short pool...");
    await shortPool.write.addVault([address]);
    console.log(`------------------------ ${name} added to pool ${shortPool.address}`);

    await delay(10_000);

    console.log("------------ 3. Verifying contract...");
    await verifyContract(address);
    console.log(`------------------------ Contract ${address} verified`);

    console.log("------------ Finished");
  }
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
