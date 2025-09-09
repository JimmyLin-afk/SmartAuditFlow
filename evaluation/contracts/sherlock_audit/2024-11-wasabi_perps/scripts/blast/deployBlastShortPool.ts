import { zeroAddress, parseEther, getAddress, defineChain } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";

async function main() {
  
  const deployer = "0x5C629f8C0B5368F523C85bFe79d2A8EFB64fB0c8";
  const feeReceiver = "0x5C629f8C0B5368F523C85bFe79d2A8EFB64fB0c8";
  const wethAddress = "0x4300000000000000000000000000000000000004";
  const perpManager = "0xff2CDb9cdb79A60A31188FE37Bdc6774107cc268";
  const addressProvider = "0x66873693E8b125dbea46274173B0d6DcD1933CCa";

  const chain = defineChain({
    id: 81457,
    name: "blast",
    nativeCurrency: {
      name: "Ether",
      symbol: "ETH",
      decimals: 18
    },
    rpcUrls: {
      default: {
        http: ["https://holy-practical-owl.blast-mainnet.quiknode.pro/590d29b28faafbe6e3f80846cdc50e3f4c5356b0"]
      }
    }
  });
  const walletClient = await hre.viem.getWalletClient(deployer, { chain });
  const publicClient = await hre.viem.getPublicClient({ chain });
  const config = {
    client: {
      public: publicClient,
      wallet: walletClient
    }
  };

  const maxApy = 300n; // 300% APY
  const maxLeverage = 500n; // 5x Leverage

  console.log("4. Deploying BlastShortPool...");
  const BlastShortPool = await hre.ethers.getContractFactory("BlastShortPool");
  const address = 
      await hre.upgrades.deployProxy(
        BlastShortPool,
        [addressProvider, perpManager],
        { kind: 'uups', redeployImplementation: "always"})
      .then(c => c.waitForDeployment())
      .then(c => c.getAddress()).then(getAddress);
  console.log(`BlastShortPool deployed to ${address}`);

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
