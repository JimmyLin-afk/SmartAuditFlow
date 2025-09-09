import { formatEther, parseEther, getAddress, Address } from "viem";
import hre from "hardhat";
import { getBlast, getBlastSepolia } from "./utils";
import { verifyContract } from "../../utils/verifyContract";
import { delay } from "../utils";
import BlastVaults from "./blastVaults.json";

async function main() {
  const {config} = await getBlast();

  const longPoolAddress = "0x046299143A880C4d01a318Bc6C9f2C0A5C1Ed355";
  const shortPoolAddress = "0x0301079DaBdC9A2c70b856B2C51ACa02bAc10c3a";

  let totalGas = 0n;

  const usdbVault = "0x4bed2a922654cacc2be974689619768fabf24855";
  const ethVault = "0x8e2b50413a53f50e2a059142a9be060294961e40";

  const allVaults = [
    "0x04ffa5a58a27d94e3364918a1597841b62620519",
    "0x09c885de423eab752bfead9de203c9fe8c33082a",
    "0x14f86c94d260489c73df7ceeb45205d5bff0d13a",
    "0x18412f323aede6786993a3f42828e817b4a876b7",
    "0x1e046bc49eeebd0f5633caa9638fe977cfdaf0c8",
    "0x237e604e8a946df3332bfa318191d72895f80144",
    "0x2cb60ad6fb60355d41fd848034c85e20740acce2",
    "0x3336f14ad8c9731352d4323212d86d231dae1558",
    "0x3563b4f9a60474a392dab6f188effff3688f9e9a",
    "0x3a76684ab84fa2c8c3fa1919726777d662ce2e8e",
    "0x3ccdbd9336711894126b5f7fc4f26d4547e768ad",
    "0x4b5cadc5cce9181de8bf9b4a3b1c6107c52fa4df",
    "0x4bed2a922654cacc2be974689619768fabf24855",
    "0x4f01f2c50acb3ae759b0b664f5b19a0f18b6b551",
    "0x5c0f73ced4b7caf05ee46385c548acb77389b5a7",
    "0x5ed244e11ecfe3b0f299b5e22f84ae6036a7ac21",
    "0x616afdcc1f2606cde40e556570b608904d103558",
    "0x6b4d371a557c2b5987fcbcc7b841819bb919303c",
    "0x7274aa6606e7c3afabc6ba3e7e345c03eee7fe81",
    "0x73ec6a715805236176de2468318f6920efed9d74",
    "0x7eda4afad0764dbe971ed3e0884ec3196ecacccf",
    "0x80d2438ae77fe7761a137b490b297cbe01d4aeaa",
    "0x8e2b50413a53f50e2a059142a9be060294961e40",
    "0x9db92282d040ba7adaab10e4787a5af0eda64cba",
    "0x9e31ef400c74630ab50066dd64c29c1f4fc57209",
    "0x9eea5bdb09670c2def6c338bf1cf4f477c48be22",
    "0xb5a09514966e9902df2525bacbd0c86fceacc078",
    "0xba74ab0bdc17d085dae189499e4d23a124d46c1a",
    "0xc8061516994aa5d884fdf6385c6b64e7b9e93014",
    "0xcb1379b4e68350d09bc39e6944de310439feca2a",
    "0xcc082c5fe2919fefaa5356386717b2c7c30d7ab5",
    "0xcc3eed7dec4471086ac8eb6f799bd1095e56d34a",
    "0xce979f9a3bc1f3bf57d573c653c8f8b0f2d4de4d",
    "0xf2abb552eb3c8a8a580c22558ad7fbeb34a6af53",
    "0xffa03410e05d66d4dbe18e31b4ad4f0289b70432"
  ]
  

  const rebasingWeth = await hre.viem.getContractAt("IERC20Rebasing", "0x4300000000000000000000000000000000000004", config);
  const rebasingUsdb = await hre.viem.getContractAt("IERC20Rebasing", "0x4300000000000000000000000000000000000003", config);
  const blast = await hre.viem.getContractAt("IBlast", "0x4300000000000000000000000000000000000002", config);

  let gasParams;
  console.log('------------ VAULTS ------------');
  for (const i in allVaults) {
    const vaultAddress = allVaults[i] as Address;
    // const vault = await hre.viem.getContractAt("BlastWasabiVault", vaultAddress, config);
    // await vault.simulate.claimAllGas
    console.log('Vault: ' + vaultAddress);
    gasParams = await blast.read.readGasParams([vaultAddress]);
    totalGas += gasParams[1];
    console.log('Gas Earned: ' + formatEther(gasParams[1]));
  };

  console.log('------------ LONG POOL ------------')
  const extraEthYieldLong = await blast.read.readClaimableYield([longPoolAddress]);
  const extraWethYieldLong = await rebasingWeth.read.getClaimableAmount([longPoolAddress]);
  const extraUSDBYieldLong = await rebasingUsdb.read.getClaimableAmount([longPoolAddress]);
  gasParams = await blast.read.readGasParams([longPoolAddress]);
  totalGas += gasParams[1];
  console.log('Gas Earned: ' + formatEther(gasParams[1]));
  console.log('ETH Yield: ' + formatEther(extraEthYieldLong));
  console.log('WETH Yield: ' + formatEther(extraWethYieldLong));
  console.log('USDB Yield: ' + formatEther(extraUSDBYieldLong));

  console.log('------------ SHORT POOL ------------')
  const extraEthYieldShort = await blast.read.readClaimableYield([shortPoolAddress]);
  const extraWethYieldShort = await rebasingWeth.read.getClaimableAmount([shortPoolAddress]);
  const extraUSDBYieldShort = await rebasingUsdb.read.getClaimableAmount([shortPoolAddress]);
  gasParams = await blast.read.readGasParams([shortPoolAddress]);
  totalGas += gasParams[1];
  console.log('Gas Earned: ' + formatEther(gasParams[1]));
  console.log('ETH Yield: ' + formatEther(extraEthYieldShort));
  console.log('WETH Yield: ' + formatEther(extraWethYieldShort));
  console.log('USDB Yield: ' + formatEther(extraUSDBYieldShort));

  console.log('------------ TOTAL ------------');
  console.log('Total Gas Earned: ' + formatEther(totalGas));
  console.log('Total Extra ETH Yield: ' + formatEther(extraEthYieldLong + extraWethYieldLong + extraEthYieldShort + extraWethYieldShort));
  console.log('Total Extra USDB Yield: ' + formatEther(extraUSDBYieldLong + extraUSDBYieldShort));

  // const blastVaults = BlastVaults;
  // for (let i = 0; i < blastVaults.length; i++) {
  //   const blastVault = blastVaults[i];
  //   const vaultAddress = blastVault.address as Address;
  //   console.log(`[${i + 1}/${allVaults.length}] Processing vault ${vaultAddress}`);

  //   const gasParams = await blast.read.readGasParams([vaultAddress]);
  //   const gasEarned = gasParams[1];

  //   if (gasEarned < 3000000000000n) {
  //     console.log(`Skipping vault ${blastVault.symbol} due to low gas earnings: ${formatEther(gasEarned)}`);
  //     console.log('----------------------- \n');
  //     continue;
  //   }

  //   console.log(`Claiming gas for vault ${blastVault.symbol} [${blastVault.address}]\nEarnings: ${formatEther(gasEarned)} ETH`);
  //   // const vault = await hre.viem.getContractAt("BlastVault", vaultAddress, config);
  //   // const tx = await vault.write.claimAllGas([vaultAddress, "0x5C629f8C0B5368F523C85bFe79d2A8EFB64fB0c8"]); 

  //   // console.log('Claimed on https://blastscan.io/tx/' + tx);
    
  //   // await delay(3000);

  //   console.log('----------------------- \n');
  // }
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
