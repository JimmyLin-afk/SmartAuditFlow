import { zeroAddress, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";

async function main() {

  const dynamicSwap = "0x14f2f68554b60150734a96a7f91bc37916275bf7";

  const tokens =[
    // ['μPudgyPenguins', 'μPPG'],
    // ['μBoredApeYachtClub','μBAYC']
    // ['μMutantApeYachtClub','μMAYC'],
    // ['μAzuki','μAZUKI'],
    // ['μDeGods','μDEGODS'],
    // ['μCaptainz','μCaptainz']
    ['μWasabi','μWasabi']
  ];

  for (let i = 0; i < tokens.length; i++) {
    const name = tokens[i][0];
    const symbol = tokens[i][1];

    console.log(`${i + 1}A: Deploying ${name}...`);
    const token = await hre.viem.deployContract("MockERC20", [name, symbol]);
    console.log(`${name} (${symbol}) deployed to ${token.address}`);

    // console.log(`${i + 1}B: Minting 2M ${symbol} to DynamicSwap...`);
    // await token.write.mint([dynamicSwap, parseEther("2000000")]);
    // console.log(`Minted 2M ${symbol} to DynamicSwap`);

    await verifyContract(token.address, [name, symbol]);
  }
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
