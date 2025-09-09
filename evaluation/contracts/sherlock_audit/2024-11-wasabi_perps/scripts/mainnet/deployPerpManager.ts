import { zeroAddress, parseEther, getAddress, defineChain } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { LIQUIDATOR_ROLE, ORDER_SIGNER_ROLE } from "../../test/utils/constants";


async function main() {

    const deployer = "0x5C629f8C0B5368F523C85bFe79d2A8EFB64fB0c8";
    const longPool = "0x8e0edfd6d15f858adbb41677b82ab64797d5afc0";
    const shortPool = "0x0fdc7b5ce282763d5372a44b01db65e14830d8ff";

    // console.log("1. Deploying Perp Manager...");
    // const PerpManager = await hre.ethers.getContractFactory("PerpManager");
    // const perpManagerAddress = 
    //     await hre.upgrades.deployProxy(
    //         PerpManager,
    //         [],
    //         { kind: 'uups'})
    //     .then(c => c.waitForDeployment())
    //     .then(c => c.getAddress())
    //     .then(getAddress);
    // console.log(`PerpManager deployed to ${perpManagerAddress}`);

    // console.log("2. Verifying PerpManager...");
    // await delay(10_000);
    // await verifyContract(perpManagerAddress, []);
    // console.log("PerpManager verified");

    // const perpManager = await hre.viem.getContractAt("PerpManager", perpManagerAddress);

    // console.log("3. Grant LIQUIDATOR role...");
    // await perpManager.write.grantRole([LIQUIDATOR_ROLE, deployer, 0]);
    // console.log("LIQUIDATOR role granted");
    
    // console.log("4. Grant ORDER_SIGNER_ROLE role...");
    // await perpManager.write.grantRole([ORDER_SIGNER_ROLE, deployer, 0]);
    // console.log("ORDER_SIGNER_ROLE role granted");

    const perpManagerAddress = "0xc0b01a4f4a4459d5a7e13c2e8566cde93a010e7d";
    
    // console.log("5. Upgrading long pool...");
    // const WasabiLongPool = await hre.ethers.getContractFactory("WasabiLongPool");
    
    // const longUpgradedAddress = await hre.upgrades.upgradeProxy(
    //     longPool,
    //     WasabiLongPool,
    //     {
    //         redeployImplementation: "always",
    //     })
    //     .then(c => c.waitForDeployment())
    //     .then(c => c.getAddress()).then(getAddress);
    // console.log(`WasabiLongPool upgraded to ${longUpgradedAddress}`);
    
    // console.log("6. Verifying long pool...");
    // await delay(10_000);
    // await verifyContract(longUpgradedAddress);
    // console.log("WasabiLongPool verified");

    console.log("7. Upgrading short pool...");
    const WasabiShortPool = await hre.ethers.getContractFactory("WasabiShortPool");

    const shortUpgradedAddress = await hre.upgrades.upgradeProxy(
        shortPool,
        WasabiShortPool,
        {
            redeployImplementation: "always"
        })
        .then(c => c.waitForDeployment())
        .then(c => c.getAddress()).then(getAddress);
    console.log(`WasabiShortPool upgraded to ${shortUpgradedAddress}`);

    console.log("8. Verifying short pool...");
    await delay(10_000);
    await verifyContract(shortUpgradedAddress);
    console.log("WasabiShortPool verified");
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
