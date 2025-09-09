import { getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";

async function main() {
  const factoryAddress = "0x1E5C9AA12B37393BCdECbBEE0892830561c15d1a";
  // const StakingAccountFactory = await hre.ethers.getContractFactory("StakingAccountFactory");

  // console.log("1. Upgrading StakingAccountFactory...");
  // const address =
  //   await hre.upgrades.upgradeProxy(
  //       factoryAddress,
  //       StakingAccountFactory
  //   )
  //   .then(c => c.waitForDeployment())
  //   .then(c => c.getAddress()).then(getAddress);
    
  // await delay(10_000);
  // const implAddress = getAddress(await hre.upgrades.erc1967.getImplementationAddress(address));
  // console.log(`StakingAccountFactory upgraded to ${implAddress}`);  
  
  // await verifyContract(address);

  console.log("2. Deploying new StakingAccount implementation...");
  const StakingAccount = await hre.ethers.getContractFactory("StakingAccount");
  const stakingAccountImplAddress = await StakingAccount.deploy()
    .then(c => c.waitForDeployment())
    .then(c => c.getAddress()).then(getAddress);

  await delay(10_000);
  await verifyContract(stakingAccountImplAddress);

  console.log("3. Upgrading StakingAccounts...");
  const stakingAccountFactory = await hre.viem.getContractAt("StakingAccountFactory", factoryAddress);
  await stakingAccountFactory.write.upgradeBeacon([stakingAccountImplAddress]);

  await delay(10_000);
  console.log("Done");
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
