import { zeroAddress, parseEther, getAddress } from "viem";
import hre from "hardhat";
import { verifyContract } from "../../utils/verifyContract";
import { LIQUIDATOR_ROLE, ORDER_SIGNER_ROLE, VAULT_ADMIN_ROLE } from "../../test/utils/constants";

async function main() {
  const feeReceiver = "0x5C629f8C0B5368F523C85bFe79d2A8EFB64fB0c8";
  const wethAddress = "0x4200000000000000000000000000000000000006";

  const maxApy = 300n; // 300% APY
  const maxLeverage = 500n; // 5x Leverage

  console.log("1. Deploying DebtController...");
  const debtController =
    await hre.viem.deployContract(  
      "DebtController",
      [maxApy, maxLeverage]);
  console.log(`DebtController deployed to ${debtController.address}`);

  await delay(10_000);
  await verifyContract(debtController.address, [maxApy, maxLeverage]);

  console.log("2. Deploying AddressProvider...");
  const addressProvider = 
    await hre.viem.deployContract(
        "AddressProvider",
        [debtController.address, zeroAddress, feeReceiver, wethAddress, feeReceiver]);
  console.log(`AddressProvider deployed to ${addressProvider.address}`);

  await delay(10_000);
  await verifyContract(addressProvider.address, [debtController.address, zeroAddress, feeReceiver, wethAddress, feeReceiver]);

  console.log("3. Deploying Perp Manager...");
  const PerpManager = await hre.ethers.getContractFactory("PerpManager");
  const perpManagerAddress = 
      await hre.upgrades.deployProxy(
          PerpManager,
          [],
          { kind: 'uups'})
      .then(c => c.waitForDeployment())
      .then(c => c.getAddress())
      .then(getAddress);
  console.log(`PerpManager deployed to ${perpManagerAddress}`);

  await delay(10_000);
  await verifyContract(perpManagerAddress, []);

  const perpManager = await hre.viem.getContractAt("PerpManager", perpManagerAddress);
  console.log("4. Grant roles...");
  console.log(`4.1 Granting LIQUIDATOR_ROLE to ${feeReceiver}...`);
  await perpManager.write.grantRole([LIQUIDATOR_ROLE, feeReceiver, 0]);
  console.log("LIQUIDATOR role granted");
  
  await delay(10_000);
  console.log(`4.2. Granting ORDER_SIGNER_ROLE to ${feeReceiver}...`);
  await perpManager.write.grantRole([ORDER_SIGNER_ROLE, feeReceiver, 0]);
  console.log("ORDER_SIGNER_ROLE role granted");

  await delay(10_000);
  console.log(`4.3. Granting VAULT_ADMIN_ROLE to ${feeReceiver}...`);
  await perpManager.write.grantRole([VAULT_ADMIN_ROLE, feeReceiver, 0]);
  console.log("VAULT_ADMIN_ROLE role granted");

  await delay(10_000);
  console.log("5. Deploying WasabiLongPool...");
  const WasabiLongPool = await hre.ethers.getContractFactory("WasabiLongPool");
  const address = 
      await hre.upgrades.deployProxy(
          WasabiLongPool,
          [addressProvider.address, perpManagerAddress],
          { kind: 'uups'})
      .then(c => c.waitForDeployment())
      .then(c => c.getAddress()).then(getAddress);
  console.log(`WasabiLongPool deployed to ${address}`);

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
