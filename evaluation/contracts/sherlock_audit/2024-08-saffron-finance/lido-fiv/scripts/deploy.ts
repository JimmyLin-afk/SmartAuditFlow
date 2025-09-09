import { ethers } from 'hardhat'
import hre from 'hardhat'

import { VaultFactory } from '../typechain-types'

async function main() {
  console.log(`Deploying to ${hre.network.name}`)
  const [deployer] = await hre.ethers.getSigners()

  const vaultFactoryContract = await ethers.getContractFactory('VaultFactory')
  
  const protocolFeeBps = 100;
  const earlyExitFeeBps = 1000;

  const vaultFactory: VaultFactory = (await vaultFactoryContract.deploy(
    protocolFeeBps,
    earlyExitFeeBps
  )) as any
  console.log(`📃🆕 \x1b[32m${await vaultFactory.getAddress()}\x1b[0m VaultFactory deployed`)

}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error)
    process.exit(1)
  })
