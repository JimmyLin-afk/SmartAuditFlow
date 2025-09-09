import { HardhatEthersSigner } from '@nomicfoundation/hardhat-ethers/signers'
import { ethers } from 'hardhat'

import { VaultFactory } from '../typechain-types'

export async function registerAdapters(vaultFactory: VaultFactory, deployer: HardhatEthersSigner) {
  const lidoAdapter = await ethers.getContractFactory('LidoAdapter')
  await addAdapter(lidoAdapter.bytecode, vaultFactory, deployer)
}

async function addAdapter(
  bytecode: any,
  vaultFactory: VaultFactory,
  deployer: HardhatEthersSigner
) {
  let addAdapterTx = await vaultFactory.connect(deployer).addAdapterType(bytecode)
  let addAdapterReceipt = await addAdapterTx.wait()
  let addAdapterId
  // Find the newly added adapter ID from factory log
  if (addAdapterReceipt?.logs) {
    for (const log of addAdapterReceipt?.logs) {
      let e = vaultFactory.interface.parseLog(log as any)
      if (e?.name === 'AdapterTypeAdded') addAdapterId = e.args.id
    }
  }

  console.log('addAdapterId', addAdapterId)

  return addAdapterId
}
