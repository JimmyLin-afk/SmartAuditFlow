import { parseEther } from 'ethers'
import { ethers } from 'hardhat'
import hre from 'hardhat'
import chai from 'chai'

import { equalWithTolerance } from './bigIntHelpers'

export async function impersonateAccounts(addresses: string[]) {
  const promises: any[] = []
  for (const address of addresses) {
    promises.push(
      hre.network.provider.request({
        method: 'hardhat_impersonateAccount',
        params: [address],
      })
    )
  }
  await Promise.all(promises)
}

/**
 * @dev Sets the code of any address. If a contract exists, it will be replaced, however existing storage will remain.
 * @param name Name of the contract to get code from.
 * @param address Address to be deployed to
 * @param deployedBytecode Code to set (must be <artifact>.deployedBytecode)
 * @returns Instance of the contract.
 */
export async function setCodeAtAddress(name: string, address: string, deployedBytecode: any) {
  await hre.network.provider.send('hardhat_setCode', [address, deployedBytecode])
  return hre.ethers.getContractAt(name, address)
}

export async function transferETH(to: string, amount: bigint = parseEther('1')) {
  const [deployer] = await ethers.getSigners()
  const tx = await deployer.sendTransaction({
    to: to,
    value: amount,
  })

  // Wait for the transaction to be mined
  await tx.wait()
}

const ETH_ROUNDING_TOLERANCE = 10 // 10 wei

chai.Assertion.addChainableMethod('equalWithTolerance', function (expected, overrideTolerance) {
  const actual = this._obj
  const tolerance = overrideTolerance ?? ETH_ROUNDING_TOLERANCE
  return this.assert(
    equalWithTolerance(actual as bigint, expected as bigint, tolerance),
    'expected ' + actual + ' to equal ' + expected + ' tolerance ' + tolerance,
    'expected ' + actual + ' to not equal ' + expected + ' tolerance ' + tolerance,
    'failed'
  )
})
