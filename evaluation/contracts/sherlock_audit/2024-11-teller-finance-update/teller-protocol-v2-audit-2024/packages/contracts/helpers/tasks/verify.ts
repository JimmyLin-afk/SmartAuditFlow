import {
  TASK_VERIFY,
  TASK_VERIFY_GET_VERIFICATION_SUBTASKS,
} from '@nomicfoundation/hardhat-verify/internal/task-names'
import { Manifest } from '@openzeppelin/upgrades-core'
import { ethers, Provider, Result } from 'ethers'
import { subtask, task } from 'hardhat/config'

task(
  'verify-all',
  'Verifies and pushes all deployed contracts to Tenderly'
).setAction(async (args, hre): Promise<void> => {
  const { network } = hre

  // Only continue on a live network
  if (!network.config.live)
    throw new Error('Must be on a live network to submit to Tenderly')

  const fqns = await hre.artifacts.getAllFullyQualifiedNames()

  const deployments = await hre.deployments.all()
  for (const [name, { address, abi }] of Object.entries(deployments)) {
    const fqn = fqns.find((fqn) => fqn.endsWith(`${name}.sol:${name}`))

    const implementation = await hre.upgrades.erc1967
      .getImplementationAddress(address)
      .catch(() => hre.upgrades.beacon.getImplementationAddress(address))
      .catch(() => undefined)
    // if (!implementation) continue

    let constructorArgs: Result | undefined
    if (implementation) {
      const manifest = await Manifest.forNetwork(hre.ethers.provider)
      // manifest.g
      try {
        const { txHash } = await manifest.getDeploymentFromAddress(
          implementation
        )

        if (txHash) {
          constructorArgs = await getConstructorArgsFromTx(
            hre.ethers.provider,
            implementation,
            txHash,
            abi
          )
        }
      } catch {
        continue
      }
    }

    await hre
      .run(TASK_VERIFY, {
        address,
        constructorArgsParams: constructorArgs,
        contract: fqn,
      })
      .catch(console.error)
  }
})

subtask(TASK_VERIFY_GET_VERIFICATION_SUBTASKS).setAction(
  async (_, hre, runSuper): Promise<string[]> => {
    const subtasks = await runSuper()
    return subtasks.concat(['verify:tenderly'])
  }
)

subtask(
  'verify:tenderly',
  'Verifies and pushes all deployed contracts to Tenderly'
)
  .addOptionalParam('address')
  .setAction(async ({ address }, hre): Promise<void> => {
    // TODO: Add contracts to Tenderly
  })

async function getConstructorArgsFromTx(
  provider: Provider,
  address: string,
  transactionHash: string,
  contractABI: any[]
): Promise<Result | undefined> {
  const iface = new ethers.Interface(contractABI)
  if (iface.deploy.inputs.length === 0) return

  const tx = await provider.getTransaction(transactionHash)
  if (!tx) throw new Error('No data found in transaction')

  const defaultArgs = iface.getAbiCoder().getDefaultValue(iface.deploy.inputs)
  const encodedArgs = iface
    .getAbiCoder()
    .encode(iface.deploy.inputs, defaultArgs)
  const data = `0x${tx.data.slice(-encodedArgs.slice(2).length)}`
  return iface.getAbiCoder().decode(iface.deploy.inputs, data)
}
