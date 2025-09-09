import chalk from 'chalk'
import { makeNodeDisklet } from 'disklet'
import { Contract } from 'ethers'
import hre from 'hardhat'
import { DeployOptions, DeployResult, Libraries } from 'hardhat-deploy/types'
import { HardhatRuntimeEnvironment } from 'hardhat/types'

interface CommonDeployArgs extends Omit<DeployOptions, 'from'> {
  hre: HardhatRuntimeEnvironment
  name?: string
  libraries?: Libraries
  log?: boolean
  indent?: number
}

export interface DeployArgs extends CommonDeployArgs {
  contract: string
  args?: any[]
  mock?: boolean
}

export type DeployedContract<C extends Contract> = C & {
  deployResult: DeployResult
}

export const deploy = async <C extends Contract>(
  _args: DeployArgs
): Promise<DeployedContract<C>> => {
  const args = Object.assign(
    {
      skipIfAlreadyDeployed: false,
      indent: 1,
    },
    _args
  )
  const {
    deployments: { deploy, get, getOrNull, fetchIfDifferent },
    getNamedAccounts,
  } = args.hre

  const { deployer } = await getNamedAccounts()

  // If marked as mock, prepend "Mock" to the contract name
  const contractName = `${args.contract}${args.mock ? 'Mock' : ''}`
  const contractDeployName = args.name ?? args.contract

  const deployOpts: DeployOptions = {
    ...args,
    contract: contractName,
    from: deployer,
  }

  const existingDeployment = await getOrNull(contractDeployName)
  const { differences: isDifferent } = await fetchIfDifferent(
    contractDeployName,
    {
      ...deployOpts,
      // We want to always check if the contract is different
      skipIfAlreadyDeployed: false,
    }
  )
  let result: DeployResult

  const skippingDifferent =
    !!existingDeployment && args.skipIfAlreadyDeployed && isDifferent
  if (isDifferent && !skippingDifferent) {
    result = await deploy(contractDeployName, deployOpts)
  } else {
    result = Object.assign(await get(contractDeployName), {
      newlyDeployed: false,
    })
  }

  await onDeployResult({
    result,
    skippingDifferent,
    contract: contractName,
    name: contractDeployName,
    hre,
    indent: args.indent,
  })

  const contract = (await hre.contracts.get(contractDeployName, {
    at: result.address,
  })) as DeployedContract<C>
  contract.deployResult = result
  return contract
}

interface DeployResultArgs {
  result: DeployResult
  skippingDifferent: boolean
  hre: HardhatRuntimeEnvironment
  contract: string
  name: string
  indent?: number
}

const onDeployResult = async (args: DeployResultArgs): Promise<void> => {
  const { result, skippingDifferent, hre, contract, name, indent = 1 } = args

  hre.log('')

  let displayName = name
  if (contract !== name) {
    displayName = `${displayName} (${chalk.bold.italic(contract)})`
  }
  // if (result.artifactName && result.artifactName !== contract) {
  //   displayName = `${displayName} (${chalk.bold.italic(result.artifactName)})`
  // }

  if (result.newlyDeployed) {
    displayName = chalk.greenBright(displayName)
  }
  if (skippingDifferent) {
    displayName = chalk.bgYellow(chalk.black(displayName))
  }

  hre.log(`${displayName}:`, {
    indent,
    star: true,
  })

  if (skippingDifferent) {
    hre.log(
      chalk.yellowBright(
        '⚠️ Skipping Contract Deployment ⚠️ Deployed Contract Different than Compiled Contract ⚠️'
      ),
      {
        indent: indent + 2,
      }
    )
  }

  if (result.newlyDeployed) {
    const tx = await hre.ethers.provider.getTransaction(
      result.receipt!.transactionHash
    )
    const gas = result.receipt
      ? ` with ${chalk.cyan(`${result.receipt.gasUsed} gas`)}`
      : ''
    const gasPrice = ''
    // tx.gasPrice
    //   ? ` @ ${chalk.cyan(
    //       `${hre.ethers.utils.formatUnits(tx.gasPrice, 'gwei')} gwei`
    //     )}`
    //   : ''
    hre.log(
      ` ${chalk.green('new')} ${chalk.bold(result.address)}${gas}${gasPrice}`,
      {
        indent: indent + 2,
      }
    )

    await saveDeploymentBlock(hre.network.name, result.receipt!.blockNumber)
  } else {
    hre.log(`${chalk.yellow('reusing')} ${chalk.bold(result.address)}`, {
      indent: indent + 2,
    })
  }
}

const saveDeploymentBlock = async (
  networkName: string,
  block: number
): Promise<void> => {
  if (networkName === 'hardhat') return

  const disklet = makeNodeDisklet('.')

  const deploymentBlockPath = `deployments/${networkName}/.latestDeploymentBlock`
  const lastDeployment = await disklet
    .getText(deploymentBlockPath)
    .catch(() => {})
  if (!lastDeployment || block > parseInt(lastDeployment)) {
    await disklet.setText(deploymentBlockPath, block.toString())
  }
}
