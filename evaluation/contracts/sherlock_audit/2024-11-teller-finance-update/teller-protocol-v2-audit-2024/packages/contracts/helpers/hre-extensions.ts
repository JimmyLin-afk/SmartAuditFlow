import '@nomicfoundation/hardhat-ethers'
import {
  AdminClient,
  ProposalResponse,
} from '@openzeppelin/defender-admin-client/lib'
import {
  PartialContract,
  ProposalFunctionInputs,
  ProposalStep,
  ProposalTargetFunction,
} from '@openzeppelin/defender-admin-client/lib/models/proposal'
import { fromChainId, Network } from '@openzeppelin/defender-base-client'
import { DefenderHardhatUpgrades } from '@openzeppelin/hardhat-upgrades'
import {
  DeployBeaconOptions,
  DeployProxyOptions,
} from '@openzeppelin/hardhat-upgrades/dist/utils'
import { PrepareUpgradeOptions } from '@openzeppelin/hardhat-upgrades/src/utils/options'
import * as ozUpgrades from '@openzeppelin/upgrades-core'
import chalk from 'chalk'
import {
  BaseContract,
  BigNumberish,
  ContractFactory,
  HDNodeWallet,
  Numeric,
  Signer,
} from 'ethers'
import { ERC20 } from 'generated/typechain'
import 'hardhat-deploy'
import { extendEnvironment } from 'hardhat/config'
import {
  HardhatRuntimeEnvironment,
  ProposeBeaconUpgradeStep,
  ProposeProxyUpgradeStep,
  VirtualExecutionPayload,
} from 'hardhat/types'
import moment from 'moment'

import { getTokens } from '../config'

import { formatMsg, FormatMsgConfig } from './formatMsg'

interface DeployProxyInitArgs {
  initArgs?: any[]
}
interface DeployExtraOpts {
  customName?: string
  libraries?: { [libraryName: string]: string }
}

declare module 'hardhat/types/runtime' {
  interface HardhatRuntimeEnvironment {
    deployProxy: <C = BaseContract>(
      contractName: string,
      opts?: DeployProxyOptions & DeployProxyInitArgs & DeployExtraOpts
    ) => Promise<C>
    deployBeacon: <C = BaseContract>(
      contractName: string,
      opts?: DeployBeaconOptions & DeployExtraOpts
    ) => Promise<C>
    contracts: ContractsExtension
    tokens: TokensExtension
    evm: EVM
    getNamedSigner: (name: string) => Promise<Signer>
    toBN: (amount: BigNumberish, decimals?: BigNumberish) => bigint
    fromBN: (amount: BigNumberish, decimals?: BigNumberish) => bigint
    log: (msg: string, config?: LogConfig) => void
  }

  interface ProposeProxyUpgradeStep {
    proxy: string | BaseContract
    implFactory: ContractFactory
    opts?: PrepareUpgradeOptions & {
      call?: {
        fn: string
        args: any[]
      }
    }
  }
  interface ProposeBeaconUpgradeStep {
    beacon: string | BaseContract
    implFactory: ContractFactory
    opts?: PrepareUpgradeOptions
  }
  // type ProposeUpgradeStep = ProposeProxyUpgradeStep | ProposeBeaconUpgradeStep

  interface HardhatUpgrades extends DefenderHardhatUpgrades {
    proposeCall: (
      contractAddress: string,
      contractImplementation: ContractFactory,
      callFn: string,
      callArgs: any[],

      title: string,
      description: string
    ) => Promise<ProposalResponse>

    proposeUpgradeAndCall: (
      proxyAddress: string,
      implFactory: ContractFactory,
      opts: PrepareUpgradeOptions & {
        title: string
        description: string
        callFn: string
        callArgs: any[]
      }
    ) => Promise<ProposalResponse>
    proposeBatch: ({
      title,
      description,
      _steps,
    }: {
      title: string
      description: string
      _steps: BatchProposalStep[]
    }) => Promise<ProposalResponse>
    proposeBatchTimelock: ({
      title,
      description,
      _steps,
    }: {
      title: string
      description: string
      _steps: BatchProposalStep[]
    }) => Promise<{
      schedule: ProposalResponse
      execute: ProposalResponse
    }>
  }

  interface VirtualExecutionPayload {
    target: string
    value: string
    payload: string // this is just targetFunction and functionInputs encoded to bytes using the abi
    targetFunction: ProposalTargetFunction
    functionInputs: ProposalFunctionInputs
  }
} // end hre module

interface ProposeCallStep {
  contractAddress: string
  contractImplementation: ContractFactory
  callFn: string
  callArgs: any[]
}

export type BatchProposalStep =
  | ProposeCallStep
  | ProposeProxyUpgradeStep
  | ProposeBeaconUpgradeStep

/*
const getStepType = (step: BatchProposalStep): string => {
  if (step.hasOwnProperty('callFn')) {
    return 'call'
  } else if (step.hasOwnProperty('proxy')) {
    return 'upgradeProxy'
  } else if (step.hasOwnProperty('beacon')) {
    return 'upgradeBeacon'
  } else {
    throw new Error('invalid step - cannot ascertain step type')
  }
}*/

function isCallStep(step: BatchProposalStep): step is ProposeCallStep {
  return step.hasOwnProperty('callFn')
}

function isUpgradeProxyStep(
  step: BatchProposalStep
): step is ProposeProxyUpgradeStep {
  return step.hasOwnProperty('proxy')
}

function isUpgradeBeaconStep(
  step: BatchProposalStep
): step is ProposeBeaconUpgradeStep {
  return step.hasOwnProperty('beacon')
}

interface TimelockBatchArgs {
  targets: string[]
  values: string[]
  payloads: string[]
  predecessor: string
  salt: string
  delay: string
}

interface LogConfig extends FormatMsgConfig {
  disable?: boolean
  error?: boolean
}

interface ContractsExtension {
  get: <C extends BaseContract>(
    name: string,
    config?: ContractsGetConfig
  ) => Promise<C>

  proxy: {
    /**
     * Returns the implementation version of a proxy contract.
     *
     * ** Warning **
     *  This function checks storage slot 0 of the proxy contract. If the
     *  proxy contract uses a different storage slot for the implementation version
     *  this function will return an incorrect value.
     *
     * @param proxy
     */
    initializedVersion: (proxy: string | BaseContract) => Promise<number>
  }
}

interface TokensExtension {
  get: (name: string) => Promise<ERC20>
}

interface AdvanceTimeOptions {
  /**
   * When set to `true`, it ensures that a block is not mined for seconds/block
   * internal.
   *
   * In order for the timestamp to take effect a block must be mined. This means
   * that if a view function is called, it will not know about the time update.
   * Use the `mine` option to mine a block after updating the timestamp if used
   * in conjunction with a view function on the next call.
   */
  withoutBlocks?: boolean

  /**
   * Ensures that a block is mined after advancing the next blocks timestamp.
   *
   * This option should be used in conjunction with the `withoutBlocks` option
   * when calling a view function to run any necessary checks.
   */
  mine?: boolean
}

interface EVM {
  /**
   * This sets the timestamp of the next block that executes
   * @param timestamp {moment.Moment} The Moment object that represents a timestamp.
   */
  setNextBlockTimestamp: (timestamp: moment.Moment) => Promise<void>

  /**
   * This increases the next block's timestamp by the specified amount of seconds.
   * @param seconds {Numeric | moment.Duration} Amount of seconds to increase the next block's timestamp by.
   */
  advanceTime: (
    seconds: Numeric | moment.Duration,
    option?: AdvanceTimeOptions
  ) => Promise<void>

  /**
   * Will mine the specified number of blocks locally. This is helpful when functionality
   * requires a certain number of blocks to be processed for values to change.
   * @param blocks {Numeric} Amount of blocks to mine.
   * @param secsPerBlock {number} Determines how many seconds to increase time by for
   *  each block that is mined. Default is 15.
   */
  advanceBlocks: (blocks?: Numeric, secsPerBlock?: number) => Promise<void>

  /**
   * Mines a new block.
   */
  mine: () => Promise<void>

  /**
   * Creates a snapshot of the blockchain in its current state. It then returns a function
   * that can be used to revert back to the state at which the snapshot was taken.
   */
  snapshot: () => Promise<() => Promise<void>>

  /**
   * This allows for functionality to executed within the scope of the specified number
   * of additional blocks to be mined. Once the supplied function to be called within
   * the scope is executed, the blockchain is reverted back to the state it started at.
   * @param blocks {number} The number of blocks that should be mined.
   * @param fn {function} A function that should be executed once blocks have been mined.
   */
  withBlockScope: <T>(blocks: number, fn: () => T) => Promise<T>

  /**
   * Impersonates a supplied address. This allows for the execution of transactions
   * with the `from` field to be the supplied address. This allows for the context
   * of `msg.sender` in a transaction to also be the supplied address.
   *
   * Once you have completed the action of impersonating an address, you may wish to
   * stop impersonating it. To do this, a `stop` function is also returned for
   * convenience. This may also be achieved by calling `evm.stopImpersonating`.
   *
   * To do this:
   *  1. Impersonate an address.
   *  2. Execute a transaction on a contract by connecting the returned signer.
   *    Ex:
   *      const impersonation = await evm.impersonate(0x123)
   *      await contract.connect(impersonation.signer).functionToCall()
   * @param address {string} An address to start impersonating.
   * @return {Promise<ImpersonateReturn>}
   */
  impersonate: (address: string) => Promise<ImpersonateReturn>

  /**
   * It stops the ability to impersonate an address on the local blockchain.
   * @param address {string} An address to stop impersonating.
   */
  stopImpersonating: (address: string) => Promise<void>
}

interface ImpersonateReturn {
  signer: Signer
  stop: () => Promise<void>
}

interface ContractsGetConfig {
  from?: string | HDNodeWallet
  at?: string
}

extendEnvironment((hre) => {
  const { deployments, ethers, network } = hre

  hre.deployProxy = async (contractName, { initArgs, ...opts } = {}) => {
    return await ozDefenderDeploy(hre, 'proxy', contractName, initArgs, opts)
  }

  hre.deployBeacon = async (contractName, opts) => {
    return await ozDefenderDeploy(hre, 'beacon', contractName, opts)
  }

  hre.contracts = {
    async get<C = BaseContract>(
      name: string,
      config?: ContractsGetConfig
    ): Promise<C> {
      const { abi, address } = await deployments
        .get(name)
        .catch(async () =>
          hre.network.name === 'localhost'
            ? await hre.artifacts.readArtifact(name)
            : await deployments.getArtifact(name)
        )
        .then((artifact) => ({
          abi: artifact.abi,
          address:
            config?.at ?? ('address' in artifact ? artifact.address : null),
        }))

      if (address == null)
        throw new Error(
          `No deployment exists for ${name}. If expected, supply an address (config.at)`
        )

      let signer: Signer
      if (config?.from) {
        signer =
          typeof config.from === 'string'
            ? await ethers.provider.getSigner(config.from)
            : config.from
      } else {
        signer = await hre.getNamedSigner('deployer')
      }
      const contract = await ethers.getContractAt(abi, address, signer)

      return contract as C
    },

    proxy: {
      initializedVersion: async (
        proxy: string | BaseContract
      ): Promise<number> => {
        const address =
          typeof proxy === 'string' ? proxy : await proxy.getAddress()
        const isProxy = await ozUpgrades.isTransparentOrUUPSProxy(
          hre.ethers.provider,
          address
        )
        if (!isProxy) throw new Error(`Address ${address} is not a proxy`)
        const storage = await ethers.provider.getStorage(address, 0)
        return Number(storage)
      },
    },
  }

  hre.tokens = {
    async get(nameOrAddress: string): Promise<ERC20> {
      let address: string
      if (ethers.isAddress(nameOrAddress)) {
        address = nameOrAddress
      } else {
        const tokens = await getTokens(hre)
        address = tokens.all[(nameOrAddress as string).toUpperCase()]
        if (!address) throw new Error(`Token ${nameOrAddress} not found`)
      }
      return await ethers.getContractAt('ERC20', address)
    },
  }

  hre.getNamedSigner = async (name: string): Promise<Signer> => {
    const accounts = await hre.getNamedAccounts()
    const signer = await ethers.provider.getSigner(accounts[name])
    // NOTE: This is a workaround for the gas estimation bug on Mantle RPC nodes
    if (typeof hre.network.config.gas === 'number') {
      signer.provider.estimateGas = () =>
        Promise.resolve(BigInt(hre.network.config.gas))
    }
    return signer
  }

  hre.evm = {
    async setNextBlockTimestamp(timestamp: moment.Moment): Promise<void> {
      await network.provider.send('evm_setNextBlockTimestamp', [
        timestamp.unix(),
      ])
    },

    async advanceTime(
      seconds: Numeric | moment.Duration,
      options?: AdvanceTimeOptions
    ): Promise<void> {
      const secs = moment.isDuration(seconds)
        ? seconds
        : moment.duration(seconds.toString(), 's')
      if (options?.withoutBlocks) {
        const block = await ethers.provider.getBlock('latest')
        if (!block) throw new Error('Could not get latest block')

        const timestamp = moment(
          secs.add(block.timestamp, 's').asMilliseconds()
        )
        await this.setNextBlockTimestamp(timestamp)
        if (options?.mine) await this.mine()
      } else {
        const secsPerBlock = 15
        const blocks = BigInt(secs.asSeconds()) / BigInt(secsPerBlock)
        await this.advanceBlocks(blocks, secsPerBlock)
      }
    },

    async advanceBlocks(blocks = 1, secsPerBlock = 15): Promise<void> {
      for (let block = 0; block < blocks; block++) {
        await network.provider.send('evm_increaseTime', [secsPerBlock])
        await this.mine()
      }
    },

    async mine(): Promise<void> {
      await network.provider.send('evm_mine')
    },

    async snapshot(): Promise<() => Promise<void>> {
      const id = await network.provider.send('evm_snapshot')
      return async () => {
        await network.provider.send('evm_revert', [id])
      }
    },

    async withBlockScope(blocks: number, fn: () => any): Promise<any> {
      const revert = await this.snapshot()
      await this.advanceBlocks(blocks)
      const result = await fn()
      await revert()
      // eslint-disable-next-line @typescript-eslint/no-unsafe-return
      return result
    },

    async impersonate(address: string): Promise<ImpersonateReturn> {
      await network.provider.request({
        method: 'hardhat_impersonateAccount',
        params: [address],
      })
      const signer = await ethers.provider.getSigner(address)
      return {
        signer,
        stop: async () => await this.stopImpersonating(address),
      }
    },

    async stopImpersonating(address: string): Promise<void> {
      await network.provider.request({
        method: 'hardhat_stopImpersonatingAccount',
        params: [address],
      })
    },
  }

  hre.toBN = (amount: BigNumberish, decimals?: BigNumberish): bigint => {
    if (typeof amount === 'string') {
      return ethers.parseUnits(amount, decimals)
    }

    const num = BigInt(amount)
    if (decimals) {
      return num * 10n ** BigInt(decimals)
    }
    return num
  }

  hre.fromBN = (amount: BigNumberish, decimals?: BigNumberish): bigint => {
    const num = BigInt(amount)
    if (decimals) {
      return num / 10n ** BigInt(decimals)
    }
    return num
  }

  hre.log = (msg: string, config: LogConfig = {}): void => {
    const { disable = process.env.DISABLE_LOGS === 'true' } = config

    if (disable) return
    const fn = config?.error ? process.stderr : process.stdout
    fn.write(formatMsg(msg, config))
  }

  const defenderAdmin = new AdminClient({
    apiKey: hre.config.defender!.apiKey,
    apiSecret: hre.config.defender!.apiSecret,
  })

  hre.upgrades.proposeCall = async (
    contractAddress,
    contractImplementation,
    callFn,
    callArgs,
    title,
    description
  ): Promise<ProposalResponse> => {
    const functionInputsInterface = JSON.parse(
      JSON.stringify(
        contractImplementation.interface.getFunction(callFn)?.inputs
      )
    )

    if (typeof functionInputsInterface == 'undefined') {
      throw new Error('Function inputs undefined')
    }

    const { protocolOwnerSafe } = await hre.getNamedAccounts()
    return await defenderAdmin.createProposal({
      contract: {
        address: contractAddress,
        network: await getOZNetwork(hre),
        abi: JSON.stringify(
          contractImplementation.interface.fragments.map((fragment) =>
            JSON.parse(fragment.format('json'))
          )
        ),
      },
      title: title,
      description: description,
      type: 'custom',

      functionInterface: {
        name: callFn,
        inputs: functionInputsInterface,
      },
      functionInputs: callArgs,
      viaType: 'Gnosis Safe',
      via: protocolOwnerSafe,
      // set simulate to true
      // simulate: true,
    })
  }

  hre.upgrades.proposeUpgradeAndCall = async (
    proxyAddress,
    implFactory,
    { title, description, callFn, callArgs, ...opts }
  ): Promise<ProposalResponse> => {
    const newImpl = await hre.upgrades.prepareUpgrade(
      proxyAddress,
      implFactory,
      opts
    )
    const newImplAddr =
      typeof newImpl === 'string'
        ? newImpl
        : await newImpl.wait(1).then((r) => {
            if (!r?.contractAddress) throw new Error('No contract address')
            return r.contractAddress
          })

    await updateDeploymentAbi({
      hre,
      address: proxyAddress,
      abi: JSON.parse(implFactory.interface.formatJson()),
    })

    const proxyAdmin = await hre.upgrades.admin.getInstance()
    const { protocolOwnerSafe } = await hre.getNamedAccounts()

    return await defenderAdmin.createProposal({
      contract: {
        address: await proxyAdmin.getAddress(),
        network: await getOZNetwork(hre),
        abi: JSON.stringify(
          proxyAdmin.interface.fragments.map((fragment) =>
            JSON.parse(fragment.format('json'))
          )
        ),
      },
      title: title,
      description: description,
      type: 'custom',
      // metadata: {
      //   sendTo: '0xA91382E82fB676d4c935E601305E5253b3829dCD',
      //   sendValue: '10000000000000000',
      //   sendCurrency: {
      //     name: 'Ethereum',
      //     symbol: 'ETH',
      //     decimals: 18,
      //     type: 'native',
      //   },
      // },
      functionInterface: {
        name: 'upgradeAndCall',
        inputs: [
          {
            name: 'proxy',
            type: 'address',
          },
          { name: 'implementation', type: 'address' },
          { name: 'data', type: 'bytes' },
        ],
      },
      functionInputs: [
        proxyAddress,
        newImplAddr,
        implFactory.interface.encodeFunctionData(callFn, callArgs),
      ],
      viaType: 'Gnosis Safe',
      via: protocolOwnerSafe,
      // set simulate to true
      // simulate: true,
    })
  }

  hre.upgrades.proposeBatch = async ({
    title,
    description,
    _steps,
  }): Promise<ProposalResponse> => {
    const network = await getOZNetwork(hre)

    const { protocolOwnerSafe } = await hre.getNamedAccounts()

    const contracts: PartialContract[] = []
    const proposalSteps: ProposalStep[] = []
    for (const step of _steps) {
      const virtualExecutionPayload: VirtualExecutionPayload =
        await getVirtualExecutionPayloadForStep(step, hre)

      const toContractAddress = virtualExecutionPayload.target

      contracts.push({
        address: toContractAddress,
        // @ts-ignore
        network,
      })
      proposalSteps.push({
        contractId: `${network}-${toContractAddress}`,
        type: 'custom',
        targetFunction: virtualExecutionPayload.targetFunction,
        functionInputs: virtualExecutionPayload.functionInputs,
      })
    } // end steps loop

    return await defenderAdmin.createProposal({
      contract: contracts,
      title: title,
      description: description,
      type: 'batch',
      viaType: 'Gnosis Safe',
      via: protocolOwnerSafe,
      metadata: {},
      steps: proposalSteps,
    })
  }

  hre.upgrades.proposeBatchTimelock = async ({
    title,
    description,
    _steps,
  }): Promise<{
    schedule: ProposalResponse
    execute: ProposalResponse
  }> => {
    const delay = moment.duration(3, 'minutes').asSeconds().toString()

    // build the timelock batch args from the steps

    const virtualExecutionPayloadArray: VirtualExecutionPayload[] = []

    const steps = Array.isArray(_steps) ? _steps : [_steps]
    for (const step of steps) {
      const virtualExecutionPayload: VirtualExecutionPayload =
        await getVirtualExecutionPayloadForStep(step, hre)

      virtualExecutionPayloadArray.push(virtualExecutionPayload)
    } // end loop for steps

    const timelockBatchArgs: TimelockBatchArgs = {
      targets: virtualExecutionPayloadArray.map((x) => x.target),
      values: virtualExecutionPayloadArray.map((x) => x.value),
      payloads: virtualExecutionPayloadArray.map((x) => x.payload),
      predecessor: ethers.encodeBytes32String(''),
      salt: ethers.encodeBytes32String(''), // should this have a value ?
      delay,
    }

    // @ts-ignore
    return await createScheduledBatchProposal(hre, {
      title,
      description,
      timelockBatchArgs,
    })
  }
})

async function getOZNetwork(hre: HardhatRuntimeEnvironment): Promise<Network> {
  const chainId = await hre.getChainId()
  const network = fromChainId(Number(chainId))
  if (!network) throw new Error(`Unknown chain id ${chainId}`)
  return network
}

const getVirtualExecutionPayloadForStep = async (
  step: BatchProposalStep,
  hre: HardhatRuntimeEnvironment
): Promise<VirtualExecutionPayload> => {
  if (isCallStep(step)) {
    return await getVirtualPayloadForCall(step, hre)
  } else if (isUpgradeProxyStep(step)) {
    return await getVirtualPayloadForUpgradeProxy(step, hre)
  } else if (isUpgradeBeaconStep(step)) {
    return await getVirtualPayloadForUpgradeBeacon(step, hre)
  } else {
    throw new Error('Invalid step type - cannot add batch args')
  }
}

const getVirtualPayloadForCall = async (
  step: ProposeCallStep,

  hre: HardhatRuntimeEnvironment
): Promise<VirtualExecutionPayload> => {
  const contractImpl = step.contractImplementation

  const iface = contractImpl.interface

  const functionInputsInterface = JSON.parse(
    JSON.stringify(iface.getFunction(step.callFn)?.inputs)
  )

  const targetFunction: ProposalTargetFunction = {
    name: step.callFn,
    inputs: functionInputsInterface,
  }

  const functionInputs: ProposalFunctionInputs = step.callArgs

  if (typeof targetFunction.name == 'undefined') {
    throw new Error('Target function for call is missing name')
  }

  return {
    value: '0',
    target: step.contractAddress,

    targetFunction,
    functionInputs,

    // is this right ?
    payload: iface.encodeFunctionData(targetFunction.name, functionInputs),
  }
}

const getVirtualPayloadForUpgradeProxy = async (
  step: ProposeProxyUpgradeStep,

  hre: HardhatRuntimeEnvironment
): Promise<VirtualExecutionPayload> => {
  const proxyAdmin = await hre.upgrades.admin.getInstance()

  const refAddress =
    typeof step.proxy === 'string' ? step.proxy : await step.proxy.getAddress()
  const call = step.opts?.call

  const newImpl = await hre.upgrades.prepareUpgrade(
    refAddress,
    step.implFactory,
    step.opts
  )
  const newImplAddr =
    typeof newImpl === 'string'
      ? newImpl
      : await newImpl.wait(1).then((r) => {
          if (!r?.contractAddress) throw new Error('No contract address')
          return r.contractAddress
        })

  await updateDeploymentAbi({
    hre,
    address: refAddress,
    abi: JSON.parse(step.implFactory.interface.formatJson()),
  })

  // produce the timelock batch args
  const value = '0'
  // batchArgs.values.push('0')

  // batchArgs.targets.push(await proxyAdmin.getAddress())
  const target = await proxyAdmin.getAddress()

  const targetFunction: ProposalTargetFunction = call
    ? {
        name: 'upgradeAndCall',
        inputs: [
          {
            name: 'proxy',
            type: 'address',
          },
          {
            name: 'implementation',
            type: 'address',
          },
          { name: 'data', type: 'bytes' },
        ],
      }
    : {
        name: 'upgrade',
        inputs: [
          {
            name: 'proxy',
            type: 'address',
          },
          {
            name: 'implementation',
            type: 'address',
          },
        ],
      }
  /*
  console.log('creating function inputs ', {
    call
  })

  for (let fragment of step.implFactory.interface.fragments) {
    console.log(fragment)
  }*/

  const functionInputs: ProposalFunctionInputs = call
    ? [
        refAddress,
        newImplAddr,
        step.implFactory.interface.encodeFunctionData(call.fn, call.args),
      ]
    : [refAddress, newImplAddr]

  if (typeof targetFunction.name === 'undefined') {
    throw new Error('Missing target function name')
  }

  const iface = proxyAdmin.interface

  const payload: string = iface.encodeFunctionData(
    targetFunction.name,
    functionInputs
  )

  return { value, target, payload, targetFunction, functionInputs }
}

const getVirtualPayloadForUpgradeBeacon = async (
  step: ProposeBeaconUpgradeStep,

  hre: HardhatRuntimeEnvironment
): Promise<VirtualExecutionPayload> => {
  const refAddress =
    typeof step.beacon === 'string'
      ? step.beacon
      : await step.beacon.getAddress()

  const newImpl = await hre.upgrades.prepareUpgrade(
    refAddress,
    step.implFactory,
    step.opts
  )
  const newImplAddr =
    typeof newImpl === 'string'
      ? newImpl
      : await newImpl.wait(1).then((r) => {
          if (!r?.contractAddress) throw new Error('No contract address')
          return r.contractAddress
        })

  await updateDeploymentAbi({
    hre,
    address: refAddress,
    abi: JSON.parse(step.implFactory.interface.formatJson()),
  })

  const value = '0'
  const target = refAddress
  // produce the timelock batch args
  // batchArgs.values.push('0')

  // batchArgs.targets.push(refAddress)

  const iface = new hre.ethers.Interface([
    hre.ethers.FunctionFragment.from({
      inputs: [
        {
          name: 'newImplementation',
          type: 'address',
        },
      ],
      name: 'upgradeTo',
      stateMutability: 'nonpayable',
      type: 'function',
    }),
  ])

  const targetFunction: ProposalTargetFunction = {
    name: 'upgradeTo',
    inputs: [
      {
        name: 'newImplementation',
        type: 'address',
      },
    ],
  }

  const functionInputs: ProposalFunctionInputs = [newImplAddr]

  if (typeof targetFunction.name == 'undefined') {
    throw new Error('Missing target function name')
  }

  const payload = iface.encodeFunctionData(targetFunction.name, functionInputs)
  // batchArgs.payloads.push(iface.encodeFunctionData('upgradeTo', [newImplAddr]))

  return { value, target, payload, targetFunction, functionInputs }
}

const PendingProposalsFile = 'pending-proposals.json'
const storePendingProposal = async (
  hre: HardhatRuntimeEnvironment,
  {
    title,
    description,
    timelockBatchArgs,
  }: {
    title: string
    description: string
    timelockBatchArgs: TimelockBatchArgs
  }
): Promise<void> => {
  const proposals = await hre.deployments
    .readDotFile(PendingProposalsFile)
    .then<Record<string, TimelockBatchArgs | undefined>>((contents) =>
      JSON.parse(contents)
    )
    .catch(() => undefined)
    .then((contents) => contents ?? {})

  const proposal: TimelockBatchArgs = proposals[title] ?? {
    targets: [],
    values: [],
    payloads: [],
    predecessor: '',
    salt: '',
    delay: '',
  }
  proposals[title] = proposal

  if (
    timelockBatchArgs.predecessor === proposal.predecessor &&
    timelockBatchArgs.salt === proposal.salt
  ) {
    proposal.targets.push(...timelockBatchArgs.targets)
    proposal.values.push(...timelockBatchArgs.values)
    proposal.payloads.push(...timelockBatchArgs.payloads)
  }
  await hre.deployments.saveDotFile(
    PendingProposalsFile,
    JSON.stringify(proposal)
  )
}

const createScheduledBatchProposal = async (
  hre: HardhatRuntimeEnvironment,
  {
    title,
    description,
    timelockBatchArgs,
  }: {
    title: string
    description: string
    timelockBatchArgs: TimelockBatchArgs
  }
) => {
  const network = await getOZNetwork(hre)
  const defenderAdmin = new AdminClient({
    apiKey: hre.config.defender!.apiKey,
    apiSecret: hre.config.defender!.apiSecret,
  })

  const { protocolOwnerSafe, protocolTimelock } = await hre.getNamedAccounts()

  return {
    schedule: await defenderAdmin.createProposal({
      title: `${title} (Schedule Timelock)`,
      description: description,
      type: 'custom',
      viaType: 'Gnosis Safe',
      via: protocolOwnerSafe,
      contract: {
        name: 'TellerV2 Protocol Timelock',
        network,
        address: protocolTimelock,
      },
      functionInterface: {
        name: 'scheduleBatch',
        inputs: [
          { name: 'targets', type: 'address[]' },
          { name: 'values', type: 'uint256[]' },
          { name: 'payloads', type: 'bytes[]' },
          { name: 'predecessor', type: 'bytes32' },
          { name: 'salt', type: 'bytes32' },
          { name: 'delay', type: 'uint256' },
        ],
      },
      functionInputs: [
        timelockBatchArgs.targets,
        timelockBatchArgs.values,
        timelockBatchArgs.payloads,
        timelockBatchArgs.predecessor,
        timelockBatchArgs.salt,
        timelockBatchArgs.delay,
      ],
    }),
    execute: await defenderAdmin.createProposal({
      title: `${title} (Execute Timelock)`,
      description: description,
      type: 'custom',
      viaType: 'Gnosis Safe',
      via: protocolOwnerSafe,
      contract: {
        name: 'TellerV2 Protocol Timelock',
        network,
        address: protocolTimelock,
      },
      functionInterface: {
        name: 'executeBatch',
        inputs: [
          { name: 'targets', type: 'address[]' },
          { name: 'values', type: 'uint256[]' },
          { name: 'payloads', type: 'bytes[]' },
          { name: 'predecessor', type: 'bytes32' },
          { name: 'salt', type: 'bytes32' },
        ],
      },
      functionInputs: [
        timelockBatchArgs.targets,
        timelockBatchArgs.values,
        timelockBatchArgs.payloads,
        timelockBatchArgs.predecessor,
        timelockBatchArgs.salt,
      ],
    }),
  }
}

type OZDefenderDeployOpts = (DeployProxyOptions | DeployBeaconOptions) &
  DeployExtraOpts

async function ozDefenderDeploy<C = BaseContract>(
  hre: HardhatRuntimeEnvironment,
  deployType: 'proxy' | 'beacon',
  contractName: string,
  opts?: OZDefenderDeployOpts
): Promise<C>
async function ozDefenderDeploy<C = BaseContract>(
  hre: HardhatRuntimeEnvironment,
  deployType: 'proxy' | 'beacon',
  contractName: string,
  initArgs?: unknown[],
  opts?: DeployProxyOptions
): Promise<C>
async function ozDefenderDeploy<C = BaseContract>(
  hre: HardhatRuntimeEnvironment,
  deployType: 'proxy' | 'beacon',
  contractName: string,
  initArgs: unknown[] | OZDefenderDeployOpts = [],
  opts: OZDefenderDeployOpts = {}
): Promise<C> {
  const isProxy = deployType === 'proxy'
  const contractTypeStr = isProxy ? 'Proxy' : 'Beacon'

  if (!Array.isArray(initArgs)) {
    // eslint-disable-next-line no-param-reassign
    opts = initArgs
    // eslint-disable-next-line no-param-reassign
    initArgs = []
  }

  const saveName = opts.customName ?? contractName

  hre.log('----------')
  hre.log('')
  hre.log(
    `${chalk.underline('Contract')} (${chalk.italic(
      contractTypeStr
    )}): ${chalk.bold(`${saveName}`)} ${
      saveName !== contractName ? ` (${chalk.italic(`${contractName}`)})` : ''
    }`
  )
  hre.log('')

  let proxy: BaseContract
  const deployer = await hre.getNamedSigner('deployer')
  const implFactory = await hre.ethers.getContractFactory(contractName, {
    libraries: opts.libraries,
    signer: deployer,
  })
  const existingDeployment = await hre.deployments.getOrNull(saveName)
  if (existingDeployment) {
    hre.log(`${chalk.bold.yellow(`Existing ${deployType} deployment found`)}`, {
      indent: 1,
    })

    proxy = await hre.ethers.getContractAt(
      contractName,
      existingDeployment.address,
      deployer
    )
  } else {
    hre.log(`${chalk.bold.green(`Deploying new ${deployType}...`)}`, {
      indent: 1,
    })

    if (isProxy) {
      proxy = await hre.upgrades.deployProxy(implFactory, initArgs, opts)
    } else {
      proxy = await hre.upgrades.deployBeacon(implFactory, opts)
    }
  }

  const proxyLabel = chalk.bold(`${contractTypeStr}: `)
  const implementationLabel = chalk.bold(`Implementation: `)
  const maxLabelLength = Math.max(proxyLabel.length, implementationLabel.length)
  hre.log(proxyLabel, {
    star: false,
    nl: false,
    pad: { start: { length: maxLabelLength, char: ' ' } },
  })
  hre.log(`${await proxy.getAddress()}`, { star: false })
  await proxy.waitForDeployment()
  const proxyAddress = await proxy.getAddress()
  const implementation = isProxy
    ? await hre.upgrades.erc1967.getImplementationAddress(proxyAddress)
    : await hre.upgrades.beacon.getImplementationAddress(proxyAddress)
  hre.log(implementationLabel, {
    star: false,
    nl: false,
    pad: { start: { length: maxLabelLength, char: ' ' } },
  })
  hre.log(`${implementation}`, { star: false })

  const abi = JSON.parse(implFactory.interface.formatJson())
  const deployTx = proxy.deploymentTransaction()
  const transactionHash = deployTx?.hash ?? existingDeployment?.transactionHash
  const receipt = Object.assign({}, deployTx, existingDeployment?.receipt)
  await hre.deployments.save(saveName, {
    address: proxyAddress,
    implementation,
    abi,
    transactionHash,
    receipt,
  })

  hre.log('')
  hre.log('----------')

  return proxy as C
}

async function updateDeploymentAbi({
  hre,
  address,
  abi,
}: {
  hre: HardhatRuntimeEnvironment
  address: string
  abi: any[]
}): Promise<void> {
  const deployments = await hre.deployments.all()
  for (const [contractName, deployment] of Object.entries(deployments)) {
    if (deployment.address === address) {
      await hre.deployments.save(contractName, {
        ...deployment,
        abi,
      })
    }
  }
}
