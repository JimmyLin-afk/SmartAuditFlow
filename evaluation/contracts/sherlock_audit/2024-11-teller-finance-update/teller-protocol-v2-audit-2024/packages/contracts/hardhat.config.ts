// This adds support for typescript paths mappings
import 'tsconfig-paths/register'
import 'hardhat-contract-sizer'
import 'hardhat-deploy'
import 'hardhat-gas-reporter'
import '@nomicfoundation/hardhat-ethers'
import '@typechain/hardhat'
import 'solidity-coverage'
import '@openzeppelin/hardhat-upgrades'
import '@nomicfoundation/hardhat-verify'

import fs from 'fs'
import path from 'path'

import { HardhatEthersHelpers } from '@nomicfoundation/hardhat-ethers/types'
// import { logger as tenderlyLogger2 } from '@teller-protocol/hardhat-tenderly/dist/utils/logger'
import chalk from 'chalk'
import { config } from 'dotenv'
import {
  Signer,
  isAddress,
  getAddress,
  formatUnits,
  parseUnits,
  parseEther,
  TransactionRequest,
  TransactionReceipt,
  ethers,
} from 'ethers'
import { HardhatUserConfig, task } from 'hardhat/config'
import {
  HardhatNetworkHDAccountsUserConfig,
  NetworkUserConfig,
} from 'hardhat/types'
import rrequire from 'helpers/rrequire'
import semver from 'semver'
// import { logger as tenderlyLogger } from 'tenderly/utils/logger'

const NODE_VERSION = 'v16'
if (!semver.satisfies(process.version, NODE_VERSION))
  throw new Error(
    `Incorrect NodeJS version being used (${process.version}). Expected: ${NODE_VERSION}`
  )

config()

const {
  COMPILING,
  CMC_KEY,
  DEFAULT_NETWORK,
  HARDHAT_DEPLOY_FORK,
  SAVE_GAS_REPORT,
  SKIP_SIZER,
  TESTING,
  ALCHEMY_API_KEY,
  DEFENDER_API_KEY,
  DEFENDER_API_SECRET,
} = process.env

const isCompiling = COMPILING === 'true'
const skipContractSizer = SKIP_SIZER === 'true' && !isCompiling
if (!isCompiling) {
  rrequire(path.resolve(__dirname, 'helpers', 'tasks'))
  require('helpers/hre-extensions')
}

const isTesting = TESTING === '1'
if (isTesting) {
  require('helpers/chai-helpers')
}

//
// Select the network you want to deploy to here:
//
const defaultNetwork = DEFAULT_NETWORK ?? 'hardhat'

const pathToMnemonic = path.resolve(__dirname, 'mnemonic.secret')

export const getMnemonic = (): string => {
  try {
    return fs.readFileSync(pathToMnemonic).toString().trim()
  } catch (e) {
    // @ts-ignore
    if (defaultNetwork !== 'localhost') {
      console.log(
        '☢️ WARNING: No mnemonic file created for a deploy account. Try `yarn run generate` and then `yarn run account`.'
      )
    }
  }
  return ''
}

const accounts: HardhatNetworkHDAccountsUserConfig = {
  mnemonic: getMnemonic(),
  count: 15,
  accountsBalance: parseEther('100000000').toString(),
}

type NetworkNames =
  | 'mainnet'
  | 'polygon'
  | 'arbitrum'
  | 'base'
  | 'mantle'
  | 'sepolia'
  | 'mumbai'
  | 'goerli'
  | 'mantle-testnet'
  | 'tenderly'
const networkUrls: Record<NetworkNames, string> = {
  // Main Networks
  mainnet:
    process.env.MAINNET_RPC_URL ??
    (ALCHEMY_API_KEY
      ? `https://eth-mainnet.g.alchemy.com/v2/${ALCHEMY_API_KEY}`
      : ''),
  polygon:
    process.env.POLYGON_RPC_URL ??
    (ALCHEMY_API_KEY
      ? `https://polygon-mainnet.g.alchemy.com/v2/${ALCHEMY_API_KEY}`
      : ''),
  arbitrum:
    process.env.ARBITRUM_RPC_URL ??
    (ALCHEMY_API_KEY
      ? `https://arb-mainnet.g.alchemy.com/v2/${ALCHEMY_API_KEY}`
      : ''),
  base: 'https://mainnet.base.org/',
  mantle: 'https://rpc.mantle.xyz',

  // Test Networks
  sepolia:
    process.env.SEPOLIA_RPC_URL ??
    (ALCHEMY_API_KEY
      ? `https://eth-sepolia.g.alchemy.com/v2/${ALCHEMY_API_KEY}`
      : ''),
  mumbai:
    process.env.MUMBAI_RPC_URL ??
    (ALCHEMY_API_KEY
      ? `https://polygon-mumbai.g.alchemy.com/v2/${ALCHEMY_API_KEY}`
      : ''),
  goerli:
    process.env.GOERLI_RPC_URL ??
    (ALCHEMY_API_KEY
      ? `https://eth-goerli.g.alchemy.com/v2/${ALCHEMY_API_KEY}`
      : ''),
  'mantle-testnet': 'https://rpc.testnet.mantle.xyz',
  tenderly: process.env.TENDERLY_RPC_URL ?? '',
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
const getLatestDeploymentBlock = (networkName: string): number | undefined => {
  try {
    return parseInt(
      fs
        .readFileSync(
          path.resolve(
            __dirname,
            'deployments',
            networkName,
            '.latestDeploymentBlock'
          )
        )
        .toString()
    )
  } catch {
    // Network deployment does not exist
  }
}

const networkConfig = (config: NetworkUserConfig): NetworkUserConfig => ({
  live: false,
  // gas: 'auto',
  ...config,
  accounts,
})

/*
      📡 This is where you configure your deploy configuration for 🏗 scaffold-eth

      check out `packages/scripts/deploy.js` to customize your deployment

      out of the box it will auto deploy anything in the `contracts` folder and named *.sol
      plus it will use *.args for constructor args
*/

// eslint-disable-next-line @typescript-eslint/consistent-type-assertions
export default <HardhatUserConfig>{
  defaultNetwork,

  etherscan: {
    apiKey: {
      // Main Networks
      mainnet: process.env.ETHERSCAN_VERIFY_API_KEY,
      polygon: process.env.POLYGONSCAN_VERIFY_API_KEY,
      arbitrumOne: process.env.ARBISCAN_VERIFY_API_KEY,
      base: process.env.BASESCAN_VERIFY_API_KEY,
      mantle: process.env.MANTLE_VERIFY_API_KEY ?? 'xyz',

      // Test Networks
      sepolia: process.env.ETHERSCAN_VERIFY_API_KEY,
      goerli: process.env.ETHERSCAN_VERIFY_API_KEY,
      mumbai: process.env.POLYGONSCAN_VERIFY_API_KEY,
      'mantle-testnet': process.env.MANTLE_VERIFY_API_KEY ?? 'xyz',
    },
    customChains: [
      {
        network: 'base',
        chainId: 8453,
        urls: {
          apiURL: 'https://api.basescan.org/api',
          browserURL: 'https://basescan.org',
        },
      },
      {
        network: 'mantle',
        chainId: 5000,
        urls: {
          apiURL: 'https://explorer.mantle.xyz/api',
          browserURL: 'https://explorer.mantle.xyz',
        },
      },
      {
        network: 'mantle-testnet',
        chainId: 5001,
        urls: {
          apiURL: 'https://explorer.testnet.mantle.xyz/api',
          browserURL: 'https://explorer.testnet.mantle.xyz',
        },
      },
    ],
  },

  defender: {
    apiKey: DEFENDER_API_KEY,
    apiSecret: DEFENDER_API_SECRET,
  },

  tenderly: {
    username: 'teller',
    project: 'v2',
    privateVerification: true,
    forkNetwork: networkUrls.tenderly,
  },

  paths: {
    cache: './generated/cache',
    artifacts: './generated/artifacts',
    sources: './contracts',
  },

  typechain: {
    outDir: './generated/typechain',
    target: 'ethers-v6',
  },

  external: {
    contracts: [
      {
        artifacts: './node_modules/hardhat-deploy/extendedArtifacts',
      },
    ],
  },

  solidity: {
    compilers: [
      {
        version: '0.8.11',
        settings: {
          optimizer: {
            enabled: true, // !isTesting, //need this for now due to large size of tellerV2.test
            runs: 200,
          },
        },
      },
    ],
  },

  ovm: {
    solcVersion: '0.8.4',
  },

  contractSizer: {
    runOnCompile: !skipContractSizer,
    alphaSort: false,
    disambiguatePaths: false,
  },

  /**
   * gas reporter configuration that let's you know
   * an estimate of gas for contract deployments and function calls
   * More here: https://hardhat.org/plugins/hardhat-gas-reporter.html
   */
  gasReporter: {
    enabled: true,
    currency: 'USD',
    coinmarketcap: CMC_KEY,
    outputFile: SAVE_GAS_REPORT ? 'gas-reporter.txt' : undefined,
    noColors: !!SAVE_GAS_REPORT,
    showMethodSig: false,
    showTimeSpent: true,
  },

  namedAccounts: {
    deployer: {
      default: 0, // here this will by default take the first account as deployer
      31337: '0x65B38b3Cd7eFe502DB579c16ECB5B49235d0DAd0', // use the goerli deployer address for hardhat forking
    },
    borrower: 1,
    lender: 2,
    lender2: 3,
    dao: 4,
    marketowner: 5,
    funder: 10,
    rando: 14,
    protocolOwnerSafe: {
      31337: 7,
      1: '0x9E3bfee4C6b4D28b5113E4786A1D9812eB3D2Db6',
      5: '0x0061CA4F1EB8c3FF93Df074061844d3dd4dC0377',
      137: '0xFea0FB908E31567CaB641865212cF76BE824D848',
      5000: '0x4496c03dA72386255Bf4af60b3CCe07787d3dCC2',
      8453: '0x2f74c448CF6d613bEE183fE35dB0c9AC5084F66A',
      42161: '0xD9149bfBfB29cC175041937eF8161600b464051B',
      11155111: '0xb1ff461BB751B87f4F791201a29A8cFa9D30490c',
    },
    protocolTimelock: {
      31337: 8,
      1: '0xe6774DAAEdf6e95b222CD3dE09456ec0a46672C4',
      5: '0x0e8A920f0338b94828aE84a7C227bC17F3a02f86',
      137: '0x6eB9b34913Bd96CA2695519eD0F8B8752d43FD2b',
      5000: '0x6BBf498C429C51d05bcA3fC67D2C720B15FC73B8',
      8453: '0x6BBf498C429C51d05bcA3fC67D2C720B15FC73B8',
      42161: '0x6BBf498C429C51d05bcA3fC67D2C720B15FC73B8',
      11155111: '0xFe5394B67196EA95301D6ECB5389E98A02984cC2',
    },
  },

  // if you want to deploy to a testnet, mainnet, or xdai, you will need to configure:
  // 1. An Infura key (or similar)
  // 2. A private key for the deployer
  // DON'T PUSH THESE HERE!!!
  // An `example.env` has been provided in the Hardhat root. Copy it and rename it `.env`
  // Follow the directions, and uncomment the network you wish to deploy to.

  networks: {
    // Local Networks
    hardhat: networkConfig({
      chainId: 31337,
      allowUnlimitedContractSize: true,
      saveDeployments: !isTesting,
      forking:
        HARDHAT_DEPLOY_FORK == null
          ? undefined
          : {
              enabled: true,
              url: networkUrls[HARDHAT_DEPLOY_FORK as keyof typeof networkUrls],
              // blockNumber: getLatestDeploymentBlock(HARDHAT_DEPLOY_FORK),
            },
    }),
    localhost: networkConfig({
      url: 'http://localhost:8545',
    }),

    // Main Networks
    mainnet: networkConfig({
      url: networkUrls.mainnet,
      chainId: 1,
      live: true,
      // gasPrice: Number(ethers.parseUnits('130', 'gwei')),

      verify: {
        etherscan: {
          apiKey: process.env.ETHERSCAN_VERIFY_API_KEY,
        },
      },
    }),
    polygon: networkConfig({
      url: networkUrls.polygon,
      chainId: 137,
      live: true,
      gasPrice: Number(ethers.parseUnits('180', 'gwei')),

      verify: {
        etherscan: {
          apiKey: process.env.POLYGONSCAN_VERIFY_API_KEY,
        },
      },
    }),
    arbitrum: networkConfig({
      url: networkUrls.arbitrum,
      chainId: 42161,
      live: true,
      // gasPrice: ethers.utils.parseUnits('110', 'gwei').toNumber(),

      verify: {
        etherscan: {
          apiKey: process.env.ARBISCAN_VERIFY_API_KEY,
        },
      },
    }),
    base: networkConfig({
      url: networkUrls.base,
      chainId: 8453,
      live: true,
      // gasPrice: ethers.utils.parseUnits('110', 'gwei').toNumber(),

      verify: {
        etherscan: {
          apiKey: process.env.BASESCAN_VERIFY_API_KEY,
        },
      },
    }),
    mantle: networkConfig({
      url: networkUrls.mantle,
      chainId: 5000,
      live: true,
      gas: 9_000_000,
      gasPrice: Number(ethers.parseUnits('0.05', 'gwei')),

      // companionNetworks: {
      //   l1: 'mainnet',
      // },

      verify: {
        etherscan: {
          apiKey: process.env.MANTLE_VERIFY_API_KEY,
        },
      },
    }),

    // Test Networks
    sepolia: networkConfig({
      url: networkUrls.sepolia,
      chainId: 11155111,
      live: true,

      verify: {
        etherscan: {
          apiKey: process.env.ETHERSCAN_VERIFY_API_KEY,
        },
      },
    }),
    goerli: networkConfig({
      url: networkUrls.goerli,
      chainId: 5,
      live: true,
      gasPrice: Number(parseUnits('200', 'gwei')),

      verify: {
        etherscan: {
          apiKey: process.env.ETHERSCAN_VERIFY_API_KEY,
        },
      },
    }),
    mumbai: networkConfig({
      url: networkUrls.mumbai,
      gasPrice: 2100000000, // @lazycoder - deserves another Sherlock badge
      chainId: 80001,
      live: true,

      verify: {
        etherscan: {
          apiKey: process.env.POLYGONSCAN_VERIFY_API_KEY,
        },
      },
    }),
    'mantle-testnet': networkConfig({
      url: networkUrls['mantle-testnet'],
      gas: 5_000_000,
      chainId: 5001,
      live: true,

      // companionNetworks: {
      //   l1: 'goerli',
      // },

      verify: {
        etherscan: {
          apiKey: process.env.MANTLE_VERIFY_API_EY,
        },
      },
    }),
    tenderly: networkConfig({
      url: networkUrls.tenderly,
    }),
  },

  mocha: {
    timeout: 60000,
  },
}

const DEBUG = false

const debug = (text: string): void => {
  if (DEBUG) {
    console.log(text)
  }
}

task('wallet', 'Create a wallet (pk) link', async (_, { ethers }) => {
  const randomWallet = ethers.Wallet.createRandom()
  console.log(`🔐 WALLET Generated as ${randomWallet.address}`)
})

task('fundedwallet', 'Create a wallet (pk) link and fund it with deployer?')
  .addOptionalParam(
    'amount',
    'Amount of ETH to send to wallet after generating'
  )
  .addOptionalParam('url', 'URL to add pk to')
  .setAction(async (taskArgs, { ethers, getNamedSigner }) => {
    const randomWallet = ethers.Wallet.createRandom()
    console.log(`🔐 WALLET Generated as ${randomWallet.address}`)
    const url: string = taskArgs.url ? taskArgs.url : 'http://localhost:3000'

    const amount: string = taskArgs.amount ? taskArgs.amount : '0.01'
    const tx = {
      to: randomWallet.address,
      value: parseEther(amount),
    }

    // SEND USING LOCAL DEPLOYER MNEMONIC IF THERE IS ONE
    // IF NOT SEND USING LOCAL HARDHAT NODE:
    const localDeployerMnemonic = getMnemonic()
    if (localDeployerMnemonic) {
      let deployerWallet = ethers.Wallet.fromPhrase(localDeployerMnemonic)
      deployerWallet = deployerWallet.connect(ethers.provider)
      console.log(
        `💵 Sending ${amount} ETH to ${randomWallet.address} using deployer account`
      )
      const sendResult = await deployerWallet.sendTransaction(tx)

      console.log()
      console.log(`${url}/pk#${randomWallet.privateKey}`)
      console.log()

      return sendResult
    } else {
      console.log(
        `💵 Sending ${amount} ETH to ${randomWallet.address} using local node`
      )
      console.log()
      console.log(`${url}/pk#${randomWallet.privateKey}`)
      console.log()

      return await send(await getNamedSigner('deployer'), tx)
    }
  })

task(
  'mineContractAddress',
  'Looks for a deployer account that will give leading zeros'
)
  .addOptionalParam('searchFor', 'String to search for')
  .addOptionalParam('startsWith', 'String to search for')
  .setAction(async (taskArgs, { ethers }) => {
    if (!taskArgs.searchFor && !taskArgs.startsWith) {
      console.error(chalk.red('No arguments set.'))
      return
    }

    let wallet = ethers.Wallet.createRandom()
    let contractAddress = ''
    let attempt = 0
    let shouldRetry = true
    while (shouldRetry) {
      if (attempt > 0) {
        process.stdout.clearLine(0)
        process.stdout.cursorTo(0)
        wallet = ethers.Wallet.createRandom()
      }
      attempt++
      process.stdout.write(`Mining attempt ${attempt}`)

      contractAddress = ethers.getCreateAddress({
        from: wallet.address,
        nonce: 0,
      })

      if (taskArgs.searchFor) {
        shouldRetry = contractAddress.indexOf(taskArgs.searchFor) != 0
      } else if (taskArgs.startsWith) {
        shouldRetry =
          !contractAddress
            .substr(2)
            .startsWith(taskArgs.startsWith.toLowerCase()) &&
          !contractAddress
            .substr(2)
            .startsWith(taskArgs.startsWith.toUpperCase())
      }
    }
    process.stdout.write('\n')

    if (DEBUG) {
      console.log('mnemonic', wallet.mnemonic!.phrase)
      console.log('fullPath', wallet.path)
      console.log('privateKey', wallet.privateKey)
    }

    console.log(
      `⛏  Account Mined as ${wallet.address} and set as mnemonic in packages/hardhat`
    )
    console.log(
      `📜 This will create the first contract: ${chalk.magenta(
        contractAddress
      )}`
    )
    console.log(
      "💬 Use 'yarn run account' to get more information about the deployment account."
    )

    fs.writeFileSync(
      `./${wallet.address}_produces${contractAddress}.secret`,
      wallet.mnemonic!.phrase
    )
    fs.writeFileSync('./mnemonic.secret', wallet.mnemonic!.phrase)
  })

task(
  'account',
  'Get balance information for the deployment account.',
  async (_, { ethers, config }) => {
    try {
      const mnemonic = getMnemonic()
      const wallet = ethers.Wallet.fromPhrase(mnemonic)

      if (DEBUG) {
        console.log('mnemonic', wallet.mnemonic!.phrase)
        console.log('fullPath', wallet.path)
        console.log('privateKey', wallet.privateKey)
      }

      const qrcode = require('qrcode-terminal')
      qrcode.generate(wallet.address)
      console.log(
        `‍📬 Deployer Account is ${wallet.address} - ${wallet.privateKey}`
      )
      for (const networkName in config.networks) {
        const network = config.networks[networkName]
        if (!('url' in network)) continue
        try {
          const provider = new ethers.JsonRpcProvider(network.url)
          const balance = await provider.getBalance(wallet.address)
          console.log(` -- ${chalk.bold(networkName)} -- -- -- 📡 `)
          console.log(`  balance: ${ethers.formatEther(balance)}`)
          console.log(
            `  nonce: ${await provider.getTransactionCount(wallet.address)}`
          )
          console.log()
        } catch (e) {
          if (DEBUG) {
            console.log(e)
          }
        }
      }
    } catch (err) {
      console.log(`--- Looks like there is no mnemonic file created yet.`)
      console.log(
        `--- Please run ${chalk.greenBright('yarn generate')} to create one`
      )
    }
  }
)

/**
 * Get a checksumed address.
 * @param ethers {HardhatEthersHelpers} Ethers object from Hardhat.
 * @param addr {string | number} The address string to be checksumed or an index in the account's mnemonic.
 * @return Promise<string> The checksumed address
 */
async function findFirstAddr(
  ethers: HardhatEthersHelpers,
  addr: string | number
): Promise<string> {
  if (typeof addr === 'string' && isAddress(addr)) {
    return getAddress(addr)
  } else if (typeof addr === 'number') {
    const signers = await ethers.getSigners()
    if (signers[addr] !== undefined) {
      const address = await signers[addr].getAddress()
      return getAddress(address)
    }
  }
  throw new Error(`Could not normalize address: ${addr}`)
}

task('accounts', 'Prints the list of accounts', async (_, { ethers }) => {
  const accounts = await ethers.getSigners()
  accounts.forEach((account) => console.log(account))
})

task('blockNumber', 'Prints the block number', async (_, { ethers }) => {
  const blockNumber = await ethers.provider.getBlockNumber()
  console.log(blockNumber)
})

task('balance', "Prints an account's balance")
  .addPositionalParam(
    'account',
    "The account's address or index in the mnemonic"
  )
  .setAction(async (taskArgs, { ethers }) => {
    const balance = await ethers.provider.getBalance(
      await findFirstAddr(ethers, taskArgs.account)
    )
    console.log(formatUnits(balance, 'ether'), 'ETH')
  })

async function send(
  signer: Signer,
  txparams: TransactionRequest
): Promise<TransactionReceipt | null> {
  const response = await signer.sendTransaction(txparams)
  debug(`transactionHash: ${response.hash}`)
  const waitBlocksForReceipt = 0 // 2

  return await response.wait(waitBlocksForReceipt)
}

task('send', 'Send ETH')
  .addParam('from', 'From address or account index')
  .addOptionalParam('to', 'To address or account index')
  .addOptionalParam('amount', 'Amount to send in ether')
  .addOptionalParam('data', 'Data included in transaction')
  .addOptionalParam('gasPrice', 'Price you are willing to pay in gwei')
  .addOptionalParam('gasLimit', 'Limit of how much gas to spend')
  .setAction(async (taskArgs, { network, ethers }) => {
    const from = await findFirstAddr(ethers, taskArgs.from)
    debug(`Normalized from address: ${from}`)
    const fromSigner = await ethers.provider.getSigner(from)

    let to
    if (taskArgs.to) {
      to = await findFirstAddr(ethers, taskArgs.to)
      debug(`Normalized to address: ${to}`)
    }

    const txRequest: TransactionRequest = {
      from: await fromSigner.getAddress(),
      to,
      value: parseUnits(taskArgs.amount ? taskArgs.amount : '0', 'ether'),
      nonce: await fromSigner.getNonce(),
      gasPrice: parseUnits(
        taskArgs.gasPrice ? taskArgs.gasPrice : '1.001',
        'gwei'
      ),
      gasLimit: taskArgs.gasLimit ? taskArgs.gasLimit : 24000,
      chainId: network.config.chainId,
    }

    if (taskArgs.data !== undefined) {
      txRequest.data = taskArgs.data
      debug(`Adding data to payload: ${txRequest.data}`)
    }
    // eslint-disable-next-line @typescript-eslint/no-base-to-string
    debug(formatUnits(txRequest.gasPrice!.toString(), 'gwei'))
    debug(JSON.stringify(txRequest, null, 2))

    return await send(fromSigner, txRequest)
  })
