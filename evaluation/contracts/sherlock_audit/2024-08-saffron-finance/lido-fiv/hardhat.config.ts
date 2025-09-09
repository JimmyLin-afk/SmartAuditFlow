import 'dotenv/config'
import { HardhatUserConfig } from 'hardhat/config'
import { removeConsoleLog } from 'hardhat-preprocessor'
import '@nomicfoundation/hardhat-toolbox'
import '@nomicfoundation/hardhat-chai-matchers'
import 'hardhat-contract-sizer'
import 'solidity-coverage'
import 'solidity-docgen'
import "hardhat-gas-reporter";

const PRIVATE_KEY = process.env.PRIVATE_KEY

const INFURA_API_KEY = process.env.INFURA_API_KEY

const MAINNET_ETHERSCAN_API_KEY = process.env.MAINNET_ETHERSCAN_API_KEY ?? ''
const ARBITRUM_ETHERSCAN_API_KEY = process.env.ARBITRUM_ETHERSCAN_API_KEY ?? ''

const SOLIDITY_VERSION = '0.8.18'

const config: HardhatUserConfig = {
  solidity: {
    compilers: [
      {
        version: SOLIDITY_VERSION,
        settings: {
          optimizer: {
            enabled: true,
            runs: 1, // Optimize vaults/adapters for low deployment, they're not called often
          },
        },
      },
    ],
    overrides: {
      'contracts/VaultFactory.sol': {
        version: SOLIDITY_VERSION,
        settings: {
          optimizer: {
            enabled: true,
            runs: 999999, // Optimize factory for many calls at expense of larger deploy
          },
        },
      },
    },
  },
  networks: {
    localhost: {
      url: 'http://127.0.0.1:8545',
      chainId: 31337,
    },
    hardhat: {
      forking: {
        url: `https://mainnet.infura.io/v3/${INFURA_API_KEY}`,
        blockNumber: 18562954, //19342016,
      },
      accounts: {
        count: 1000,
      },
    },
    mainnet: {
      chainId: 1,
      url: `https://mainnet.infura.io/v3/${INFURA_API_KEY}`,
      accounts: [`0x${PRIVATE_KEY}`],
    },
    arbitrumOne: {
      chainId: 42161,
      url: `https://arbitrum-mainnet.infura.io/v3/${INFURA_API_KEY}`,
      accounts: [`0x${PRIVATE_KEY}`],
    },
  },
  preprocess: {
    // Remove console.log from deployed code
    eachLine: removeConsoleLog(
      (hre) =>
        (hre.network.name !== 'hardhat' && hre.network.name !== 'localhost') ||
        (( process.env.REPORT_GAS == "true" ) && !( process.env.FORCE_LOGS == "true" ))
    ),
  },
  etherscan: {
    apiKey: {
      mainnet: MAINNET_ETHERSCAN_API_KEY,
      arbitrumOne: ARBITRUM_ETHERSCAN_API_KEY,
    },
  },
  gasReporter: {
    enabled: ( process.env.REPORT_GAS == "true" ),
    coinmarketcap: process.env.COINMARKETCAP_API_KEY, // https://coinmarketcap.com/api/pricing/
    currency: 'USD',
    gasPrice: 26,
  },
}

export default { ...config, docgen: { exclude: ['echidna', 'mocks'] } }
