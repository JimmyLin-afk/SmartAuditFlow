import { config as dotEnvConfig } from "dotenv";
dotEnvConfig();

import { HardhatUserConfig } from "hardhat/config";
import "@nomicfoundation/hardhat-toolbox-viem";
import "@nomicfoundation/hardhat-chai-matchers";
import "@openzeppelin/hardhat-upgrades";
import "hardhat-contract-sizer";
import "hardhat-dependency-compiler";

const config: HardhatUserConfig = {
  solidity: {
    version: "0.8.26",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200
      }
    }
  },
  networks: {
    hardhat: {
      allowUnlimitedContractSize: true,
    },
    mainnet: {
      url: process.env.MAINNET_URL || '',
      accounts: process.env.MAINNET_PRIVATE_KEY ? [process.env.MAINNET_PRIVATE_KEY] : [],
    }, 
    sepolia: {
      url: process.env.SEPOLIA_URL || '',
      accounts: process.env.SEPOLIA_PRIVATE_KEY ? [process.env.SEPOLIA_PRIVATE_KEY] : [],
    },
    blast: {
      url: process.env.BLAST_URL || '',
      accounts: process.env.BLAST_PRIVATE_KEY ? [process.env.BLAST_PRIVATE_KEY] : [],
    }, 
    goerli: {
      url: process.env.GOERLI_URL || '',
      accounts: process.env.GOERLI_PRIVATE_KEY ? [process.env.GOERLI_PRIVATE_KEY] : [],
    },
    base: {
      url: process.env.BASE_URL || '',
      accounts: process.env.BASE_PRIVATE_KEY ? [process.env.BASE_PRIVATE_KEY] : [],
    },
    berachain: {
      url: process.env.BERACHAIN_URL || '',
      accounts: process.env.BERACHAIN_PRIVATE_KEY ? [process.env.BERACHAIN_PRIVATE_KEY] : [],
    }
  },
  gasReporter: {
    enabled: true,
    currency: "USD",
    outputFile: "gas-report.txt",
    noColors: true,
    coinmarketcap: process.env.COINMARKETCAP_API_KEY,
    token: "ETH",
    etherscan:  process.env.ETHERSCAN_API_KEY || '',
  },
  etherscan: {
    apiKey: process.env.ETHERSCAN_API_KEY || '',
    customChains: [
      {
        network: "blast",
        chainId: 81457,
        urls: {
          apiURL: "https://api.blastscan.io/api",
          browserURL: "https://blastscan.io"
        }
      },
      {
        network: "berachain",
        chainId: 80094,
        urls: {
          apiURL: "https://api.berascan.com/api",
          browserURL: "https://berascan.com"
        }
      }
    ]
  },
  contractSizer: {
    runOnCompile: true,
    
    only: [
      "Hash",
      "WasabiVault",
      "WasabiLongPool",
      "WasabiShortPool",
      "BlastLongPool",
      "BlastShortPool",
      "BlastVault",
      "BeraLongPool",
      "BeraShortPool",
      "BeraVault",
      "PerpUtils",
      "AddressProvider",
      "DebtController",
      "WasabiRouter",
      "StakingAccountFactory",
      "StakingAccount"
    ],
    except: ['/mock/']
  },
  dependencyCompiler: {
    paths: [
      "@berachain/pol-contracts/src/pol/BGT.sol",
      "@berachain/pol-contracts/src/pol/BGTStaker.sol",
      "@berachain/pol-contracts/src/pol/rewards/BeraChef.sol",
      "@berachain/pol-contracts/src/pol/rewards/RewardVaultFactory.sol",
      "@berachain/pol-contracts/src/pol/rewards/BlockRewardController.sol",
    ]
  }
};

export default config;
