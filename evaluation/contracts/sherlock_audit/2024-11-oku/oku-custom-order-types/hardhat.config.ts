import { HardhatUserConfig } from "hardhat/config";
import "@nomicfoundation/hardhat-toolbox";
import "@nomicfoundation/hardhat-ignition-ethers";
import "@matterlabs/hardhat-zksync-verify";
import "@matterlabs/hardhat-zksync-solc";

import * as dotenv from "dotenv";

dotenv.config();
const zaddr =
  "0000000000000000000000000000000000000000000000000000000000000000";
const config: HardhatUserConfig = {
  defaultNetwork: "hardhat",
  networks: {
    hardhat: {
      forking: {
        url: process.env.MAINNET_URL ? process.env.MAINNET_URL : zaddr,
        blockNumber: 14546835,
      },
      mining: {
        auto: true,
      },
    },
    mainnet: {
      url: process.env.MAINNET_URL ? process.env.MAINNET_URL : zaddr,
      accounts: [
        process.env.MAINNET_PRIVATE_KEY
          ? process.env.MAINNET_PRIVATE_KEY
          : zaddr,
        process.env.PERSONAL_PRIVATE_KEY
          ? process.env.PERSONAL_PRIVATE_KEY
          : zaddr
      ],
      minGasPrice: 32000000000,
    },
    arbitrum: {
      url: process.env.ARB_URL ? process.env.ARB_URL : zaddr,
      accounts: [
        process.env.MAINNET_PRIVATE_KEY
          ? process.env.MAINNET_PRIVATE_KEY
          : zaddr
      ]
    },
    op: {
      url: process.env.OP_URL ? process.env.OP_URL : zaddr,
      accounts: [
        process.env.MAINNET_PRIVATE_KEY
          ? process.env.MAINNET_PRIVATE_KEY
          : zaddr,
        process.env.PERSONAL_PRIVATE_KEY
          ? process.env.PERSONAL_PRIVATE_KEY
          : zaddr
      ],
      minGasPrice: 32000000000,
      chainId: 10
    },
    polygon: {
      url: process.env.POLYGON_URL ? process.env.POLYGON_URL : zaddr,
      accounts: [
        process.env.MAINNET_PRIVATE_KEY
          ? process.env.MAINNET_PRIVATE_KEY
          : zaddr,
        process.env.PERSONAL_PRIVATE_KEY
          ? process.env.PERSONAL_PRIVATE_KEY
          : zaddr
      ],
    },
    goerli: {
      url: process.env.GOERLI_URL ? process.env.GOERLI_URL : zaddr,
      accounts: [
        process.env.MAINNET_PRIVATE_KEY
          ? process.env.MAINNET_PRIVATE_KEY
          : zaddr
      ],
      minGasPrice: 32000000000,
      chainId: 5
    },
    base: {
      url: process.env.BASE_URL ? process.env.BASE_URL : zaddr,
      accounts: [
        process.env.MAINNET_PRIVATE_KEY
          ? process.env.MAINNET_PRIVATE_KEY
          : zaddr
      ],
      chainId: 8453
    },
    bsc: {
      url: process.env.BSC_URL ? process.env.BSC_URL : zaddr,
      accounts: [
        process.env.MAINNET_PRIVATE_KEY
          ? process.env.MAINNET_PRIVATE_KEY
          : zaddr
      ],
      chainId: 56
    },
    avax: {
      url: process.env.AVAX_URL ? process.env.AVAX_URL : zaddr,
      accounts: [
        process.env.MAINNET_PRIVATE_KEY
          ? process.env.MAINNET_PRIVATE_KEY
          : zaddr
      ],
      chainId: 43114,
    },
    zksolc: {//todo https://docs.zksync.io/build/tooling/hardhat/hardhat-zksync-solc.html#configuration
      url: process.env.ZKSYNC_URL ? process.env.ZKSYNC_URL : zaddr,
      accounts: [
        process.env.MAINNET_PRIVATE_KEY
          ? process.env.MAINNET_PRIVATE_KEY
          : zaddr
      ],
      chainId: 324,
      verifyURL: "https://zksync2-mainnet-explorer.zksync.io/contract_verification"
    },
    scroll: {
      url: process.env.SCROLL_URL ? process.env.SCROLL_URL : zaddr,
      accounts: [
        process.env.MAINNET_PRIVATE_KEY
          ? process.env.MAINNET_PRIVATE_KEY
          : zaddr
      ],
      chainId: 534352
    },
    filecoin: {
      url: process.env.FILECOIN_URL ? process.env.FILECOIN_URL : zaddr,
      accounts: [
        process.env.MAINNET_PRIVATE_KEY
          ? process.env.MAINNET_PRIVATE_KEY
          : zaddr
      ],
      chainId: 314
    },
    moonbeam: {
      url: process.env.MOONBEAM_URL ? process.env.MOONBEAM_URL : zaddr,
      accounts: [
        process.env.MAINNET_PRIVATE_KEY
          ? process.env.MAINNET_PRIVATE_KEY
          : zaddr
      ],
      chainId: 1284
    },
    polygonzkevm: {
      url: process.env.POLYGONZKEVM_URL ? process.env.POLYGONZKEVM_URL : zaddr,
      accounts: [
        process.env.MAINNET_PRIVATE_KEY
          ? process.env.MAINNET_PRIVATE_KEY
          : zaddr
      ],
      chainId: 1101
    },
    blast: {
      url: process.env.BLAST_URL ? process.env.BLAST_URL : zaddr,
      accounts: [
        process.env.MAINNET_PRIVATE_KEY
          ? process.env.MAINNET_PRIVATE_KEY
          : zaddr
      ],
      chainId: 81457
    },
    rootstock: {
      url: process.env.ROOTSTOCK_URL ? process.env.ROOTSTOCK_URL : zaddr,
      accounts: [
        process.env.MAINNET_PRIVATE_KEY
          ? process.env.MAINNET_PRIVATE_KEY
          : zaddr
      ],
      chainId: 30
    },
    mantapacific: {
      url: process.env.MANTAPACIFIC_URL ? process.env.MANTAPACIFIC_URL : zaddr,
      accounts: [
        process.env.MAINNET_PRIVATE_KEY
          ? process.env.MAINNET_PRIVATE_KEY
          : zaddr
      ],
      chainId: 169
    },
    boba: {
      url: process.env.BOBA_URL ? process.env.BOBA_URL : zaddr,
      accounts: [
        process.env.MAINNET_PRIVATE_KEY
          ? process.env.MAINNET_PRIVATE_KEY
          : zaddr
      ],
      chainId: 288
    },
    linea: {
      url: process.env.LINEA_URL ? process.env.LINEA_URL : zaddr,
      accounts: [
        process.env.MAINNET_PRIVATE_KEY
          ? process.env.MAINNET_PRIVATE_KEY
          : zaddr
      ]
    }
  },
  //todo boba


  etherscan: {
    apiKey: {
      mainnet: process.env.API_KEY!,
      goerli: process.env.API_KEY!,
      polygon: process.env.ETHERSCAN_POLYGON_KEY!,
      optimisticEthereum: process.env.OP_KEY!,
      arbitrumOne: process.env.ARB_API_KEY!,
      base: process.env.BASE_API_KEY!,
      bsc: process.env.BSC_API_KEY!,
      routescan: "routescan",
      scroll: process.env.SCROLL_API_KEY!,
      filecoin: "filecoin",
      moonbeam: process.env.MOONBEAM_API_KEY!,
      polygonZkEVM: process.env.POLYGONZKEVM_API_KEY!,
      blast: process.env.BLAST_API_KEY!,
      boba: "boba",
      rootstock: "rootstock",
      mantapacific: "mantapacific",
      linea_mainnet: process.env.LINEA_API_KEY!
    },
    customChains: [
      {
        network: "base",
        chainId: 8453,
        urls: {
          apiURL: "https://api.basescan.org/api",
          browserURL: "https://basescan.org/api"
        }
      },
      {
        network: "routescan",
        chainId: 43114,
        urls: {
          apiURL: "https://api.routescan.io/v2/network/mainnet/evm/43114/etherscan",
          browserURL: "https://routescan.io"
        }
      },
      {
        network: "boba",
        chainId: 288,
        urls: {
          apiURL: "https://api.routescan.io/v2/network/mainnet/evm/288/etherscan",
          browserURL: "https://bobascan.com"
        }
      },
      {
        network: "blast",
        chainId: 81457,
        urls: {
          apiURL: "https://api.blastscan.io/api",
          browserURL: "https://blastscan.io/"
        }
      },
      {
        network: "rootstock",
        chainId: 30,
        urls: {
          apiURL: "https://rootstock.blockscout.com/api/",
          browserURL: "https://rootstock.blockscout.com/"
        }
      },
      {
        network: "mantapacific",
        chainId: 169,
        urls: {
          apiURL: "https://manta-pacific.calderaexplorer.xyz/api/",
          browserURL: "https://manta-pacific.calderaexplorer.xyz/"
        }
      },
      {
        network: "scroll",
        chainId: 534352,
        urls: {
          apiURL: "https://api.scrollscan.com/api",
          browserURL: "https://scrollscan.com/"
        }
      },
      {
        network: "filecoin",
        chainId: 314,
        urls: {
          apiURL: "https://api.filscan.io/api",
          browserURL: "https://filscan.io/"
        }
      },
      {
        network: "linea_mainnet",
        chainId: 59144,
        urls: {
          apiURL: "https://api.lineascan.build/api",
          browserURL: "https://lineascan.build/"
        }
      },
    ]
  },
  solidity: {
    version: "0.8.24",
    settings: {
      viaIR: true
    }
  },
  sourcify: {
    enabled: true
  }
};

export default config;
