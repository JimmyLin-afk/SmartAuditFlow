import { HardhatRuntimeEnvironment, Network } from 'hardhat/types'
import { AllNetworkTokens, NetworkTokens, Tokens } from 'helpers/types'

import { tokens } from './tokens'

/**
 * Checks if the network is Ethereum mainnet or one of its testnets
 * @param network {Network} Hardhat Network object
 * @param strict {boolean} Weather the check for mainnet should be restricted to exact match
 * @return boolean Boolean if the current network is Ethereum
 */
export const isEthereumNetwork = (network: Network, strict = false): boolean =>
  strict
    ? getNetworkName(network) === 'mainnet'
    : ['mainnet', 'goerli'].some((n) => n === getNetworkName(network))

/**
 * Gets the current network name. If there is a `HARDHAT_DEPLOY_FORK` environment variable set that is returned instead.
 * @param network {Network} Hardhat network object
 * @return string The current network name
 */
export const getNetworkName = (network: Network): string =>
  ['hardhat', 'localhost'].includes(network.name)
    ? process.env.HARDHAT_DEPLOY_FORK ?? network.name
    : network.name

/**
 * Gets the object of tokens specified by the config file including an `all` field which list every token by its symbol.
 * @param hre {HardhatRuntimeEnvironment} Hardhat Network object
 * @return AllNetworkTokens Object of all tokens for the specified network
 */
export const getTokens = async (
  hre: HardhatRuntimeEnvironment
): Promise<AllNetworkTokens> => {
  const networkTokens =
    hre.network.name === 'hardhat'
      ? await deployHardhatTokens(hre)
      : tokens[getNetworkName(hre.network)]
  const all = Object.keys(networkTokens).reduce<Tokens>(
    (map, type) => ({
      ...map,
      ...networkTokens[type],
    }),
    {}
  )
  return {
    ...networkTokens,
    all,
  }
}

let hardhatTokens: NetworkTokens | undefined
const deployHardhatTokens = async (
  hre: HardhatRuntimeEnvironment
): Promise<NetworkTokens> => {
  if (hardhatTokens) return hardhatTokens

  const { deployments, getNamedAccounts } = hre
  const { deploy } = deployments
  const { funder } = await getNamedAccounts()

  const deployToken = async (name: string, symbol: string): Promise<string> => {
    const { address } = await deploy('ERC20PresetMinterPauser', {
      from: funder,
      args: [name, symbol],
    })
    return address
  }

  hardhatTokens = {
    erc20: {
      DAI: await deployToken('Dai', 'DAI'),
      USDC: await deployToken('USD Coin', 'USDC'),
      WETH: await deployToken('Wrapped ETH', 'WETH'),
    },
  }

  return hardhatTokens
}
