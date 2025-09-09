import { getEtherscanAPIConfig } from '@openzeppelin/hardhat-upgrades/dist/utils/etherscan-api'
import { HARDHAT_NETWORK_NAME } from 'hardhat/plugins'
import { HardhatRuntimeEnvironment } from 'hardhat/types'
import { LogConfig } from 'helpers/hre-extensions'

export const logTxLink = async (
  hre: HardhatRuntimeEnvironment,
  txHash: string,
  logConfig: LogConfig = { star: true, indent: 1 }
): Promise<void> => {
  if (hre.network.name === HARDHAT_NETWORK_NAME) return
  try {
    const etherscanConfig = await getEtherscanAPIConfig(hre)
    hre.log(`${etherscanConfig.url}/tx/${txHash}`, logConfig)
  } catch (e) {
    hre.log(`${txHash} - failed to get network explorer URL`, logConfig)
  }
}
