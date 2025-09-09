import path from 'path'

import { DeployFunction } from 'hardhat-deploy/dist/types'
import { HardhatRuntimeEnvironment } from 'hardhat/types'
import moment from 'moment'

import { getNetworkName } from '~~/config'

type MigrationFunction = (hre: HardhatRuntimeEnvironment) => Promise<void>

export const migrate = (
  filename: string,
  deployFn: MigrationFunction
): DeployFunction => {
  const id = path.basename(filename)
  const deploy: DeployFunction = async (hre: HardhatRuntimeEnvironment) => {
    const { log } = hre

    const [, timestamp, description = ''] = id.match(/(\d+)__(.*)?\.ts/) ?? []

    log('')
    log('------------------------------------------------------------')
    log('')
    log(`Running migration`)
    log(`Description: ${description.replace(/_(.)/g, ' $1')}`, {
      indent: 1,
      star: true,
    })
    log(`Network: ${getNetworkName(hre.network)}`, { indent: 1, star: true })
    log(
      `Created at: ${moment.unix(parseInt(timestamp)).format('LL hh:mm:ss a')}`,
      { indent: 1, star: true }
    )
    log('')

    await deployFn(hre)

    log('')
    log('------------------------------------------------------------')
    log('')

    /* !!!!!!! WARNING !!!!!!! */
    // Migration scripts MUST return true to ensure it will only run once per network
    return true
  }

  deploy.id = id
  deploy.tags = ['migration']

  return deploy
}
