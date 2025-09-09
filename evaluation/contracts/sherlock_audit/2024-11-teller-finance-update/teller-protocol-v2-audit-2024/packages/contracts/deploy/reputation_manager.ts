import { DeployFunction } from 'hardhat-deploy/dist/types'

const deployFn: DeployFunction = async (hre) => {
  const tellerV2 = await hre.contracts.get('TellerV2')

  const reputationManager = await hre.deployProxy('ReputationManager', {
    initArgs: [await tellerV2.getAddress()],
  })

  return true
}

// tags and deployment
deployFn.id = 'reputation-manager:deploy'
deployFn.tags = ['reputation-manager', 'reputation-manager:deploy']
deployFn.dependencies = ['teller-v2:deploy']
export default deployFn
