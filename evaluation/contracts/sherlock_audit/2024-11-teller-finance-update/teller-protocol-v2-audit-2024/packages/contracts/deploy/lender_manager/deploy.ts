import { DeployFunction } from 'hardhat-deploy/dist/types'

const deployFn: DeployFunction = async (hre) => {
  const marketRegistry = await hre.contracts.get('MarketRegistry')

  const lenderManager = await hre.deployProxy('LenderManager', {
    constructorArgs: [await marketRegistry.getAddress()],
    unsafeAllow: ['constructor', 'state-variable-immutable'],
  })

  return true
}

// tags and deployment
deployFn.id = 'lender-manager:deploy'
deployFn.tags = ['lender-manager', 'lender-manager:deploy']
deployFn.dependencies = ['market-registry:deploy']
export default deployFn
