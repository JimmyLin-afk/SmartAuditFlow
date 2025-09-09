import { DeployFunction } from 'hardhat-deploy/dist/types'
import { deploy } from 'helpers/deploy-helpers'

const deployFn: DeployFunction = async (hre) => {
  const registry = await deploy({
    contract: 'TellerASRegistry',
    skipIfAlreadyDeployed: true,
    hre,
  })
  const verifier = await deploy({
    contract: 'TellerASEIP712Verifier',
    skipIfAlreadyDeployed: true,
    hre,
  })
  const tellerAS = await deploy({
    contract: 'TellerAS',
    args: [await registry.getAddress(), await verifier.getAddress()],
    skipIfAlreadyDeployed: true,
    hre,
  })

  const marketRegistry = await hre.deployProxy('MarketRegistry', {
    initArgs: [await tellerAS.getAddress()],
  })

  return true
}

// tags and deployment
deployFn.id = 'market-registry:deploy'
deployFn.tags = ['market-registry', 'market-registry:deploy']
deployFn.dependencies = []
export default deployFn
