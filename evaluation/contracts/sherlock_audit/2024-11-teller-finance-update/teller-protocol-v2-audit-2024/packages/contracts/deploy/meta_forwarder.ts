import { DeployFunction } from 'hardhat-deploy/dist/types'

const deployFn: DeployFunction = async (hre) => {
  const trustedForwarder = await hre.deployProxy('MetaForwarder', {})

  return true
}

// tags and deployment
deployFn.id = 'meta-forwarder:deploy'
deployFn.tags = ['meta-forwarder', 'meta-forwarder:deploy']
deployFn.dependencies = []
export default deployFn
