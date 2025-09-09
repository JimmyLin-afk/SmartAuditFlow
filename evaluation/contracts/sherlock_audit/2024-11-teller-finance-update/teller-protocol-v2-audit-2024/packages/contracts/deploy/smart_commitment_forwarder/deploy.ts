import { DeployFunction } from 'hardhat-deploy/dist/types'

const deployFn: DeployFunction = async (hre) => {
  const tellerV2 = await hre.contracts.get('TellerV2')
  const marketRegistry = await hre.contracts.get('MarketRegistry')

  const smartCommitmentForwarder = await hre.deployProxy(
    'SmartCommitmentForwarder',
    {
      unsafeAllow: ['constructor', 'state-variable-immutable'],
      constructorArgs: [
        await tellerV2.getAddress(),
        await marketRegistry.getAddress(),
      ],
    }
  )

  return true
}

// tags and deployment
deployFn.id = 'smart-commitment-forwarder:deploy'
deployFn.tags = [
  'smart-commitment-forwarder',
  'smart-commitment-forwarder:deploy',
]
deployFn.dependencies = ['teller-v2:deploy', 'market-registry:deploy']

deployFn.skip = async (hre) => {
  return (
    !hre.network.live || !['localhost', 'polygon'].includes(hre.network.name)
  )
}
export default deployFn
