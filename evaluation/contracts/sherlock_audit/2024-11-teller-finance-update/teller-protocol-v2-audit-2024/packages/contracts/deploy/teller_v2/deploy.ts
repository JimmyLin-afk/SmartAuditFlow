import { DeployFunction } from 'hardhat-deploy/dist/types'

const deployFn: DeployFunction = async (hre) => {
  const trustedForwarder = await hre.contracts.get('MetaForwarder')
  const v2Calculations = await hre.deployments.get('V2Calculations')

  const tellerV2 = await hre.deployProxy('TellerV2', {
    unsafeAllow: [
      'constructor',
      'state-variable-immutable',
      'external-library-linking',
    ],
    constructorArgs: [await trustedForwarder.getAddress()],
    initializer: false,
    libraries: {
      V2Calculations: v2Calculations.address,
    },
  })

  return true
}

// tags and deployment
deployFn.id = 'teller-v2:deploy'
deployFn.tags = ['teller-v2', 'teller-v2:deploy']
deployFn.dependencies = ['meta-forwarder:deploy', 'teller-v2:v2-calculations']
export default deployFn
