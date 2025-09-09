import { DeployFunction } from 'hardhat-deploy/dist/types'

const deployFn: DeployFunction = async (hre) => {
  const { deployer } = await hre.getNamedAccounts()
  const v2Calculations = await hre.deployments.deploy('V2Calculations', {
    from: deployer,
  })
}

// tags and deployment
deployFn.id = 'teller-v2:v2-calculations'
deployFn.tags = ['teller-v2', 'teller-v2:v2-calculations']
deployFn.dependencies = ['']
export default deployFn
