import { DeployFunction } from 'hardhat-deploy/dist/types'

const deployFn: DeployFunction = async (hre) => {
  const escrowVault = await hre.deployProxy('EscrowVault', {
    initArgs: [],
  })

  return true
}

// tags and deployment
deployFn.id = 'escrow-vault:deploy'
deployFn.tags = ['escrow-vault', 'escrow-vault:deploy']
deployFn.dependencies = []
export default deployFn
