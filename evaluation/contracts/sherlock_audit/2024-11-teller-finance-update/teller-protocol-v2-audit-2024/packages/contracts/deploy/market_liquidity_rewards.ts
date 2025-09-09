import { DeployFunction } from 'hardhat-deploy/dist/types'

const deployFn: DeployFunction = async (hre) => {
  const tellerV2 = await hre.contracts.get('TellerV2')
  const marketRegistry = await hre.contracts.get('MarketRegistry')
  const collateralManager = await hre.contracts.get('CollateralManager')

  const marketLiquidityRewards = await hre.deployProxy(
    'MarketLiquidityRewards',
    {
      unsafeAllow: ['constructor', 'state-variable-immutable'],
      constructorArgs: [
        await tellerV2.getAddress(),
        await marketRegistry.getAddress(),
        await collateralManager.getAddress(),
      ],
    }
  )

  return true
}

// tags and deployment
deployFn.id = 'liquidity-rewards:deploy'
deployFn.tags = ['liquidity-rewards', 'liquidity-rewards:deploy']
deployFn.dependencies = [
  'teller-v2:deploy',
  'market-registry:deploy',
  'collateral:deploy',
]
export default deployFn
