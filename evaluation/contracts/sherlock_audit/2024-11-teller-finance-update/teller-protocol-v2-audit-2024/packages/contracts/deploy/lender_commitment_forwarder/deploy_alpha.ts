import { DeployFunction } from 'hardhat-deploy/dist/types'

const deployFn: DeployFunction = async (hre) => {
  const tellerV2 = await hre.contracts.get('TellerV2')
  const marketRegistry = await hre.contracts.get('MarketRegistry')

  let uniswapFactoryAddress: string
  switch (hre.network.name) {
    case 'mainnet':
    case 'goerli':
    case 'arbitrum':
    case 'optimism':
    case 'polygon':
    case 'localhost':
      uniswapFactoryAddress = '0x1F98431c8aD98523631AE4a59f267346ea31F984'
      break
    case 'base':
      uniswapFactoryAddress = '0x33128a8fC17869897dcE68Ed026d694621f6FDfD'
      break
    case 'sepolia':
      uniswapFactoryAddress = '0x0227628f3F023bb0B980b67D528571c95c6DaC1c'
      break
    default:
      throw new Error('No swap factory address found for this network')
  }

  const lenderCommitmentForwarderAlpha = await hre.deployProxy(
    'LenderCommitmentForwarderAlpha',
    {
      unsafeAllow: ['constructor', 'state-variable-immutable'],
      constructorArgs: [
        await tellerV2.getAddress(),
        await marketRegistry.getAddress(),
        uniswapFactoryAddress,
      ],
    }
  )

  return true
}

// tags and deployment
deployFn.id = 'lender-commitment-forwarder:alpha:deploy'
deployFn.tags = [
  'lender-commitment-forwarder',
  'lender-commitment-forwarder:alpha',
  'lender-commitment-forwarder:alpha:deploy',
]
deployFn.dependencies = ['teller-v2:deploy', 'market-registry:deploy']
export default deployFn
