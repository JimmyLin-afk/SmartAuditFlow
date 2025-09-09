import { DeployFunction } from 'hardhat-deploy/dist/types'

const aavePoolAddressProvider: { [networkName: string]: string } = {
  mainnet: '0x2f39d218133AFaB8F2B819B1066c7E434Ad94E9e',
  goerli: '0xC911B590248d127aD18546B186cC6B324e99F02c',
  sepolia: '0x0496275d34753A48320CA58103d5220d394FF77F',
  polygon: '0xa97684ead0e402dC232d5A977953DF7ECBaB3CDb',
  arbitrum: '0xa97684ead0e402dC232d5A977953DF7ECBaB3CDb',
  base: '0xe20fCBdBfFC4Dd138cE8b2E6FBb6CB49777ad64D',
}

const networksWithAave: string[] = Object.keys(aavePoolAddressProvider)

const deployFn: DeployFunction = async (hre) => {
  const tellerV2 = await hre.contracts.get('TellerV2')
  const LenderCommitmentForwarderStaging = await hre.contracts.get(
    'LenderCommitmentForwarderStaging'
  )

  const networkName = hre.network.name

  const flashRolloverLoan = await hre.deployProxy('FlashRolloverLoan', {
    unsafeAllow: ['constructor', 'state-variable-immutable'],
    constructorArgs: [
      await tellerV2.getAddress(),
      await LenderCommitmentForwarderStaging.getAddress(),
      aavePoolAddressProvider[networkName],
    ],
  })

  return true
}

// tags and deployment
deployFn.id = 'lender-commitment-forwarder:extensions:flash-rollover:deploy'
deployFn.tags = [
  'lender-commitment-forwarder',
  'lender-commitment-forwarder:extensions',
  'lender-commitment-forwarder:extensions:deploy',
  'lender-commitment-forwarder:extensions:flash-rollover',
  'lender-commitment-forwarder:extensions:flash-rollover:deploy',
]
deployFn.dependencies = [
  'teller-v2:deploy',
  'lender-commitment-forwarder:deploy',
]

deployFn.skip = async (hre) => {
  return !hre.network.live || !networksWithAave.includes(hre.network.name)
}
export default deployFn
