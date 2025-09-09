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
  hre.log('----------')
  hre.log('')
  hre.log('FlashRolloverLoan G3: Proposing upgrade...')

  const networkName = hre.network.name

  const flashRolloverLoan = await hre.contracts.get('FlashRolloverLoan')
  const tellerV2 = await hre.contracts.get('TellerV2')
  const LenderCommitmentForwarderStaging = await hre.contracts.get(
    'LenderCommitmentForwarderStaging'
  )

  await hre.upgrades.proposeBatchTimelock({
    title: 'Flash Rollover Loan Extension Upgrade',
    description: ` 
# FlashRolloverLoan G3 (Extensions Upgrade)

* Upgrades the flash rollover loan contract to support merkle proofs.
`,
    _steps: [
      {
        proxy: flashRolloverLoan,
        implFactory: await hre.ethers.getContractFactory('FlashRolloverLoan'),

        opts: {
          unsafeAllow: ['constructor', 'state-variable-immutable'],
          constructorArgs: [
            await tellerV2.getAddress(),
            await LenderCommitmentForwarderStaging.getAddress(),
            aavePoolAddressProvider[networkName],
          ],
        },
      },
    ],
  })

  hre.log('done.')
  hre.log('')
  hre.log('----------')

  return true
}

// tags and deployment
deployFn.id = 'lender-commitment-forwarder:extensions:flash-rollover:g3-upgrade'
deployFn.tags = [
  'proposal',
  'upgrade',
  'lender-commitment-forwarder',
  'lender-commitment-forwarder:extensions',
  'lender-commitment-forwarder:extensions:flash-rollover',
  'lender-commitment-forwarder:extensions:flash-rollover:g3-upgrade',
]
deployFn.dependencies = [
  'teller-v2:deploy',
  'lender-commitment-forwarder:staging:deploy',
  'lender-commitment-forwarder:extensions:flash-rollover:deploy',
]
deployFn.skip = async (hre) => {
  return !(
    hre.network.live &&
    ['mainnet', 'polygon', 'arbitrum', 'base', 'goerli', 'sepolia'].includes(
      hre.network.name
    )
  )
}
export default deployFn
