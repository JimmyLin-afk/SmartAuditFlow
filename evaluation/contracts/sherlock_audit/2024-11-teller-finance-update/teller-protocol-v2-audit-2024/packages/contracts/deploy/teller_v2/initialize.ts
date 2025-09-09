import { DeployFunction } from 'hardhat-deploy/dist/types'
import { logTxLink } from 'helpers/logTxLink'
import { TellerV2 } from 'types/typechain'

const deployFn: DeployFunction = async (hre) => {
  hre.log('----------')
  hre.log('')
  hre.log('TellerV2: Initializing...', { nl: false })

  const protocolFee = 5 // 0.05%

  const marketRegistry = await hre.contracts.get('MarketRegistry')
  const reputationManager = await hre.contracts.get('ReputationManager')
  const lenderCommitmentForwarder = await hre.contracts.get(
    'LenderCommitmentForwarder'
  )
  const collateralManager = await hre.contracts.get('CollateralManager')
  const lenderManager = await hre.contracts.get('LenderManager')
  const escrowVault = await hre.contracts.get('EscrowVault')
  const protocolPausingManager = await hre.contracts.get('ProtocolPausingManager')

  const tellerV2 = await hre.contracts.get<TellerV2>('TellerV2')
  const tx = await tellerV2.initialize(
    protocolFee,
    marketRegistry,
    reputationManager,
    lenderCommitmentForwarder,
    collateralManager,
    lenderManager,
    escrowVault,
    protocolPausingManager
  )
  await tx.wait(1) // wait one block

  hre.log('done.')
  await logTxLink(hre, tx.hash)
  hre.log('')
  hre.log('----------')

  return true
}

// tags and deployment
deployFn.id = 'teller-v2:init'
deployFn.tags = ['teller-v2', 'teller-v2:init']
deployFn.dependencies = [
  'teller-v2:deploy',
  'market-registry:deploy',
  'reputation-manager:deploy',
  'lender-commitment-forwarder:deploy',
  'collateral:manager:deploy',
  'lender-manager:deploy',
  'escrow-vault:deploy',
]
export default deployFn
