import { task } from 'hardhat/config'
import { HardhatRuntimeEnvironment } from 'hardhat/types'
import { CollateralManager } from 'types/typechain'

interface ClaimCollateralArgs {
  bidId: number
}

export const withdrawCollateral = async (
  args: ClaimCollateralArgs,
  hre: HardhatRuntimeEnvironment
): Promise<void> => {
  const { contracts, log } = hre
  const { bidId } = args

  // Load collateral manager contract
  const collateralManager = await contracts.get<CollateralManager>(
    'CollateralManager'
  )

  // Claim collateral
  await collateralManager.withdraw(bidId)

  log(`Collateral claimed for defaulted loan id ${bidId}`)
}

task('withdraw-collateral', 'withdraws collateral for a defaulted loan')
  .addParam('bidId', 'The id of the bid to claim collateral for')
  .setAction(withdrawCollateral)
