import { BigNumberish } from 'ethers'
import { task } from 'hardhat/config'
import { HardhatRuntimeEnvironment } from 'hardhat/types'
import { TellerV2 } from 'types/typechain'

interface RepayLoanArgs {
  bidId: BigNumberish
}

export const repayLoan = async (
  args: RepayLoanArgs,
  hre: HardhatRuntimeEnvironment
): Promise<void> => {
  const { contracts, log, tokens, toBN } = hre
  const { bidId } = args

  // Load contracts
  const tellerV2 = await contracts.get<TellerV2>('TellerV2')
  const usdc = await tokens.get('USDC')

  // Get total amount due to repay
  const block = await hre.ethers.provider.getBlock('latest')
  const amountDue = await tellerV2.calculateAmountOwed(bidId, block!.timestamp)
  log(`${amountDue}`)
  // Approve funds to be repaid
  await usdc.approve(tellerV2, toBN(110, 6))

  // Repay loan
  await tellerV2.repayLoanFull(bidId)

  log(`Loan ${bidId} repaid`, { indent: 4 })
}

task('repay-loan', 'repays a loan in full')
  .addParam('bidId', 'The id of the bid to repay')
  .setAction(repayLoan)
