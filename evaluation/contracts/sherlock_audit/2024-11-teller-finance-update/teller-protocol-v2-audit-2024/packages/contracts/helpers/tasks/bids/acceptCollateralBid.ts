import { task } from 'hardhat/config'
import { HardhatRuntimeEnvironment } from 'hardhat/types'
import { TellerV2 } from 'types/typechain'

interface AcceptBidArgs {
  bidId: number
}

export const acceptCollateralBid = async (
  args: AcceptBidArgs,
  hre: HardhatRuntimeEnvironment
): Promise<void> => {
  const { contracts, log, tokens, toBN } = hre
  const { bidId } = args

  // Load contracts
  const tellerV2 = await contracts.get<TellerV2>('TellerV2')
  const usdc = await tokens.get('USDC')

  // Approve funds to be lent
  await usdc.approve(tellerV2, toBN(100, 6))

  // Accept bid
  const lentAmount = await tellerV2.lenderAcceptBid(bidId)

  log(`Amount lent to borrower of ${bidId}`, { indent: 4 })
}

task('accept-collateral-bid', 'accepts a bid')
  .addParam('bidId', 'The id of the bid to accept')
  .setAction(acceptCollateralBid)
