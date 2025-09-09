import { task } from 'hardhat/config'
import { HardhatRuntimeEnvironment } from 'hardhat/types'
import { TellerV2 } from 'types/typechain'

export interface activeLoansReturn {
  activeLoans: Array<bigint>
  activeLoanLenders: string[]
}

export const getActiveLoans = async (
  args: null,
  hre: HardhatRuntimeEnvironment
): Promise<activeLoansReturn> => {
  const tellerV2 = await hre.contracts.get<TellerV2>('TellerV2')

  // Get events for current active loans on Teller V2
  const events = await tellerV2.queryFilter(tellerV2.filters.AcceptedBid())

  // Get list of active loans & lenders
  const activeLoans = []
  const activeLoanLenders = []

  for (const event of events) {
    const bidId = event.args.bidId
    const bid = await tellerV2.bids(bidId)
    // If Bid state is "Accepted"
    if (bid.state == 3n) {
      // Get active loan lender
      const lender = await tellerV2.getLoanLender(bidId)
      activeLoanLenders.push(lender)
      activeLoans.push(event.args.bidId)
    }
  }

  return {
    activeLoans: activeLoans,
    activeLoanLenders: activeLoanLenders,
  }
}

task(
  'get-active-loans',
  'Gets a list of current active loans & associated lenders'
).setAction(getActiveLoans)
