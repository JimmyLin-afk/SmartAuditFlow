import { BigNumber as BN } from 'ethers'
import { deployments } from 'hardhat'

import {
  submittedBid,
  SubmittedBidMainReturn,
  SubmittedBidOptions,
  SubmittedBidValues,
} from './submittedBid'

export interface CancelledBidOptions extends SubmittedBidOptions {}

export interface CancelledBidReturn
  extends CancelledBidMainReturn,
    CancelledBidValues {}

export interface CancelledBidMainReturn extends SubmittedBidMainReturn {}

export interface CancelledBidValues extends SubmittedBidValues {
  bidId: BN
}

export const cancelledBid = deployments.createFixture<
  CancelledBidReturn,
  CancelledBidOptions
>(async (hre, options) => {
  const submittedBidReturn = await submittedBid(options as SubmittedBidOptions)
  const {
    tellerV2,
    bidId,
    borrower,
    lendingToken,
    amount,
    marketplaceId,
    receiver,
  } = submittedBidReturn

  // Common values used in this fixture
  const values: CancelledBidValues = {
    tellerV2,
    borrower,
    lendingToken,
    bidId,
    amount,
    marketplaceId,
    receiver,
  }

  // Execute the transaction
  const tx = await tellerV2.connect(borrower.signer).cancelBid(bidId)

  return {
    ...values,
    tx,
  }
}, 'cancelled-bid')
