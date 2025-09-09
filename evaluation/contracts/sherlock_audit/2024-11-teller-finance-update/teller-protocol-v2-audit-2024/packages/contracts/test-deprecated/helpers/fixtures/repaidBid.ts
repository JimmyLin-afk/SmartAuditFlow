import hre, { deployments } from 'hardhat'
import { getFunds } from 'helpers/get-funds'
import moment from 'moment'

import {
  acceptedBid,
  AcceptedBidMainReturn,
  AcceptedBidOptions,
  AcceptedBidValues,
} from './acceptedBid'

export interface RepaidBidOptions {
  fullRepay?: boolean
}

export interface RepaidBidReturn extends RepaidBidMainReturn, RepaidBidValues {}

export interface RepaidBidMainReturn extends AcceptedBidMainReturn {}

export interface RepaidBidValues extends AcceptedBidValues {}

export const repaidBid = deployments.createFixture<
  RepaidBidReturn,
  RepaidBidOptions
>(async (hre, options) => {
  // Create and accept a bid
  const acceptedBidReturn = await acceptedBid(options as AcceptedBidOptions)
  const {
    tellerV2,
    bidId,
    borrower,
    lender,
    lendingToken,
    amount,
    marketplaceId,
    receiver,
  } = acceptedBidReturn

  // Common values used in this fixture
  const values: RepaidBidValues = {
    tellerV2,
    borrower,
    lender,
    lendingToken,
    bidId,
    amount,
    marketplaceId,
    receiver,
  }

  const block = await hre.ethers.provider.getBlock('latest')
  const future = moment.unix(block.timestamp).add(moment.duration(10, 'm'))

  const amountOwed = await tellerV2['calculateAmountOwed(uint256,uint256)'](
    bidId,
    future.unix()
  )
  const totalOwed = amountOwed.principal.add(amountOwed.interest)

  // Get funds to the borrower
  await getFunds({
    tokenSym: await lendingToken.symbol(),
    amount: totalOwed,
    to: borrower.address,
    hre,
  })

  // Approve the contract to pull funds from borrower
  await lendingToken
    .connect(borrower.signer)
    .approve(tellerV2.address, totalOwed)

  await hre.evm.setNextBlockTimestamp(future)

  // Repay the loan
  const repayFnName = options?.fullRepay ? 'repayLoanFull' : 'repayLoanMinimum'

  const tx = await tellerV2.connect(borrower.signer)[repayFnName](bidId)

  return {
    ...values,
    tx,
  }
}, 'repaid-bid')
