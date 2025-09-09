import { BigNumber as BN, Signer } from 'ethers'
import hre, { deployments } from 'hardhat'
import { HardhatRuntimeEnvironment } from 'hardhat/types'
import { getFunds } from 'helpers/get-funds'

import {
  submittedBid,
  SubmittedBidMainReturn,
  SubmittedBidOptions,
  SubmittedBidReturn,
  SubmittedBidValues,
} from './submittedBid'

export interface AcceptedBidOptions extends SubmittedBidOptions {
  lender?: Signer
}

export interface AcceptedBidReturn
  extends AcceptedBidMainReturn,
    AcceptedBidValues {}

export interface AcceptedBidMainReturn extends SubmittedBidMainReturn {}

export interface AcceptedBidValues extends SubmittedBidValues {
  bidId: BN
  borrower: SubmittedBidValues['borrower'] & { fundsBefore: BN }
  lender: {
    signer: Signer
    address: string
  }
}

export const acceptBid = async (
  submittedBidReturn: SubmittedBidReturn,
  lender?: Signer
): Promise<AcceptedBidReturn> => {
  const {
    tellerV2,
    bidId,
    borrower,
    lendingToken,
    amount,
    marketplaceId,
    receiver,
  } = submittedBidReturn
  const bid = await tellerV2.bids(bidId)

  // eslint-disable-next-line no-param-reassign
  if (!lender) lender = await hre.getNamedSigner('lender')

  // Common values used in this fixture
  const values: AcceptedBidValues = {
    tellerV2,
    borrower: {
      ...borrower,
      fundsBefore: await lendingToken.balanceOf(borrower.address),
    },
    lender: {
      signer: lender,
      address: await lender.getAddress(),
    },
    lendingToken,
    bidId,
    amount,
    marketplaceId,
    receiver,
  }

  // get funds for lender
  const feePercent = await tellerV2.protocolFee()
  const feeAmount = amount.mul(feePercent).div(10000)
  const lenderAmount = bid.loanDetails.principal.add(feeAmount)

  await getFunds({
    tokenSym: await lendingToken.symbol(),
    amount: lenderAmount,
    to: values.lender.address,
    hre,
  })

  // approve lender allowance and accept bid
  await lendingToken.connect(lender).approve(tellerV2.address, lenderAmount)
  const tx = await tellerV2.connect(lender).lenderAcceptBid(bidId)
  await tx.wait()

  return {
    ...values,
    tx,
  }
}

export const acceptedBid = deployments.createFixture<
  AcceptedBidReturn,
  AcceptedBidOptions
>(
  async (
    hre: HardhatRuntimeEnvironment,
    options?: AcceptedBidOptions
  ): Promise<AcceptedBidReturn> => {
    const submittedBidReturn = await submittedBid(
      options as SubmittedBidOptions
    )
    return await acceptBid(submittedBidReturn)
  },
  'accepted-bid'
)
