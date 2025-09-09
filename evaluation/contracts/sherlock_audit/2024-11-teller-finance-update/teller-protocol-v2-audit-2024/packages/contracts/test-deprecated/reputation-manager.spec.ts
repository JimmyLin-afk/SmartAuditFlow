import chai, { expect } from 'chai'
import { solidity } from 'ethereum-waffle'
import { BigNumber, Signer, Wallet } from 'ethers'
import { hexlify } from 'ethers/lib/utils'
import hre from 'hardhat'
import { deploy } from 'helpers/deploy-helpers'
import { getFunds } from 'helpers/get-funds'
import moment from 'moment'
import {
  CollateralManager,
  MarketRegistry,
  ReputationManager,
  TellerV2,
} from 'types/typechain'

import {
  acceptBid,
  acceptedBid,
  AcceptedBidReturn,
} from './helpers/fixtures/acceptedBid'
import { cancelledBid } from './helpers/fixtures/cancelledBid'
import { repaidBid } from './helpers/fixtures/repaidBid'
import {
  submitBid,
  submittedBid,
  trySubmitBid,
} from './helpers/fixtures/submittedBid'

import { BidState, NULL_ADDRESS } from '~~/constants'

const paymentCycleDuration = moment.duration(30, 'days').asSeconds()
const paymentDefaultDuration = moment.duration(180, 'days').asSeconds()
const loanExpirationTime = moment.duration(1, 'days').asSeconds()

chai.should()
chai.use(solidity)

const { getNamedSigner, getNamedAccounts, ethers, toBN, deployments, tokens } =
  hre

const abiCoder = new ethers.utils.AbiCoder() // '0x0000'

// eslint-disable-next-line @typescript-eslint/no-empty-interface
interface SetupOptions {}

interface SetupReturn {
  reputationManager: ReputationManager
  tellerV2: TellerV2
  marketRegistry: MarketRegistry
}

const reputationMarks = {
  good: 0,
  delinquent: 1,
  default: 2,
}

const setup = deployments.createFixture<SetupReturn, SetupOptions>(
  async (hre, _opts) => {
    await hre.deployments.fixture(['teller-v2'], {
      keepExistingDeployments: false,
    })

    const reputationManager = await hre.contracts.get<ReputationManager>(
      'ReputationManager'
    )

    const marketRegistry = await hre.contracts.get<MarketRegistry>(
      'MarketRegistry'
    )

    const tellerV2 = await hre.contracts.get<TellerV2>('TellerV2')

    return {
      reputationManager,
      tellerV2,
      marketRegistry,
    }
  }
)

describe('ReputationManager', () => {
  let reputationManager: ReputationManager
  let marketRegistry: MarketRegistry
  let borrower: Signer
  let lender: Signer
  let marketOwner: Signer
  let borrowerAddress: string

  before(async () => {
    const result = await setup()
    reputationManager = result.reputationManager

    marketRegistry = result.marketRegistry

    borrower = await getNamedSigner('borrower')
    lender = await getNamedSigner('lender')
    marketOwner = await getNamedSigner('marketowner')

    borrowerAddress = await borrower.getAddress()

    const marketOwnerAddress = await marketOwner.getAddress()

    await marketRegistry
      .connect(marketOwner)
      [
        'createMarket(address,uint32,uint32,uint32,uint16,bool,bool,uint8,uint8,string)'
      ](
        marketOwnerAddress,
        paymentCycleDuration,
        paymentDefaultDuration,
        loanExpirationTime,
        0,
        false,
        false,
        '0',
        0,
        'uri://'
      )
      .should.emit(marketRegistry, 'MarketCreated')
      .withArgs(marketOwnerAddress, 1)
  })

  describe('getDelinquentLoanIds', () => {
    it('should update account reputation before returning values', async () => {
      const { tx, tellerV2, bidId } = await acceptedBid({ borrower, lender })

      const bidData = await tellerV2.bids(bidId)

      const bidDefaultDuration = await tellerV2.bidDefaultDuration(bidId)

      const paymentCycle = bidData.terms.paymentCycle

      //push the loan into delinquency
      await hre.evm.advanceTime(paymentCycle + 1000, {
        withoutBlocks: true,
        mine: true,
      })

      const isPaymentLate = await tellerV2.isPaymentLate(bidId)
      expect(isPaymentLate).to.eql(true, 'Expected payment late to be true')

      const isLoanDefaulted = await tellerV2.isLoanDefaulted(bidId)
      expect(isLoanDefaulted).to.eql(
        false,
        'Expected loan defaulted to be false'
      )

      const activeBidIds = await tellerV2.getBorrowerActiveLoanIds(
        borrowerAddress
      )

      expect(activeBidIds.length).to.eql(1)
      expect(activeBidIds[0]).to.eql(bidId)

      const loanIds = await reputationManager.callStatic.getDelinquentLoanIds(
        borrowerAddress
      )

      expect(loanIds.length).to.eql(1)
      expect(loanIds[0]).to.eql(bidId)
    })
  })

  describe('getDefaultedLoanIds', () => {
    it('should update account reputation before returning values', async () => {
      const { tx, tellerV2, bidId } = await acceptedBid({ borrower, lender })

      const bidData = await tellerV2.bids(bidId)

      const paymentCycle = bidData.terms.paymentCycle
      const bidDefaultDuration = await tellerV2.bidDefaultDuration(bidId)

      //push the loan into delinquency
      await hre.evm.advanceTime(bidDefaultDuration + 1000, {
        withoutBlocks: true,
        mine: true,
      })

      const isPaymentLate = await tellerV2.isPaymentLate(bidId)
      expect(isPaymentLate).to.eql(true, 'Expected payment late to be true')

      const isLoanDefaulted = await tellerV2.isLoanDefaulted(bidId)
      expect(isLoanDefaulted).to.eql(true, 'Expected loan defaulted to be true')

      const activeBidIds = await tellerV2.getBorrowerActiveLoanIds(
        borrowerAddress
      )

      expect(activeBidIds.length).to.eql(1)

      const loanIds = await reputationManager.callStatic.getDefaultedLoanIds(
        borrowerAddress
      )

      expect(loanIds.length).to.eql(1)
      expect(loanIds[0]).to.eql(bidId)
    })
  })

  describe('getCurrentDelinquentLoanIds', () => {
    it('should update account reputation before returning values', async () => {
      const { tx, tellerV2, bidId } = await acceptedBid({ borrower, lender })

      const bidData = await tellerV2.bids(bidId)

      const bidDefaultDuration = await tellerV2.bidDefaultDuration(bidId)

      const paymentCycle = bidData.terms.paymentCycle

      //push the loan into delinquency
      await hre.evm.advanceTime(paymentCycle + 1000, {
        withoutBlocks: true,
        mine: true,
      })

      const isPaymentLate = await tellerV2.isPaymentLate(bidId)
      expect(isPaymentLate).to.eql(true, 'Expected payment late to be true')

      const isLoanDefaulted = await tellerV2.isLoanDefaulted(bidId)
      expect(isLoanDefaulted).to.eql(
        false,
        'Expected loan defaulted to be false'
      )

      const activeBidIds = await tellerV2.getBorrowerActiveLoanIds(
        borrowerAddress
      )

      expect(activeBidIds.length).to.eql(1)

      const loanIds =
        await reputationManager.callStatic.getCurrentDelinquentLoanIds(
          borrowerAddress
        )

      expect(loanIds.length).to.eql(1)
      expect(loanIds[0]).to.eql(bidId)
    })
  })

  describe('getCurrentDefaultedLoanIds', () => {
    it('should update account reputation before returning values', async () => {
      const { tx, tellerV2, bidId } = await acceptedBid({ borrower, lender })

      const bidData = await tellerV2.bids(bidId)

      const paymentCycle = bidData.terms.paymentCycle
      const bidDefaultDuration = await tellerV2.bidDefaultDuration(bidId)

      //push the loan into delinquency
      await hre.evm.advanceTime(bidDefaultDuration + 1000, {
        withoutBlocks: true,
        mine: true,
      })

      const isPaymentLate = await tellerV2.isPaymentLate(bidId)
      expect(isPaymentLate).to.eql(true, 'Expected payment late to be true')

      const isLoanDefaulted = await tellerV2.isLoanDefaulted(bidId)
      expect(isLoanDefaulted).to.eql(true, 'Expected loan defaulted to be true')

      const activeBidIds = await tellerV2.getBorrowerActiveLoanIds(
        borrowerAddress
      )

      expect(activeBidIds.length).to.eql(1)

      const loanIds =
        await reputationManager.callStatic.getCurrentDefaultLoanIds(
          borrowerAddress
        )

      expect(loanIds.length).to.eql(1)
      expect(loanIds[0]).to.eql(bidId)
    })
  })

  describe('updateAccountReputation', () => {
    describe('(address _account)', () => {
      it('should fetch active loan IDs and call `_applyReputation` for each', async () => {
        const { tx, tellerV2, bidId } = await acceptedBid({ borrower, lender })

        const bidData = await tellerV2.bids(bidId)

        const paymentCycle = bidData.terms.paymentCycle
        const bidDefaultDuration = await tellerV2.bidDefaultDuration(bidId)

        //push the loan into delinquency
        await hre.evm.advanceTime(bidDefaultDuration + 1000, {
          withoutBlocks: true,
          mine: true,
        })

        await reputationManager['updateAccountReputation(address)'](
          borrowerAddress
        )
          .should.emit(reputationManager, 'MarkAdded')
          .withArgs(borrowerAddress, reputationMarks.default, bidId)

        const loanIds =
          await reputationManager.callStatic.getCurrentDefaultLoanIds(
            borrowerAddress
          )

        expect(loanIds.length).to.eql(1)
        expect(loanIds[0]).to.eql(bidId)
      })
    })

    describe('(address _account, uint256 _bidId)', () => {
      it('should call `_applyReputation`', async () => {
        const { tx, tellerV2, bidId } = await acceptedBid({ borrower, lender })

        const bidData = await tellerV2.bids(bidId)

        const paymentCycle = bidData.terms.paymentCycle
        const bidDefaultDuration = await tellerV2.bidDefaultDuration(bidId)

        //push the loan into delinquency
        await hre.evm.advanceTime(bidDefaultDuration + 1000, {
          withoutBlocks: true,
          mine: true,
        })

        await reputationManager['updateAccountReputation(address,uint256)'](
          borrowerAddress,
          bidId
        )
          .should.emit(reputationManager, 'MarkAdded')
          .withArgs(borrowerAddress, reputationMarks.default, bidId)

        const loanIds =
          await reputationManager.callStatic.getCurrentDefaultLoanIds(
            borrowerAddress
          )

        expect(loanIds.length).to.eql(1)
        expect(loanIds[0]).to.eql(bidId)
      })
    })
  })

  describe('_applyReputation', () => {
    it('should do nothing and return a Good mark for loan in good standing', async () => {
      const { tx, tellerV2, bidId } = await acceptedBid({ borrower, lender })

      const repMarkType = await reputationManager.callStatic[
        'updateAccountReputation(address,uint256)'
      ](borrowerAddress, bidId)

      expect(repMarkType).to.eql(reputationMarks.good)
    })

    it('should remove Delinquent Mark ', async () => {
      const { tx, tellerV2, bidId } = await acceptedBid({ borrower, lender })

      const bidData = await tellerV2.bids(bidId)

      const paymentCycle = bidData.terms.paymentCycle
      const bidDefaultDuration = await tellerV2.bidDefaultDuration(bidId)

      //push the loan into delinquency
      await hre.evm.advanceTime(paymentCycle + 1000, {
        withoutBlocks: true,
        mine: true,
      })

      await reputationManager['updateAccountReputation(address,uint256)'](
        borrowerAddress,
        bidId
      )
        .should.emit(reputationManager, 'MarkAdded')
        .withArgs(borrowerAddress, reputationMarks.delinquent, bidId)

      await hre.evm.advanceTime(bidDefaultDuration + 1000, {
        withoutBlocks: true,
        mine: true,
      })

      await reputationManager['updateAccountReputation(address,uint256)'](
        borrowerAddress,
        bidId
      )
        .should.emit(reputationManager, 'MarkRemoved')
        .withArgs(borrowerAddress, reputationMarks.delinquent, bidId)
    })

    it('should add Default Mark ', async () => {
      const { tx, tellerV2, bidId } = await acceptedBid({ borrower, lender })

      const bidData = await tellerV2.bids(bidId)

      const paymentCycle = bidData.terms.paymentCycle
      const bidDefaultDuration = await tellerV2.bidDefaultDuration(bidId)

      //push the loan into delinquency
      await hre.evm.advanceTime(paymentCycle + 1000, {
        withoutBlocks: true,
        mine: true,
      })

      await reputationManager['updateAccountReputation(address,uint256)'](
        borrowerAddress,
        bidId
      )
        .should.emit(reputationManager, 'MarkAdded')
        .withArgs(borrowerAddress, reputationMarks.delinquent, bidId)

      await hre.evm.advanceTime(bidDefaultDuration + 1000, {
        withoutBlocks: true,
        mine: true,
      })

      await reputationManager['updateAccountReputation(address,uint256)'](
        borrowerAddress,
        bidId
      )
        .should.emit(reputationManager, 'MarkAdded')
        .withArgs(borrowerAddress, reputationMarks.default, bidId)
    })

    it('should add Delinquent mark', async () => {
      const { tx, tellerV2, bidId } = await acceptedBid({ borrower, lender })

      const bidData = await tellerV2.bids(bidId)

      const paymentCycle = bidData.terms.paymentCycle

      //push the loan into delinquency
      await hre.evm.advanceTime(paymentCycle + 1000, {
        withoutBlocks: true,
        mine: true,
      })

      await reputationManager['updateAccountReputation(address,uint256)'](
        borrowerAddress,
        bidId
      )
        .should.emit(reputationManager, 'MarkAdded')
        .withArgs(borrowerAddress, reputationMarks.delinquent, bidId)
    })

    /*
    let's create an additional test that ensures bid IDs are:

      included when a loan is active
      removed when fully repaid

    */
    it('should ensure bid id are included when loan is active', async () => {
      const { tx, tellerV2, bidId } = await acceptedBid({ borrower, lender })

      const bidData = await tellerV2.bids(bidId)

      const bidDefaultDuration = await tellerV2.bidDefaultDuration(bidId)

      const paymentCycle = bidData.terms.paymentCycle

      //push the loan into delinquency
      await hre.evm.advanceTime(paymentCycle + 1000, {
        withoutBlocks: true,
        mine: true,
      })

      const isPaymentLate = await tellerV2.isPaymentLate(bidId)
      expect(isPaymentLate).to.eql(true, 'Expected payment late to be true')

      const isLoanDefaulted = await tellerV2.isLoanDefaulted(bidId)
      expect(isLoanDefaulted).to.eql(
        false,
        'Expected loan defaulted to be false'
      )

      const activeBidIds = await tellerV2.getBorrowerActiveLoanIds(
        borrowerAddress
      )

      expect(activeBidIds.length).to.eql(1)
      expect(activeBidIds[0]).to.eql(bidId)
    })

    it('should ensure bid id are removed when fully repaid', async () => {
      const { tx, tellerV2, bidId, lendingToken } = await acceptedBid({
        borrower,
        lender,
      })

      const bidData = await tellerV2.bids(bidId)

      const block = await hre.ethers.provider.getBlock('latest')
      const future = moment.unix(block.timestamp).add(moment.duration(10, 'm'))

      const totalOwedResult = await tellerV2[
        'calculateAmountOwed(uint256,uint256)'
      ](bidId, future.unix())
      const totalOwed = totalOwedResult.principal.add(totalOwedResult.interest)

      const activeBidIds = await tellerV2.getBorrowerActiveLoanIds(
        borrowerAddress
      )

      expect(activeBidIds.length).to.eql(1)
      expect(activeBidIds[0]).to.eql(bidId)

      await getFunds({
        tokenSym: await lendingToken.symbol(),
        amount: totalOwed,
        to: borrowerAddress,
        hre,
      })

      await lendingToken.connect(borrower).approve(tellerV2.address, totalOwed)

      const repayTx = await tellerV2.connect(borrower).repayLoanFull(bidId)

      const activeBidIdsPostRepay = await tellerV2.getBorrowerActiveLoanIds(
        borrowerAddress
      )

      expect(activeBidIdsPostRepay.length).to.eql(0)
    })
  })
})
