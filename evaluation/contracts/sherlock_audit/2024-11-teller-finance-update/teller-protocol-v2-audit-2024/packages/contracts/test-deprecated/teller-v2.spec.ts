import chai, { expect } from 'chai'
import chaiAsPromised from 'chai-as-promised'
import { BigNumber as BN, BigNumberish, Signer } from 'ethers'
import hre from 'hardhat'
import { Address } from 'hardhat-deploy/dist/types'
import { getFunds } from 'helpers/get-funds'
import { isInitialized } from 'helpers/oz-contract-helpers'
import moment from 'moment'
import {
  CollateralManager,
  LenderManager,
  MarketRegistry,
  ReputationManager,
  TellerV2,
  TellerV2Mock,
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

import { getTokens } from '~~/config'
import { BidState, NULL_ADDRESS } from '~~/constants'

chai.should()
chai.use(chaiAsPromised)

const paymentCycleDuration = moment.duration(30, 'days').asSeconds()
const loanDefaultDuration = moment.duration(180, 'days').asSeconds()
const loanExpirationDuration = moment.duration(1, 'days').asSeconds()

const { getNamedSigner, getNamedAccounts, ethers, toBN, deployments, tokens } =
  hre

// eslint-disable-next-line @typescript-eslint/no-empty-interface
interface SetupOptions {}

interface SetupReturn {
  marketRegistry: MarketRegistry
  tellerV2: TellerV2Mock
  reputationManager: ReputationManager
  lenderManager: LenderManager
}

const setup = deployments.createFixture<SetupReturn, SetupOptions>(
  async (hre, _opts) => {
    await hre.deployments.fixture(['teller-v2'], {
      keepExistingDeployments: false,
    })

    const tellerV2 = await hre.contracts.get<TellerV2Mock>('TellerV2')
    const marketRegistry = await hre.contracts.get<MarketRegistry>(
      'MarketRegistry'
    )

    const reputationManager = await hre.contracts.get<ReputationManager>(
      'ReputationManager'
    )

    const lenderManager = await hre.contracts.get<LenderManager>(
      'LenderManager'
    )

    return {
      tellerV2,
      marketRegistry,
      reputationManager,
      lenderManager,
    }
  }
)

describe('TellerV2', () => {
  let tellerV2: TellerV2Mock
  let marketRegistry: MarketRegistry
  let reputationManager: ReputationManager
  let borrower: Signer
  let lender: Signer
  let marketOwner: Signer
  let lenderManager: LenderManager

  beforeEach(async () => {
    const result = await setup()
    tellerV2 = result.tellerV2
    marketRegistry = result.marketRegistry
    reputationManager = result.reputationManager
    lenderManager = result.lenderManager
    borrower = await getNamedSigner('borrower')
    lender = await getNamedSigner('lender')
    marketOwner = await getNamedSigner('marketowner')

    const marketOwnerAddress = await marketOwner.getAddress()
    const uri = 'http://'

    await marketRegistry
      .connect(marketOwner)
      [
        'createMarket(address,uint32,uint32,uint32,uint16,bool,bool,uint8,uint8,string)'
      ](
        marketOwnerAddress,
        paymentCycleDuration,
        loanDefaultDuration,
        loanExpirationDuration,
        0,
        false,
        false,
        '0',
        0,
        uri
      )
      .should.emit(marketRegistry, 'MarketCreated')
      .withArgs(marketOwnerAddress, 1)
  })

  describe('initialize', () => {
    it('should initialize', async () => {
      await isInitialized(tellerV2.address).should.eventually.be.true
    })

    it('should revert if calling initialize again', async () => {
      const deployer = await getNamedSigner('deployer')
      await tellerV2
        .connect(deployer)
        .initialize(
          30,
          marketRegistry.address,
          reputationManager.address,
          AddressZero,
          AddressZero,
          AddressZero
        )
        .should.be.revertedWith(
          'Initializable: contract is already initialized'
        )
    })

    it('should set the owner as the deployer', async () => {
      const accounts = await getNamedAccounts()
      await tellerV2.owner().should.eventually.become(accounts.deployer)
    })
  })
  const AddressZero = ethers.constants.AddressZero
  describe('submitBid', () => {
    // eslint-disable-next-line @typescript-eslint/explicit-function-return-type
    const successfulBid = async (receiver?: Address): Promise<any> => {
      const amount = 1000
      const apr = toBN(1, 2) // 1%
      const dataHash = 'ipfs://QmMyDataHash'
      const marketplaceId = 1
      const { tx, tellerV2, bidId, borrower, lendingToken } =
        await submittedBid({
          amount,
          apr,
          dataHash,
          marketplaceId,
          receiver,
        })
      await tx.should
        .emit(tellerV2, `SubmittedBid`)
        .withArgs(
          bidId,
          borrower.address,
          receiver ?? borrower.address,
          ethers.utils.id(dataHash)
        )
      const uri = await tellerV2.getMetadataURI(bidId)
      uri.should.eql(dataHash)

      const amountBN = toBN(amount, await lendingToken.decimals())

      const bid = await tellerV2.bids(bidId)
      bid.loanDetails.principal.should.eql(
        amountBN,
        'Bid principal does not match requested value'
      )
      bid.terms.APR.should.eql(apr, 'Bid APR does not match requested value')
      return bid
    }

    it('should submit a bid for 1000 amount and 1% APR', async () => {
      await successfulBid()
    })

    it('should submit a bid for 1000 amount and 0% APR', async () => {
      const amount = 1000
      const apr = 0 // 0%
      const dataHash = 'ipfs://QmMyDataHash'
      const marketplaceId = 1
      const receiver = await getNamedSigner('receiver')
      const receiverAddress = await receiver.getAddress()
      const { tx, tellerV2, bidId, borrower, lendingToken } =
        await submittedBid({
          amount,
          apr,
          dataHash,
          marketplaceId,
          receiver: receiverAddress,
        })
      await tx.should
        .emit(tellerV2, `SubmittedBid`)
        .withArgs(
          bidId,
          borrower.address,
          receiverAddress ?? borrower.address,
          ethers.utils.id(dataHash)
        )
      const stakeholder = await tellerV2.getLoanBorrower(bidId)
      stakeholder.should.be.eql(borrower.address)
    })

    it('should submit a bid for 1000 amount and 1% APR, with an Escrow as the receiver', async () => {
      const receiverAddress = await borrower.getAddress()
      const bid = await successfulBid(receiverAddress)
      bid.receiver.should.eq(receiverAddress)
    })

    it('should be able to submit bid with custom payment cycles from market', async () => {
      const marketplaceId = 1

      const sample = async (paymentCycle: moment.Duration): Promise<void> => {
        await marketRegistry
          .connect(marketOwner)
          .setPaymentCycle(marketplaceId, 0, paymentCycle.asSeconds())

        const { tellerV2, bidId } = await submitBid({ marketplaceId })

        const bid = await tellerV2.bids(bidId)
        bid.terms.paymentCycle.should.eql(
          paymentCycle.asSeconds(),
          'Bid payment cycle doe not match requested value'
        )
      }

      // Test different values
      await sample(moment.duration(1, 'months'))

      await sample(moment.duration(2, 'months'))
    })

    it('should not be able to submit a bid when the protocol is paused', async () => {
      const deployer = await getNamedSigner('deployer')
      const { rando: dummyAddress } = await getNamedAccounts()
      await tellerV2.connect(deployer).pauseProtocol()
      const { txPromise } = await trySubmitBid({
        amountBN: 10000,
      })

      await txPromise.should.be.revertedWith('Pausable: paused')
    })

    it('should not be able to submit a bid when the market is closed', async () => {
      const deployer = await getNamedSigner('deployer')
      const { rando: dummyAddress } = await getNamedAccounts()

      await marketRegistry.connect(marketOwner).closeMarket(1)

      const { txPromise } = await trySubmitBid({
        amountBN: 10000,
        marketplaceId: 1,
      })

      await txPromise.should.be.revertedWith('Market is closed')
    })

    it('should not be able to submit a bit with a invalid token contract', async () => {
      const { rando: dummyAddress } = await getNamedAccounts()
      const { txPromise } = await trySubmitBid({
        lendingToken: dummyAddress,
        amountBN: 10000,
      })

      await txPromise.should.be.revertedWith('Lending token not authorized')
    })
  })

  describe('cancelBid', () => {
    it('should allow the borrower to cancel a pending bid', async () => {
      // submit the bid fixture
      const { tx, tellerV2, bidId } = await cancelledBid()

      // cancel the bid once (should be ok)
      await tx.should.emit(tellerV2, `CancelledBid`).withArgs(bidId)

      const bid = await tellerV2.bids(bidId)
      bid.state.should.equal(
        BidState.CANCELLED,
        'Bid state was not correctly updated'
      )
    })

    it('should not allow a rando to cancel a bid', async () => {
      const borrower = await getNamedSigner('borrower')
      const { tellerV2, bidId } = await submittedBid({ borrower })

      const rando = await getNamedSigner('rando')
      await tellerV2
        .connect(rando)
        .cancelBid(bidId)
        .should.be.revertedWith(
          `ActionNotAllowed(${bidId.toString()}, "cancelBid", "Only the bid owner can cancel!")`
        )
    })

    it('should not be able to cancel a loan that was already cancelled', async () => {
      // submit the bid fixture
      const { tellerV2, borrower, bidId } = await cancelledBid()

      // cancel the bid again (should fail)
      await tellerV2
        .connect(borrower.signer)
        .cancelBid(bidId)
        .should.be.revertedWith(
          `ActionNotAllowed(${bidId.toString()}, "cancelBid", "Bid must be pending")`
        )
    })

    it('should be able to cancel a bid when the protocol is paused', async () => {
      const borrower = await getNamedSigner('borrower')
      const deployer = await getNamedSigner('deployer')
      const { tellerV2, bidId } = await submittedBid({ borrower })

      await tellerV2.connect(deployer).pauseProtocol()

      await tellerV2
        .connect(borrower)
        .cancelBid(bidId)
        .should.emit(tellerV2, `CancelledBid`)
        .withArgs(bidId)
    })
  })

  describe('marketOwnerCancelBid', () => {
    it("should allow the bid's market owner to cancel a bid", async () => {
      const borrower = await getNamedSigner('borrower')
      const { tellerV2, bidId } = await submittedBid({ borrower })

      const tx = tellerV2.connect(marketOwner).marketOwnerCancelBid(bidId)
      await tx.should.emit(tellerV2, `CancelledBid`).withArgs(bidId)
      await tx.should.emit(tellerV2, `MarketOwnerCancelledBid`).withArgs(bidId)

      const bid = await tellerV2.bids(bidId)
      bid.state.should.equal(
        BidState.CANCELLED,
        'Bid state was not correctly updated'
      )
    })
  })

  describe('lenderAcceptBid', () => {
    it('accepts a bid', async () => {
      const { tx, tellerV2, lender, bidId } = await acceptedBid()

      const bid = await tellerV2.bids(bidId)

      const principalAmount = bid.loanDetails.principal
      const feePercent = await tellerV2.protocolFee()
      const feeAmount = principalAmount.mul(feePercent).div(10000)
      const amountToBorrower = principalAmount.sub(feeAmount)

      // Check the event was emitted
      await tx.should
        .emit(tellerV2, 'AcceptedBid')
        .withArgs(bidId, lender.address)

      const stakeholder = await lenderManager.ownerOf(bidId)
      stakeholder.should.be.eql(lender.address)
    })

    it('should not be able to accept a bid when the protocol is paused', async () => {
      // submit the bid fixture
      const { tellerV2, bidId } = await submittedBid()

      // pause the protocol
      const deployer = await getNamedSigner('deployer')
      await tellerV2.connect(deployer).pauseProtocol()

      // get lender
      const lender = await getNamedSigner('lender')

      // accept the bid (should fail)
      await tellerV2
        .connect(lender)
        .lenderAcceptBid(bidId)
        .should.be.revertedWith('Pausable: paused')
    })

    it('should not be able to accept a bid when the market is closed', async () => {
      // submit the bid fixture
      const { tellerV2, bidId } = await submittedBid()

      await marketRegistry.connect(marketOwner).closeMarket(1)

      // pause the protocol
      const deployer = await getNamedSigner('deployer')
      // /await tellerV2.connect(deployer).pauseProtocol()

      // get lender
      const lender = await getNamedSigner('lender')

      // accept the bid (should fail)
      await tellerV2
        .connect(lender)
        .lenderAcceptBid(bidId)
        .should.be.revertedWith('Market is closed')
    })

    it('should send borrower funds from lender + protocol fee', async () => {
      const marketOwnerAddress = await marketOwner.getAddress()

      const uri = 'ipfs://QmMyDataHash'
      const marketplaceId = 1

      await marketRegistry
        .connect(marketOwner)
        .setMarketFeePercent(marketplaceId, 100)

      const amountBN = BN.from(100000)
      const protocolFee = await tellerV2.protocolFee()
      const protocolFeeAmount = amountBN.mul(protocolFee).div(10000)

      const marketFee = await marketRegistry.getMarketplaceFee(marketplaceId)
      const marketFeeAmount = amountBN.mul(marketFee).div(10000)

      marketFee.should.eql(100)

      const borrowerAmount = amountBN
        .sub(protocolFeeAmount)
        .sub(marketFeeAmount)

      const bidId = await tellerV2.bidId()
      const token = await tokens.get('usdc')

      await tellerV2.mockBid(
        buildBid({
          borrower: await borrower.getAddress(),
          lender: NULL_ADDRESS,
          receiver: await borrower.getAddress(),
          loanDetails: {
            lendingToken: token.address,
            principal: amountBN,
          },
        })
      )

      await getFunds({
        tokenSym: await token.symbol(),
        to: await lender.getAddress(),
        amount: amountBN,
        hre,
      })
      await token.connect(lender).approve(tellerV2.address, amountBN)
      const fn = (): Promise<any> =>
        tellerV2.connect(lender).lenderAcceptBid(bidId)

      await fn.should.changeTokenBalances(
        token,
        [
          { getAddress: () => tellerV2.owner() },
          { getAddress: () => marketOwnerAddress },
          borrower,
          lender,
        ],
        [protocolFeeAmount, marketFeeAmount, borrowerAmount, amountBN.mul(-1)]
      )
    })

    it('should update volume filled', async () => {
      const lenderVolume: { [lender: string]: BN | undefined } = {}
      let expectedTotalVolume: BN = BN.from(0)
      const check = async (result: AcceptedBidReturn): Promise<void> => {
        // Increase the expected volume
        lenderVolume[result.lender.address] =
          lenderVolume[result.lender.address]?.add(result.amount) ??
          BN.from(result.amount)
        expectedTotalVolume = expectedTotalVolume.add(result.amount)

        const lenderFilled = await tellerV2.lenderVolumeFilled(
          result.lendingToken.address,
          result.lender.address
        )
        lenderFilled.should.eql(
          lenderVolume[result.lender.address],
          'Lender volume filled should equal the bid amount'
        )

        const totalFilled = await tellerV2.totalVolumeFilled(
          result.lendingToken.address
        )
        totalFilled.should.eql(
          expectedTotalVolume,
          'Total volume filled should equal the bid amount'
        )
      }

      // First time around use the fixture
      const result1 = await acceptedBid({
        lender: await getNamedSigner('lender'),
      })
      await check(result1)

      // Submit another bid to then accept
      const result2 = await acceptBid(
        await submitBid(),
        await getNamedSigner('lender2')
      )
      await check(result2)
    })

    it('should not be able to accept a bid after it expires', async () => {
      // submit the bid fixture
      const { tellerV2, bidId } = await submittedBid()

      // extend the time past the expiration window
      const block = await hre.ethers.provider.getBlock('latest')
      const future = moment.unix(block.timestamp).add(moment.duration(10, 'd'))
      await hre.evm.setNextBlockTimestamp(future)

      // get lender
      const lender = await getNamedSigner('lender')

      // accept the bid (should fail)
      await tellerV2
        .connect(lender)
        .lenderAcceptBid(bidId)
        .should.be.revertedWith('Bid has expired')
    })

    it('should fail when approving a non-existent bid', async () => {
      const bogusId = 100
      await tellerV2
        .lenderAcceptBid(bogusId)
        .should.be.revertedWith(
          `ActionNotAllowed(${bogusId}, "lenderAcceptBid", "Bid must be pending")`
        )
    })

    it('should fail when trying to accept an already accepted bid', async () => {
      const { tellerV2, bidId, lender } = await acceptedBid()
      await tellerV2
        .connect(lender.signer)
        .lenderAcceptBid(bidId)
        .should.be.revertedWith(
          `ActionNotAllowed(${bidId}, "lenderAcceptBid", "Bid must be pending")`
        )
    })

    it('cannot accept a loan that was cancelled', async () => {
      // submit the bid fixture
      const { tellerV2, borrower, bidId } = await submittedBid()

      // cancel the bid
      await tellerV2.connect(borrower.signer).cancelBid(bidId)

      // get lender
      const lender = await getNamedSigner('lender')

      // accept the bid (should fail)
      await tellerV2
        .connect(lender)
        .lenderAcceptBid(bidId)
        .should.be.revertedWith(
          `ActionNotAllowed(${bidId.toString()}, "lenderAcceptBid", "Bid must be pending")`
        )
    })
  })

  describe('repayLoan', () => {
    it('borrower repays the minimum owed on loan', async () => {
      const { tx, tellerV2, bidId } = await repaidBid()

      await tx.should.emit(tellerV2, 'LoanRepayment').withArgs(bidId)
    })

    it('cannot repay a paid loan', async () => {
      // repay the loan completely then try to repay again. should revert with error
      const { tellerV2, borrower, bidId } = await repaidBid({ fullRepay: true })

      // repay the loan
      await tellerV2
        .connect(borrower.signer)
        .repayLoan(bidId, 1)
        .should.be.revertedWith(
          `ActionNotAllowed(${bidId.toString()}, "repayLoan", "Loan must be accepted")`
        )
    })

    it('cannot repay a loan that was not accepted', async () => {
      // submit the bid fixture
      const { tellerV2, borrower, bidId } = await submittedBid()

      await tellerV2
        .connect(borrower.signer)
        .repayLoan(bidId, 1)
        .should.be.revertedWith(
          `ActionNotAllowed(${bidId.toString()}, "repayLoan", "Loan must be accepted")`
        )
    })

    it('cannot repay a loan that was cancelled', async () => {
      // submit the bid fixture
      const { tellerV2, borrower, bidId } = await submittedBid()

      // cancel the bid
      await tellerV2.connect(borrower.signer).cancelBid(bidId)

      await tellerV2
        .connect(borrower.signer)
        .repayLoan(bidId, 1)
        .should.be.revertedWith(
          `ActionNotAllowed(${bidId.toString()}, "repayLoan", "Loan must be accepted")`
        )
    })

    it('should fail when repaying less than minimum owed', async () => {
      const { tellerV2, borrower, bidId } = await acceptedBid()

      const block = await hre.ethers.provider.getBlock('latest')
      const future = moment.unix(block.timestamp).add(moment.duration(10, 'd'))
      const amountDue = await tellerV2['calculateAmountDue(uint256,uint256)'](
        bidId,
        future.unix()
      )
      const minimumOwed = amountDue.principal.add(amountDue.interest)
      const amountToRepay = minimumOwed.sub(1)

      await hre.evm.setNextBlockTimestamp(future)

      // Try repaying the loan
      await tellerV2
        .connect(borrower.signer)
        .repayLoan(bidId, amountToRepay)
        .should.be.revertedWith(
          `PaymentNotMinimum(${bidId}, ${amountToRepay.toString()}, ${minimumOwed.toString()})`
        )
    })

    it('should only transfer amount owed when over paying loan in full', async () => {
      const { tellerV2, borrower, bidId, lendingToken } = await acceptedBid()

      const block = await hre.ethers.provider.getBlock('latest')
      const future = moment.unix(block.timestamp).add(moment.duration(10, 'd'))
      const amountOwed = await tellerV2['calculateAmountOwed(uint256,uint256)'](
        bidId,
        future.unix()
      )
      const totalOwed = amountOwed.principal.add(amountOwed.interest)
      const amountToRepay = totalOwed.add(100)

      // Get funds to ensure we can pay
      await getFunds({
        to: borrower.address,
        tokenSym: await lendingToken.symbol(),
        amount: amountToRepay,
        hre,
      })
      // Make sure borrower allowance is set
      await lendingToken
        .connect(borrower.signer)
        .approve(tellerV2.address, amountToRepay)

      const beforeBal = await lendingToken.balanceOf(borrower.address)

      await hre.evm.setNextBlockTimestamp(future)

      // Repay the loan
      await tellerV2
        .connect(borrower.signer)
        .repayLoan(bidId, amountToRepay)
        .should.emit(tellerV2, 'LoanRepaid')
        .withArgs(bidId)

      const afterBal = await lendingToken.balanceOf(borrower.address)
      beforeBal
        .sub(afterBal)
        .should.eql(
          totalOwed,
          'Total amount owed was not the value transferred for repayment'
        )
    })
  })

  describe('liquidateLoan', () => {
    // liquidate loan full
    it('should not be able to liquidate loan that is in good standing', async () => {
      const { tx, tellerV2, borrower, lender, bidId, lendingToken } =
        await acceptedBid({})

      const bidData = await tellerV2.bids(bidId)

      const block = await hre.ethers.provider.getBlock('latest')
      const future = moment.unix(block.timestamp).add(moment.duration(10, 'm'))

      const totalOwedResult = await tellerV2[
        'calculateAmountOwed(uint256,uint256)'
      ](bidId, future.unix())
      const totalOwed = totalOwedResult.principal.add(totalOwedResult.interest)

      const activeBidIds = await tellerV2.getBorrowerActiveLoanIds(
        borrower.address
      )

      expect(activeBidIds.length).to.eql(1)
      expect(activeBidIds[0]).to.eql(bidId)

      await getFunds({
        tokenSym: await lendingToken.symbol(),
        amount: totalOwed,
        to: lender.address,
        hre,
      })

      await lendingToken
        .connect(lender.signer)
        .approve(tellerV2.address, totalOwed)

      const liquidateTx = await tellerV2
        .connect(lender.signer)
        .liquidateLoanFull(bidId)
        .should.be.revertedWith('Loan must be defaulted.')
    })

    it('should be able to liquidate loan that is in bad standing', async () => {
      const { tx, tellerV2, borrower, lender, bidId, lendingToken } =
        await acceptedBid({})

      const bidDefaultDuration = await tellerV2.bidDefaultDuration(bidId)

      // push the loan into delinquency
      await hre.evm.advanceTime(bidDefaultDuration + 1000, {
        withoutBlocks: true,
        mine: true,
      })

      const isPaymentLate = await tellerV2.isPaymentLate(bidId)
      expect(isPaymentLate).to.eql(true, 'Expected payment late to be true')

      const isLoanDefaulted = await tellerV2.isLoanDefaulted(bidId)
      expect(isLoanDefaulted).to.eql(true, 'Expected loan defaulted to be true')

      const block = await hre.ethers.provider.getBlock('latest')
      const future = moment.unix(block.timestamp).add(moment.duration(10, 'm'))

      const totalOwedResult = await tellerV2[
        'calculateAmountOwed(uint256,uint256)'
      ](bidId, future.unix())
      const totalOwed = totalOwedResult.principal.add(totalOwedResult.interest)

      const activeBidIds = await tellerV2.getBorrowerActiveLoanIds(
        borrower.address
      )

      expect(activeBidIds.length).to.eql(1)
      expect(activeBidIds[0]).to.eql(bidId)

      await getFunds({
        tokenSym: await lendingToken.symbol(),
        amount: totalOwed,
        to: lender.address,
        hre,
      })

      await lendingToken
        .connect(lender.signer)
        .approve(tellerV2.address, totalOwed)

      const liquidatePromise = tellerV2
        .connect(lender.signer)
        .liquidateLoanFull(bidId)

      await liquidatePromise.should.emit(tellerV2, 'LoanRepaid').withArgs(bidId)

      await liquidatePromise.should
        .emit(tellerV2, 'LoanLiquidated')
        .withArgs(bidId, lender.address)

      const bidData = await tellerV2.bids(bidId)
      bidData.state.should.eql(
        BidState.LIQUIDATED,
        'Bid state should be set to LIQUIDATED'
      )
    })
  })
})

type BidParams = Parameters<TellerV2Mock['mockBid']>[0]
export const buildBid = (values: PartialNested<BidParams>): BidParams =>
  // eslint-disable-next-line @typescript-eslint/consistent-type-assertions
  <BidParams>{
    borrower: NULL_ADDRESS,
    lender: NULL_ADDRESS,
    receiver: NULL_ADDRESS,
    _metadataURI: ethers.utils.id(''),
    marketplaceId: 1,
    state: BidState.PENDING,
    ...values,

    // NOTE: Nested objects must be extended __AFTER__ top level properties
    loanDetails: {
      lendingToken: values.loanDetails?.lendingToken ?? NULL_ADDRESS,
      principal: values.loanDetails?.principal ?? 0,
      timestamp: moment().unix(),
      acceptedTimestamp: 0,
      lastRepaidTimestamp: 0,
      loanDuration: moment.duration(365, 'd').asSeconds() * 3,
      totalRepaid: {
        principal: 0,
        interest: 0,
        ...(values.loanDetails?.totalRepaid ?? {}),
      },
      ...(values.loanDetails ?? {}),
    },
    terms: {
      paymentCycleAmount: 0,
      paymentCycle: moment.duration(365, 'd').asSeconds() / 12,
      APR: 100,
      defaultDuration: moment.duration(180, 'd').asSeconds() / 12,
      loanExpirationTime: moment.duration(1, 'd').asSeconds() / 12,
      ...(values.terms ?? {}),
    },
    paymentType: 0,
  }

describe('TellerV2', () => {
  let tellerV2: TellerV2Mock
  let lender: Signer

  beforeEach(async () => {
    const result = await setup()
    tellerV2 = result.tellerV2
    lender = await getNamedSigner('lender')
  })

  describe('calculateAmountOwed', () => {
    interface TestArgs {
      expected: BigNumberish
      bidParams: PartialNested<BidParams>
      nowIncrease: moment.Duration
    }

    const test =
      (args: TestArgs): (() => Promise<void>) =>
      async () => {
        const { expected, bidParams, nowIncrease } = args

        const bidId = await tellerV2.bidId()

        const builtBid = buildBid(bidParams)

        await tellerV2.mockBid(builtBid)

        await tellerV2['mockAcceptedTimestamp(uint256)'](bidId)

        await hre.evm.advanceTime(nowIncrease, {
          withoutBlocks: true,
          mine: true,
        })

        const owed = await tellerV2['calculateAmountOwed(uint256)'](bidId)

        owed.principal
          .add(owed.interest)
          .should.eql(expected, 'Estimated amount owed value is incorrect')
      }

    // Build test scenario arguments
    const scenarios: TestArgs[] = [
      {
        expected: toBN(110, 6),
        bidParams: {
          loanDetails: { principal: toBN(100, 6) },
          terms: {
            paymentCycleAmount: 3226719,
            APR: 1000,
          },
          state: BidState.ACCEPTED,
        },
        nowIncrease: moment.duration(365, 'd'),
      },
      {
        expected: toBN(105, 6),
        bidParams: {
          loanDetails: { principal: toBN(100, 6) },
          terms: {
            paymentCycleAmount: 3226719,
            APR: 1000,
          },
          state: BidState.ACCEPTED,
        },
        nowIncrease: moment.duration(182.5, 'd'),
      },
      {
        expected: toBN(1025, 5),
        bidParams: {
          loanDetails: { principal: toBN(100, 6) },
          terms: {
            paymentCycleAmount: 3226719,
            APR: 1000,
          },
          state: BidState.ACCEPTED,
        },
        nowIncrease: moment.duration(91.25, 'd'),
      },
    ]

    // Run the test scenarios
    scenarios.forEach((args, i) => it(`scenario #${i}`, test(args)))
  })

  describe('calculateAmountDue', () => {
    interface TestArgs {
      expected: BigNumberish
      bidParams: PartialNested<BidParams>
      nowIncrease: moment.Duration
      repayIncrease: moment.Duration
    }

    const test = (args: TestArgs) => async () => {
      const { expected, bidParams, nowIncrease, repayIncrease } = args

      const bidId = await tellerV2.bidId()
      await tellerV2.mockBid(buildBid(bidParams))

      const acceptedTimestamp = moment().unix()
      const acceptTime = moment.duration(acceptedTimestamp, 's')
      await tellerV2['mockAcceptedTimestamp(uint256,uint32)'](
        bidId,
        acceptedTimestamp
      )

      if (repayIncrease.asSeconds() > 0) {
        await tellerV2.mockLastRepaidTimestamp(
          bidId,
          acceptTime.clone().add(repayIncrease).asSeconds()
        )
      }

      const future = acceptTime.clone().add(nowIncrease).asSeconds()

      const amountDue = await tellerV2['calculateAmountDue(uint256,uint256)'](
        bidId,
        future
      )
      amountDue.principal
        .add(amountDue.interest)
        .should.eql(expected, 'Amount due value incorrect')
    }

    const scenarios: TestArgs[] = [
      {
        expected: 38720628,
        bidParams: {
          loanDetails: { principal: toBN(100, 6) },
          terms: {
            paymentCycleAmount: 3226719,
            APR: 1000,
          },
          state: BidState.ACCEPTED,
        },
        nowIncrease: moment.duration(365, 'd'),
        repayIncrease: moment.duration(0, 'd'),
      },
      {
        expected: 9680157,
        bidParams: {
          loanDetails: { principal: toBN(100, 6) },
          terms: {
            paymentCycleAmount: 3226719,
            APR: 1000,
          },
          state: BidState.ACCEPTED,
        },
        nowIncrease: moment.duration(91.25, 'd'),
        repayIncrease: moment.duration(0, 'd'),
      },
      {
        expected: 9680157,
        bidParams: {
          loanDetails: { principal: toBN(100, 6) },
          terms: {
            paymentCycleAmount: 3226719,
            APR: 1000,
          },
          state: BidState.ACCEPTED,
        },
        nowIncrease: moment.duration(182.5, 'd'),
        repayIncrease: moment.duration(91.25, 'd'),
      },
    ]

    scenarios.forEach((args, i) => it(`scenario #${i}`, test(args)))
  })
})

describe('TellerV2', () => {
  let tellerV2: TellerV2Mock
  let result: AcceptedBidReturn
  let bid: PromiseReturnType<TellerV2['bids']>

  let marketRegistry: MarketRegistry
  let marketOwner: Signer

  beforeEach(async () => {
    const setupResult = await setup()
    tellerV2 = setupResult.tellerV2
    marketRegistry = setupResult.marketRegistry
    marketOwner = await getNamedSigner('marketowner')

    const marketOwnerAddress = await marketOwner.getAddress()
    const uri = 'http://'

    await marketRegistry
      .connect(marketOwner)
      [
        'createMarket(address,uint32,uint32,uint32,uint16,bool,bool,uint8,uint8,string)'
      ](
        marketOwnerAddress,
        paymentCycleDuration,
        loanDefaultDuration,
        loanExpirationDuration,
        0,
        false,
        false,
        '0',
        0,
        uri
      )
      .should.emit(marketRegistry, 'MarketCreated')
      .withArgs(marketOwnerAddress, 1)

    result = await acceptedBid()
    bid = await tellerV2.bids(result.bidId)
  })

  describe('calculateNextDueDate', () => {
    it('should be able to calculate the first due date on a new loan', async () => {
      const expectedDueDate = moment
        .duration(bid.loanDetails.acceptedTimestamp, 's')
        .add(moment.duration(bid.terms.paymentCycle, 's'))
        .asSeconds()

      const dueDate = await result.tellerV2.calculateNextDueDate(result.bidId)
      dueDate.should.eql(expectedDueDate, 'Due date does not match expected')
    })

    it('should calculate the next due date after being repaid in the first cycle', async () => {
      result = await repaidBid()

      const cycle = moment.duration(bid.terms.paymentCycle, 's')
      const expectedDueDate = moment
        .duration(bid.loanDetails.acceptedTimestamp, 's')
        .add(cycle)
        .add(cycle)
        .asSeconds()

      const dueDate = await result.tellerV2.calculateNextDueDate(result.bidId)
      dueDate.should.eql(expectedDueDate, 'Due date does not match expected')
    })

    it('should return false with non-accepted bid id', async () => {
      const { bidId } = await submittedBid()

      const response = await result.tellerV2.calculateNextDueDate(bidId)

      response.should.eql(0)
    })
  })

  describe('isPaymentLate', () => {
    it('should not be late after being accepted', async () => {
      const isLate = await tellerV2.isPaymentLate(result.bidId)
      // eslint-disable-next-line @typescript-eslint/no-unused-expressions
      isLate.should.be.false
    })

    it('should be late after surpassing payment cycle duration', async () => {
      await hre.evm.advanceTime(
        moment.duration(bid.terms.paymentCycle + 1, 's'),
        {
          withoutBlocks: true,
          mine: true,
        }
      )

      const isLate = await tellerV2.isPaymentLate(result.bidId)
      // eslint-disable-next-line @typescript-eslint/no-unused-expressions
      isLate.should.be.true
    })
  })
})
