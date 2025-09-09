import chai from 'chai'
import { solidity } from 'ethereum-waffle'
import { Signer } from 'ethers'
import hre from 'hardhat'
import { LenderCommitmentForwarder } from 'types/typechain'

const { ethers, deployments } = hre

chai.should()
chai.use(solidity)

// eslint-disable-next-line @typescript-eslint/no-empty-interface
interface SetupOptions {}

interface SetupReturn {
  LenderCommitmentForwarder: LenderCommitmentForwarder
  lender: Signer
}

// const setup = deployments.createFixture<SetupReturn, SetupOptions>(
//   async (hre, _opts) => {
//     await hre.deployments.fixture(['teller-v2'], {
//       keepExistingDeployments: false,
//     })
//
//     const LenderCommitmentForwarder =
//       await hre.contracts.get<LenderCommitmentForwarder>(
//         'LenderCommitmentForwarder'
//       )
//
//     const lender = await hre.getNamedSigner('lender')
//
//     return {
//       LenderCommitmentForwarder,
//       lender,
//     }
//   }
// )
//
// describe.skip('LenderCommitmentForwarder', () => {
//   let LenderCommitmentForwarder: LenderCommitmentForwarder
//   let lender: Signer
//   let lenderAddress: string
//   const tokenAddress = '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174'
//   const marketId = 2
//   const maxAmount = 1000000
//   const maxLoanDuration = 24800
//   const minAPR = 3000
//   const expiration = 64000
//
//   before(async () => {
//     const result = await setup()
//     LenderCommitmentForwarder = result.LenderCommitmentForwarder
//     lender = result.lender
//     lenderAddress = await lender.getAddress()
//   })
//
//   describe('updateCommitment', () => {
//     it('should update a lender commitment', async () => {
//       const blockNumber = await hre.ethers.provider.getBlockNumber()
//
//       const block = await hre.ethers.provider.getBlock(blockNumber)
//
//       await LenderCommitmentForwarder.connect(lender)
//         .updateCommitment(
//           marketId,
//           tokenAddress,
//           maxAmount,
//           maxLoanDuration,
//           minAPR,
//           expiration + block.timestamp
//         )
//         .should.emit(LenderCommitmentForwarder, 'UpdatedCommitment')
//         .withArgs(lenderAddress, marketId, tokenAddress, maxAmount)
//
//       const storedCommitment =
//         await LenderCommitmentForwarder.lenderMarketCommitments(
//           lenderAddress,
//           marketId,
//           tokenAddress
//         )
//       storedCommitment.expiration.should.be.eq(expiration + block.timestamp)
//       storedCommitment.minInterestRate.should.be.eq(minAPR)
//       storedCommitment.maxPrincipal.should.be.eq(maxAmount)
//       storedCommitment.maxDuration.should.be.eq(maxLoanDuration)
//     })
//   })
//   describe('deleteCommitment', () => {
//     it('should remove a lender commitment', async () => {
//       const blockNumber = await hre.ethers.provider.getBlockNumber()
//
//       const block = await hre.ethers.provider.getBlock(blockNumber)
//
//       await LenderCommitmentForwarder.connect(lender).updateCommitment(
//         marketId,
//         tokenAddress,
//         maxAmount,
//         maxLoanDuration,
//         minAPR,
//         expiration + block.timestamp
//       )
//
//       await LenderCommitmentForwarder.connect(lender)
//         .deleteCommitment(marketId, tokenAddress)
//         .should.emit(LenderCommitmentForwarder, 'DeletedCommitment')
//         .withArgs(lenderAddress, marketId, tokenAddress)
//
//       const storedCommitment =
//         await LenderCommitmentForwarder.lenderMarketCommitments(
//           lenderAddress,
//           marketId,
//           tokenAddress
//         )
//       storedCommitment.expiration.should.be.eq(0)
//       storedCommitment.minInterestRate.should.be.eq(0)
//       storedCommitment.maxPrincipal.should.be.eq(0)
//       storedCommitment.maxDuration.should.be.eq(0)
//     })
//   })
// })
