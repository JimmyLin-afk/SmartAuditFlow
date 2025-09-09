import { task } from 'hardhat/config'
import { HardhatRuntimeEnvironment } from 'hardhat/types'
import { CollateralManager, TellerV2 } from 'types/typechain'

interface SubmitCollateralBidArgs {
  marketPlaceId: number
  lendingTokenName: string
  amountToBorrow: number
  durationInSeconds: number
  apr: number
  metadataURI: string
  collateralTokenName: string
  collateralAmount: number
}

export const submitCollateralBid = async (
  args: SubmitCollateralBidArgs,
  hre: HardhatRuntimeEnvironment
): Promise<void> => {
  const { contracts, toBN, log, tokens } = hre
  const {
    marketPlaceId,
    lendingTokenName,
    amountToBorrow,
    durationInSeconds,
    apr,
    metadataURI,
    collateralTokenName,
    collateralAmount,
  } = args

  // Load contracts
  const collateralToken = await tokens.get(collateralTokenName)
  const lendingToken = await tokens.get(lendingTokenName)
  const tellerV2 = await contracts.get<TellerV2>('TellerV2')
  const collateralManager = await contracts.get<CollateralManager>(
    'CollateralManager'
  )

  const borrower = await hre.ethers.provider.getSigner()
  const collateralAmountBN = toBN(
    collateralAmount,
    await collateralToken.decimals()
  )

  // Submit bid
  const bidId = await tellerV2[
    'submitBid(address,uint256,uint256,uint32,uint16,string,address,(uint8,uint256,uint256,address)[])'
  ](
    lendingToken, // Lending token
    marketPlaceId, // Marketplace Id
    toBN(amountToBorrow, await lendingToken.decimals()), // Principal
    durationInSeconds.toString(), // Duration
    apr.toString(), // APR
    metadataURI, // MetadataURI
    borrower, // Receiver
    [
      {
        _collateralType: 0, // ERC20
        _amount: collateralAmountBN,
        _tokenId: 0,
        _collateralAddress: collateralToken,
      },
    ]
  )

  // Approve collateral manager to be able to pull in committed collateral
  await collateralToken.approve(collateralManager, collateralAmountBN)

  log(`Submitted collateral bid for ${await borrower.getAddress()}`, {
    indent: 4,
  })
}

task('submit-collateral-bid', 'submits a collateral backed bid')
  .addParam('marketPlaceId', 'The id of the marketplace to submit the bid to')
  .addParam('lendingTokenName', 'The name of the ERC20 token to borrow')
  .addParam('amountToBorrow', 'The amount to borrow')
  .addParam('durationInSeconds', 'The length of time to take out the loan for')
  .addParam('apr', 'The proposed apr for the loan')
  .addParam(
    'metadataURI',
    'The URI of the associated metadata required for the loan'
  )
  .addParam(
    'collateralTokenName',
    'The name of the ERC20 token to use as collateral'
  )
  .addParam(
    'collateralAmount',
    'The amount of the collateral token to use as collateral for the loan'
  )
  .setAction(submitCollateralBid)
