import { ethers, network } from 'hardhat'
import { OrderLib } from '../typechain-types/contracts/AegisMinting'
import { ClaimRewardsLib } from '../typechain-types/contracts/AegisRewards'
import { HDNodeWallet } from 'ethers'

export const DEFAULT_ADMIN_ROLE = '0x00'
export const SETTINGS_MANAGER_ROLE = ethers.id('SETTINGS_MANAGER_ROLE')
export const FUNDS_MANAGER_ROLE = ethers.id('FUNDS_MANAGER_ROLE')
export const COLLATERAL_MANAGER_ROLE = ethers.id('COLLATERAL_MANAGER_ROLE')
export const REWARDS_MANAGER_ROLE = ethers.id('REWARDS_MANAGER_ROLE')
export const OPERATOR_ROLE = ethers.id('OPERATOR_ROLE')

export const USD_FEED_ADDRESS = '0x0000000000000000000000000000000000000348'

export enum OrderType {
  MINT,
  REDEEM,
  DEPOSIT_INCOME,
}

export enum RedeemRequestStatus {
  PENDING,
  APPROVED,
  REJECTED,
  WITHDRAWN,
}

export const MAX_BPS = 10_000n
export const INCOME_FEE_BP = 500n // 5%

export const trustedSignerAccount = ethers.Wallet.createRandom()
export const insuranceFundAccount = ethers.Wallet.createRandom()
export const custodianAccount = ethers.Wallet.createRandom()

export async function deployFixture() {
  const [owner] = await ethers.getSigners()

  const assetContract = await ethers.deployContract('TestToken', ['Test', 'TST', 18])
  const assetAddress = await assetContract.getAddress()

  const yusdContract = await ethers.deployContract('YUSD', [owner.address])
  const yusdAddress = await yusdContract.getAddress()

  const aegisConfig = await ethers.deployContract('AegisConfig', [trustedSignerAccount, [owner], owner])
  const aegisConfigAddress = await aegisConfig.getAddress()

  const aegisRewardsContract = await ethers.deployContract('AegisRewards', [
    yusdAddress,
    aegisConfig,
    owner,
  ])
  const aegisRewardsAddress = await aegisRewardsContract.getAddress()

  const aegisMintingContract = await ethers.deployContract('AegisMinting', [
    yusdAddress,
    aegisConfig,
    aegisRewardsAddress,
    ethers.ZeroAddress,
    ethers.ZeroAddress,
    insuranceFundAccount.address,
    [
      assetAddress,
    ],
    [
      86400,
    ],
    [
      custodianAccount.address,
    ],
    owner.address,
  ])
  const aegisMintingAddress = await aegisMintingContract.getAddress()

  await yusdContract.setMinter(aegisMintingAddress)
  await aegisRewardsContract.setAegisMintingAddress(aegisMintingAddress)

  return {
    yusdContract,
    yusdAddress,
    aegisRewardsContract,
    aegisRewardsAddress,
    aegisMintingContract,
    aegisMintingAddress,
    assetContract,
    assetAddress,
    aegisConfig,
    aegisConfigAddress,
  }
}

export async function signOrderByWallet(order: OrderLib.OrderStruct, contractAddress: string, wallet: HDNodeWallet) {
  return wallet.signTypedData(
    {
      name: 'AegisMinting',
      version: '1',
      chainId: 1337n,
      verifyingContract: contractAddress,
    },
    {
      Order: [
        {name: 'orderType', type: 'uint8'},
        {name: 'userWallet', type: 'address'},
        {name: 'collateralAsset', type: 'address'},
        {name: 'collateralAmount', type: 'uint256'},
        {name: 'yusdAmount', type: 'uint256'},
        {name: 'slippageAdjustedAmount', type: 'uint256'},
        {name: 'expiry', type: 'uint256'},
        {name: 'nonce', type: 'uint256'},
        {name: 'additionalData', type: 'bytes'},
      ],
    },
    order,
  )
}

export async function signOrder(order: OrderLib.OrderStruct, contractAddress: string) {
  return signOrderByWallet(order, contractAddress, trustedSignerAccount)
}

export async function signClaimRequestByWallet(request: ClaimRewardsLib.ClaimRequestStruct, contractAddress: string, wallet: HDNodeWallet) {
  return wallet.signTypedData(
    {
      name: 'AegisRewards',
      version: '1',
      chainId: 1337n,
      verifyingContract: contractAddress,
    },
    {
      ClaimRequest: [
        {name: 'claimer', type: 'address'},
        {name: 'ids', type: 'bytes32[]'},
        {name: 'amounts', type: 'uint256[]'},
      ],
    },
    request,
  )
}

export async function signClaimRequest(request: ClaimRewardsLib.ClaimRequestStruct, contractAddress: string) {
  return signClaimRequestByWallet(request, contractAddress, trustedSignerAccount)
}

export function encodeString(str: string) {
  return ethers.AbiCoder.defaultAbiCoder().encode(
    ['string'],
    [str],
  )
}

export async function executeInBatch(...promises: Promise<any>[]) {
  await network.provider.send('evm_setAutomine', [false])
  await network.provider.send('evm_setIntervalMining', [0])
  await Promise.all(promises)
  await network.provider.send('hardhat_setNextBlockBaseFeePerGas', ['0x0'])
  await network.provider.send('hardhat_mine', ['0x1'])
  await network.provider.send('evm_setAutomine', [true])
}

