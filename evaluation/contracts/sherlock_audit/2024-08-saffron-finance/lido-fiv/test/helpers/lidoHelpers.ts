import { Interface, parseEther } from 'ethers'
import { ethers } from 'hardhat'
import { HardhatEthersSigner } from '@nomicfoundation/hardhat-ethers/signers'

import WithdrawalQueueERC721ABI from '../abis/LidoWithdrawalQueueERC721ABI.json'
import LidoABI from '../abis/LidoABI.json'
import AccountOracleABI from '../abis/LidoAccountOracleABI.json'
import { impersonateAccounts, transferETH } from './testHelpers'

export const LIDO_CONTRACT_ADDRESS = {
  lido: '0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84',
  accountOracle: '0x852deD011285fe67063a08005c71a85690503Cee',
  withdrawalQueueERC721: '0x889edC2eDab5f40e902b864aD4d7AdE8E412F9B1',
  oracleReportSubmitter: '0x1d0813bf088BE3047d827D98524fBf779Bc25F00',
}

const errorsABI = [
  'error InsufficientBalance(uint256 available, uint256 required)',

  // WithdrawalQueue errors
  'error AdminZeroAddress()',
  'error RequestAmountTooSmall(uint256 _amountOfStETH)',
  'error RequestAmountTooLarge(uint256 _amountOfStETH)',
  'error InvalidReportTimestamp()',
  'error RequestIdsNotSorted()',
  'error ZeroRecipient()',
  'error ArraysLengthMismatch(uint256 _firstArrayLength, uint256 _secondArrayLength)',

  // WithdrawalQueueBase errors
  'error ZeroAmountOfETH()',
  'error ZeroShareRate()',
  'error ZeroTimestamp()',
  'error TooMuchEtherToFinalize(uint256 sent, uint256 maxExpected)',
  'error NotOwner(address _sender, address _owner)',
  'error InvalidRequestId(uint256 _requestId)',
  'error InvalidRequestIdRange(uint256 startId, uint256 endId)',
  'error InvalidState()',
  'error BatchesAreNotSorted()',
  'error EmptyBatches()',
  'error RequestNotFoundOrNotFinalized(uint256 _requestId)',
  'error NotEnoughEther()',
  'error RequestAlreadyClaimed(uint256 _requestId)',
  'error InvalidHint(uint256 _hint)',
  'error CantSendValueRecipientMayHaveReverted()',

  // AccountingOracle errors
  'error LidoLocatorCannotBeZero()',
  'error AdminCannotBeZero()',
  'error LegacyOracleCannotBeZero()',
  'error LidoCannotBeZero()',
  'error IncorrectOracleMigration(uint256 code)',
  'error SenderNotAllowed()',
  'error InvalidExitedValidatorsData()',
  'error UnsupportedExtraDataFormat(uint256 format)',
  'error UnsupportedExtraDataType(uint256 itemIndex, uint256 dataType)',
  'error CannotSubmitExtraDataBeforeMainData()',
  'error ExtraDataAlreadyProcessed()',
  'error ExtraDataListOnlySupportsSingleTx()',
  'error UnexpectedExtraDataHash(bytes32 consensusHash, bytes32 receivedHash)',
  'error UnexpectedExtraDataFormat(uint256 expectedFormat, uint256 receivedFormat)',
  'error ExtraDataItemsCountCannotBeZeroForNonEmptyData()',
  'error ExtraDataHashCannotBeZeroForNonEmptyData()',
  'error UnexpectedExtraDataItemsCount(uint256 expectedCount, uint256 receivedCount)',
  'error UnexpectedExtraDataIndex(uint256 expectedIndex, uint256 receivedIndex)',
  'error InvalidExtraDataItem(uint256 itemIndex)',
  'error InvalidExtraDataSortOrder(uint256 itemIndex)',

  // BaseOracle errors
  'error AddressCannotBeZero()',
  'error AddressCannotBeSame()',
  'error VersionCannotBeSame()',
  'error UnexpectedChainConfig()',
  'error SenderIsNotTheConsensusContract()',
  'error InitialRefSlotCannotBeLessThanProcessingOne(uint256 initialRefSlot, uint256 processingRefSlot)',
  'error RefSlotMustBeGreaterThanProcessingOne(uint256 refSlot, uint256 processingRefSlot)',
  'error RefSlotCannotDecrease(uint256 refSlot, uint256 prevRefSlot)',
  'error NoConsensusReportToProcess()',
  'error ProcessingDeadlineMissed(uint256 deadline)',
  'error RefSlotAlreadyProcessing()',
  'error UnexpectedRefSlot(uint256 consensusRefSlot, uint256 dataRefSlot)',
  'error UnexpectedConsensusVersion(uint256 expectedVersion, uint256 receivedVersion)',
  'error HashCannotBeZero()',
  'error UnexpectedDataHash(bytes32 consensusHash, bytes32 receivedHash)',
  'error SecondsPerSlotCannotBeZero()',

  // HashConsensus errors
  'error InvalidChainConfig()',
  'error NumericOverflow()',
  'error AdminCannotBeZero()',
  'error ReportProcessorCannotBeZero()',
  'error DuplicateMember()',
  'error AddressCannotBeZero()',
  'error InitialEpochIsYetToArrive()',
  'error InitialEpochAlreadyArrived()',
  'error InitialEpochRefSlotCannotBeEarlierThanProcessingSlot()',
  'error EpochsPerFrameCannotBeZero()',
  'error NonMember()',
  'error UnexpectedConsensusVersion(uint256 expected, uint256 received)',
  'error QuorumTooSmall(uint256 minQuorum, uint256 receivedQuorum)',
  'error InvalidSlot()',
  'error DuplicateReport()',
  'error EmptyReport()',
  'error StaleReport()',
  'error NonFastLaneMemberCannotReportWithinFastLaneInterval()',
  'error NewProcessorCannotBeTheSame()',
  'error ConsensusReportAlreadyProcessing()',
  'error FastLanePeriodCannotBeLongerThanFrame()',
]

const errorsABIInterface = new Interface(errorsABI)

export function decodeLidoErrorData(error: any) {
  return errorsABIInterface.parseError(error.data)
}

export function getWithdrawalQueueERC721Contract(signer: HardhatEthersSigner) {
  const withdrawalQueueERC721Contract = new ethers.Contract(
    LIDO_CONTRACT_ADDRESS.withdrawalQueueERC721,
    WithdrawalQueueERC721ABI,
    signer
  )

  return withdrawalQueueERC721Contract
}

const MAX_STETH_WITHDRAWAL_AMOUNT = BigInt('1000')

export async function finalizeLidoWithdrawalRequests(requestIds: BigInt[], ethAmount: bigint) {
  const BIG_INT_MAX = parseEther(`${Number.MAX_SAFE_INTEGER}`)

  await transferETH(LIDO_CONTRACT_ADDRESS.lido, ethAmount + BigInt(10))

  await impersonateAccounts([LIDO_CONTRACT_ADDRESS.lido])

  // Get signer for the impersonated account
  const signer = await ethers.getSigner(LIDO_CONTRACT_ADDRESS.lido)
  const lidoWithdrawalQueueContract = getWithdrawalQueueERC721Contract(signer)

  let amountLeft = ethAmount
  for (let i = 0; i < requestIds.length; i++) {
    let amount = amountLeft
    if (amountLeft >= MAX_STETH_WITHDRAWAL_AMOUNT) {
      amount = MAX_STETH_WITHDRAWAL_AMOUNT
      amountLeft -= MAX_STETH_WITHDRAWAL_AMOUNT
    } else {
      amount = amountLeft
      amountLeft = BigInt(0)
    }

    // use BIG_INT_MAX for max share rate - this util is not testing oracle calculated values
    const tx = await lidoWithdrawalQueueContract.finalize(requestIds[i], BIG_INT_MAX, {
      value: amount,
    })

    await tx.wait()
  }
}

export function getLidoContract(signer: HardhatEthersSigner) {
  const lidoContract = new ethers.Contract(LIDO_CONTRACT_ADDRESS.lido, LidoABI, signer)

  return lidoContract
}

export function getAccountOracleContract(signer: HardhatEthersSigner) {
  const accountOracleContract = new ethers.Contract(
    LIDO_CONTRACT_ADDRESS.accountOracle,
    AccountOracleABI,
    signer
  )

  return accountOracleContract
}

// submitReportData transaction
// https://etherscan.io/tx/0xd5d807d83b9adafeb6cf93e02c51e465024b86c1af29498e941256b1d7d60129
// submitReportData call right after hardhat configured fork block number 18562967

const SUBMIT_REPORT_DATA_ARGS = {
  consensusVersion: 1,
  refSlot: 7754399,
  numValidators: 289361,
  clBalanceGwei: BigInt('8952732133547390'),
  stakingModuleIdsWithNewlyExitedValidators: [1],
  numExitedValidatorsByStakingModule: [9954],
  withdrawalVaultBalance: BigInt('695700231270000000000'),
  elRewardsVaultBalance: BigInt('292371454327804792200'),
  sharesRequestedToBurn: 0,
  withdrawalFinalizationBatches: [15450],
  simulatedShareRate: BigInt('1146437846183007603074489632'),
  isBunkerMode: false,
  extraDataFormat: 1,
  extraDataHash: '0x7f7ff851294ee88f4adea8bc305ec926726cd65eff1a30db74f807ffe6df6d88',
  extraDataItemsCount: 1,
}

export async function submitOracleReport() {
  await impersonateAccounts([LIDO_CONTRACT_ADDRESS.oracleReportSubmitter])

  await transferETH(LIDO_CONTRACT_ADDRESS.oracleReportSubmitter, parseEther('1'))

  // Get signer for the impersonated account
  const oracleSigner = await ethers.getSigner(LIDO_CONTRACT_ADDRESS.oracleReportSubmitter)

  const accountOracleContract = await getAccountOracleContract(oracleSigner)

  const txSubmitReportData = await accountOracleContract.submitReportData(
    SUBMIT_REPORT_DATA_ARGS,
    SUBMIT_REPORT_DATA_ARGS.consensusVersion
  )

  await txSubmitReportData.wait()
}
