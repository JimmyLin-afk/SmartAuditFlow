import { parseEther } from 'ethers'

import { LidoVault, MockLido, MockLidoWithdrawalQueue } from '../../typechain-types'
import { setCodeAtAddress, transferETH } from './testHelpers'

export async function setupMockLidoContracts(lidoVault: LidoVault) {
  const lidoAddress = await lidoVault.lido()
  const lidoWithdrawalQueueAddress = await lidoVault.lidoWithdrawalQueue()
  const mockLidoArtifact = require('../../artifacts/contracts/mocks/MockLido.sol/MockLido.json')
  const lidoMock: MockLido = (await setCodeAtAddress(
    'MockLido',
    lidoAddress,
    mockLidoArtifact.deployedBytecode
  )) as any
  const mockLidoWithdrawalQueueArtifact = require('../../artifacts/contracts/mocks/MockLidoWithdrawalQueue.sol/MockLidoWithdrawalQueue.json')
  const lidoWithdrawalQueueMock: MockLidoWithdrawalQueue = (await setCodeAtAddress(
    'MockLidoWithdrawalQueue',
    lidoWithdrawalQueueAddress,
    mockLidoWithdrawalQueueArtifact.deployedBytecode
  )) as any

  await lidoMock.initialize()
  await lidoWithdrawalQueueMock.initialize(lidoAddress)

  await transferETH(lidoAddress, parseEther('10'))
  // need decent amount of ETH for withdrawals
  await transferETH(lidoWithdrawalQueueAddress, parseEther('2000'))

  return { lidoMock, lidoWithdrawalQueueMock }
}
