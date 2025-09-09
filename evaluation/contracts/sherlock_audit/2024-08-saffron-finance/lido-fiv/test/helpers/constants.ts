import { parseEther } from 'ethers'

export const ZERO_ADDRESS = '0x0000000000000000000000000000000000000000'
export const ONE_ADDRESS = '0x0000000000000000000000000000000000000001'

export const DEFAULTS = {
  durationSeconds: 120,
  fixedSideCapacity: parseEther('1000'),
  fixedPremiumBps: 300,
  variableSideCapacity: parseEther('30'),
  earlyExitFeeBps: 1000,
  protocolFeeBps: 100,
  minimumDepositAmount: parseEther('0.01'),
  minimumFixedDepositBps: 500,
}

export const BIG_INT_ZERO = BigInt(0)
export const BIG_INT_ONE = BigInt(1)
export const BIG_INT_10K = BigInt(10_000) // useful for basis point calculations

export enum SIDE {
  FIXED = 0,
  VARIABLE = 1,
}
