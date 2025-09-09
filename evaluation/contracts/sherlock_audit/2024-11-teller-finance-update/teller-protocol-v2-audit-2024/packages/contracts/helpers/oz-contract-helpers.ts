import { ethers } from 'hardhat'

export const isInitialized = async (address: string): Promise<boolean> => {
  const storage = await ethers.provider.getStorage(address, 0)
  return BigInt(storage) !== 0n
}
