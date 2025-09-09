import { BigNumberish, Signer } from 'ethers'
import { ERC20PresetMinterPauser } from 'generated/typechain'
import { HardhatRuntimeEnvironment } from 'hardhat/types'
import { Address, TokenSymbol } from 'helpers/types'

import { getTokens } from '../config'

export interface SwapArgs {
  to: Address | Signer
  tokenSym: TokenSymbol
  amount: BigNumberish
  hre: HardhatRuntimeEnvironment
}

export const getFunds = async (args: SwapArgs): Promise<void> => {
  const { getNamedSigner, contracts } = args.hre

  const funder = await getNamedSigner('funder')
  const { all: tokenAddresses } = await getTokens(args.hre)

  const toAddress =
    typeof args.to === 'string' ? args.to : await args.to.getAddress()

  const token = await contracts.get<ERC20PresetMinterPauser>(
    'ERC20PresetMinterPauser',
    {
      at: tokenAddresses[args.tokenSym],
    }
  )
  await token.connect(funder).mint(toAddress, args.amount)
}
