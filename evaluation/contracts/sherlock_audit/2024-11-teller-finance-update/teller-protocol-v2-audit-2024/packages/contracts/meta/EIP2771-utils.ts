import { JsonRpcProvider } from '@ethersproject/providers'
import { BigNumber, Contract, Wallet } from 'ethers'

export interface DomainData {
  contractName: string
  contractVersion: string
  contractAddress: string
  chainId: number
}

export interface EIP2771Request {
  from: string
  to: string
  value: BigNumber
  gas: BigNumber
  nonce: BigNumber
  data: string
}

export async function generateRequest(
  forwarder: Contract,
  provider: JsonRpcProvider,
  from: string,
  to: string,
  value: BigNumber,
  data: string
): Promise<EIP2771Request> {
  const gas = await provider.estimateGas({ to, from, data })
  const nonce = await forwarder.getNonce(from)

  const req: EIP2771Request = {
    from,
    to,
    value: value,
    gas: gas,
    nonce: nonce,
    data,
  }

  return req
}

export async function signMetatransaction(
  request: any,
  domainData: DomainData,
  privateKey: string
): Promise<string> {
  const domain = {
    name: domainData.contractName,
    version: domainData.contractVersion,
    chainId: domainData.chainId,
    verifyingContract: domainData.contractAddress,
  }

  // The named list of all type definitions
  const types = {
    ForwardRequest: [
      { name: 'from', type: 'address' },
      { name: 'to', type: 'address' },
      { name: 'value', type: 'uint256' },
      { name: 'gas', type: 'uint256' },
      { name: 'nonce', type: 'uint256' },
      { name: 'data', type: 'bytes' },
    ],
  }

  // The data to sign
  const value = request

  const wallet = new Wallet(privateKey)

  const signature = await wallet._signTypedData(domain, types, value)

  return signature
}
