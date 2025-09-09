import type { Address } from 'abitype'
import hre from "hardhat";
import { Hex, parseSignature, getAddress} from "viem";
import { ClosePositionRequest, OpenPositionRequest, ClosePositionOrder, AddCollateralRequest, RemoveCollateralRequest } from './PerpStructUtils';

export type Account = {
  address: Address;
}

export type Signer = {
  signTypedData: (data: EIP712SignatureParams<any>) => Promise<Hex>;
  account: Account;
}

export type Signature = {
  v: number;
  r: Hex;
  s: Hex;
}

type EIP712TypeField = {
  name: string;
  type: string;
}

export type EIP712Domain = {
    name: string;
    version: string;
    chainId: number;
    verifyingContract: Address;
}

type EIP712SignatureParams<T> = {
  account: Address;
  domain: EIP712Domain;
  primaryType: string;
  message: T;
  types: Record<string, EIP712TypeField[]>;
}

const EIP712DomainTypes: EIP712TypeField[] = [
  { name: "name", type: "string" },
  { name: "version", type: "string" },
  { name: "chainId", type: "uint256" },
  { name: "verifyingContract", type: "address" },
];

const FunctionCallDataTypes: EIP712TypeField[] = [
  { name: "to", type: "address" },
  { name: "value", type: "uint256" },
  { name: "data", type: "bytes" },
];

const OpenPositionRequestTypes: EIP712TypeField[] = [
  { name: "id", type: "uint256" },
  { name: "currency", type: "address" },
  { name: "targetCurrency", type: "address" },
  { name: "downPayment", type: "uint256" },
  { name: "principal", type: "uint256" },
  { name: "minTargetAmount", type: "uint256" },
  { name: "expiration", type: "uint256" },
  { name: "fee", type: "uint256" },
  { name: "functionCallDataList", type: "FunctionCallData[]" },
  { name: "existingPosition", type: "Position" },
  { name: "referrer", type: "address" },
];

const AddCollateralRequestTypes: EIP712TypeField[] = [
  { name: "amount", type: "uint256" },
  { name: "interest", type: "uint256" },
  { name: "expiration", type: "uint256" },
  { name: "position", type: "Position" },
];

const RemoveCollateralRequestTypes: EIP712TypeField[] = [
  { name: "amount", type: "uint256" },
  { name: "expiration", type: "uint256" },
  { name: "position", type: "Position" },
];

const PositionTypes: EIP712TypeField[] = [
  { name: "id", type: "uint256" },
  { name: "trader", type: "address" },
  { name: "currency", type: "address" },
  { name: "collateralCurrency", type: "address" },
  { name: "lastFundingTimestamp", type: "uint256" },
  { name: "downPayment", type: "uint256" },
  { name: "principal", type: "uint256" },
  { name: "collateralAmount", type: "uint256" },
  { name: "feesToBePaid", type: "uint256" },
];

const ClosePositionRequestTypes: EIP712TypeField[] = [
  { name: "expiration", type: "uint256" },
  { name: "interest", type: "uint256" },
  { name: "amount", type: "uint256" },
  { name: "position", type: "Position" },
  { name: "functionCallDataList", type: "FunctionCallData[]" },
  { name: "referrer", type: "address" },
];

const ClosePositionOrderTypes: EIP712TypeField[] = [
  { name: "orderType", type: "uint8" },
  { name: "positionId", type: "uint256" },
  { name: "createdAt", type: "uint256" },
  { name: "expiration", type: "uint256" },
  { name: "makerAmount", type: "uint256" },
  { name: "takerAmount", type: "uint256" },
  { name: "executionFee", type: "uint256" },
];

// const SwapRequestTypes: EIP712TypeField[] = [
//   { name: "currencyIn", type: "address" },
//   { name: "currencyOut", type: "address" },
//   { name: "amount", type: "uint256" },
//   { name: "exactIn", type: "bool" },
//   { name: "expiration", type: "uint256" },
//   { name: "price", type: "uint256" },
//   { name: "priceDenominator", type: "uint256" },
// ];

/**
 * Creates the EIP712 domain data for the given contract
 * @param name the contract name
 * @param verifyingContract the verifying contract address
 * @returns an EIP712 domain
 */
export function getDomainData(name: string, verifyingContract: Address): EIP712Domain {
  return {
      name,
      version: '1',
      chainId: hre.network.config.chainId!,
      verifyingContract: getAddress(verifyingContract),
  };
}

/**
 * Signs an OpenPositionRequest using EIP712
 * @param signer the signer
 * @param contractName the contract name
 * @param verifyingContract the verifying contract
 * @param request the request
 * @returns a signature object
 */
export async function signOpenPositionRequest(
  signer: Signer,
  contractName: string,
  verifyingContract: Address, 
  request: OpenPositionRequest
): Promise<Signature> {
  const domain = getDomainData(contractName, verifyingContract);
  const typeData: EIP712SignatureParams<OpenPositionRequest>  = {
    account: signer.account.address,
    types: {
      OpenPositionRequest: OpenPositionRequestTypes,
      FunctionCallData: FunctionCallDataTypes,
      Position: PositionTypes,
    },
    primaryType: "OpenPositionRequest",
    domain,
    message: request,
  };

  const signature = await signer.signTypedData(typeData);
  const signatureData = parseSignature(signature);
  return {
    v: Number(signatureData.v),
    r: signatureData.r,
    s: signatureData.s,
  };
}

export async function signAddCollateralRequest(
  signer: Signer,
  contractName: string,
  verifyingContract: Address,
  request: AddCollateralRequest
): Promise<Signature> {
  const domain = getDomainData(contractName, verifyingContract);
  const typeData: EIP712SignatureParams<AddCollateralRequest> = {
    account: signer.account.address,
    types: {
      AddCollateralRequest: AddCollateralRequestTypes,
      Position: PositionTypes,
    },
    primaryType: "AddCollateralRequest",
    domain,
    message: request,
  };

  const signature = await signer.signTypedData(typeData);
  const signatureData = parseSignature(signature);
  return {
    v: Number(signatureData.v),
    r: signatureData.r,
    s: signatureData.s,
  };
}

export async function signRemoveCollateralRequest(
  signer: Signer,
  contractName: string,
  verifyingContract: Address,
  request: RemoveCollateralRequest
): Promise<Signature> {
  const domain = getDomainData(contractName, verifyingContract);
  const typeData: EIP712SignatureParams<RemoveCollateralRequest> = {
    account: signer.account.address,
    types: {
      RemoveCollateralRequest: RemoveCollateralRequestTypes,
      Position: PositionTypes,
    },
    primaryType: "RemoveCollateralRequest",
    domain,
    message: request,
  };

  const signature = await signer.signTypedData(typeData);
  const signatureData = parseSignature(signature);
  return {
    v: Number(signatureData.v),
    r: signatureData.r,
    s: signatureData.s,
  };
}

export async function signClosePositionRequest(
  signer: Signer,
  contractName: string,
  verifyingContract: Address, 
  request: ClosePositionRequest
): Promise<Signature> {
  const domain = getDomainData(contractName, verifyingContract);
  const typeData: EIP712SignatureParams<ClosePositionRequest>  = {
    account: signer.account.address,
    types: {
      ClosePositionRequest: ClosePositionRequestTypes,
      Position: PositionTypes,
      FunctionCallData: FunctionCallDataTypes,
    },
    primaryType: "ClosePositionRequest",
    domain,
    message: request,
  };

  const signature = await signer.signTypedData(typeData);
  const signatureData = parseSignature(signature);
  return {
    v: Number(signatureData.v),
    r: signatureData.r,
    s: signatureData.s,
  };
}

export async function signClosePositionOrder(
  signer: Signer,
  contractName: string,
  verifyingContract: Address, 
  order: ClosePositionOrder
): Promise<Signature> {
  const domain = getDomainData(contractName, verifyingContract);
  const typeData: EIP712SignatureParams<ClosePositionOrder>  = {
    account: signer.account.address,
    types: {
      ClosePositionOrder: ClosePositionOrderTypes,
    },
    primaryType: "ClosePositionOrder",
    domain,
    message: order,
  };

  const signature = await signer.signTypedData(typeData);
  const signatureData = parseSignature(signature);
  return {
    v: Number(signatureData.v),
    r: signatureData.r,
    s: signatureData.s,
  };
}