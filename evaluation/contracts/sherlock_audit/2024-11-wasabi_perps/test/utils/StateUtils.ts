
import { PublicClient, getAddress, zeroAddress } from "viem";

import hre from "hardhat";
import { ERC20Abi } from "./ERC20Abi";

export type BalanceSnapshot = {
    get: (address: string) => bigint;
}

export async function getBalance(client: PublicClient, currency: string, address: string): Promise<bigint> {
    if (currency === zeroAddress) {
        return await client.getBalance({address: getAddress(address.toLowerCase())});
    } else {
        return await client.readContract({
            address: getAddress(currency),
            functionName: "balanceOf",
            args: [getAddress(address.toLowerCase())],
            abi: ERC20Abi,
        });
    }
}

export async function takeBalanceSnapshot(client: PublicClient, currency: string, ...address: string[]): Promise<BalanceSnapshot> {
    const balances = await Promise.all(address.map(getAddress).map(a => getBalance(client, currency, a)));
    const balanceMap = new Map<string, bigint>();
    for (let i = 0; i < address.length; i++) {
        balanceMap.set(getAddress(address[i]), balances[i]);
    }
    return {
        get: (address: string) => balanceMap.get(getAddress(address))!
    }
}