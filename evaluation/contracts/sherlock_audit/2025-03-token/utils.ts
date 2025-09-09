import { ethers } from "ethers";

export function e(value: string | number) {
	return ethers.parseEther(value + "")
}
