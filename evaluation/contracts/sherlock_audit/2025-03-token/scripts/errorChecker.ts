import { ethers } from "hardhat"

const abi = []

// Create an Interface instance using the ABI
const iface = new ethers.Interface(abi)

// Your error data (for instance, from a reverted transaction)
const errorData = ""

try {
	// Attempt to parse the error data using the ABI
	const parsedError = iface.parseError(errorData)!
	console.log("Error Name:", parsedError.name)
	console.log("Error Arguments:", parsedError.args)
} catch (error) {
	console.error("Error parsing the data:", error)
}
