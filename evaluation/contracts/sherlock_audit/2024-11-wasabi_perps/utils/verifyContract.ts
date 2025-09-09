import { run } from "hardhat"
import { Address } from "viem"

export async function verifyContract(contractAddress: Address, args?: any[]) {
    console.log("Verifying contract...")
    try {
        await run("verify:verify", {
            address: contractAddress,
            constructorArguments: args || [],
        })
    } catch (e) {
        if (typeof e === "string") {
            if (e.toLowerCase().includes("already verified")) {
                console.log("Already verified!");
            } else {
                console.error(e);
            }
        } else if (e instanceof Error) {
            if (e.message.toLowerCase().includes("already verified")) {
                console.log("Already verified!");
            } else {
                console.error(e);
            }
        } else {
            console.error(e);
        }
    }
}