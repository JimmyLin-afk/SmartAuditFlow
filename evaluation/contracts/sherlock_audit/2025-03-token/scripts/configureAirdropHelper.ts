import { ethers } from "hardhat"
import fs from "fs"
import { parse } from "csv-parse/sync"

async function main() {
	// Read and parse the CSV file
	const fileContent = fs.readFileSync("airdrop_checked_merged.csv", "utf-8")
	const records = parse(fileContent, {
		columns: true,
		skip_empty_lines: true,
	})

	let address = ""
	let contract = await ethers.getContractAt("AirdropHelper", address)

	// Start from index 2000
	const startIndex = 2000
	const batchSize = 1000

	let total = 0n
	// Process records in batches of 1000
	for (let i = startIndex; i < records.length; i += batchSize) {
		let airdropRecipients = []
		let airdropAmounts = []

		// Get the current batch
		const currentBatch = records.slice(i, i + batchSize)

		// Process records in current batch
		for (const record of currentBatch) {
			airdropRecipients.push(record.user)
			airdropAmounts.push(ethers.parseUnits(record.amount.toString(), 18))
			total += ethers.parseUnits(record.amount.toString(), 18)
		}

		console.log(
			`\nProcessing batch from index ${i} to ${i + currentBatch.length - 1}`,
		)
		console.log(`Number of recipients in this batch: ${currentBatch.length}`)

		try {
			// Send the transaction
			console.log("Sending transaction...")
			const tx = await contract.configureAirdrop(
				airdropRecipients,
				airdropAmounts,
			)

			// Wait for the transaction to be mined
			console.log("Waiting for transaction to be mined...")
			const receipt = await tx.wait()

			console.log(`Transaction successful! Tx hash: ${receipt?.hash}`)

			// Optional: Add some delay between batches to prevent rate limiting
			await new Promise(resolve => setTimeout(resolve, 5000))
		} catch (error) {
			console.error(`Error processing batch starting at index ${i}:`, error)
			// You might want to throw the error here to stop the script
			throw error
		}
	}
	console.log("Total Amount", total)
}

main()
	.then(() => process.exit(0))
	.catch(error => {
		console.error(error)
		process.exit(1)
	})
