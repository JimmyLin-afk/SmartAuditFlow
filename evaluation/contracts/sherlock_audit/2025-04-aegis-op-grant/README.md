# Aegis.im YUSD contest details

- Join [Sherlock Discord](https://discord.gg/MABEWyASkp)
- Submit findings using the **Issues** page in your private contest repo (label issues as **Medium** or **High**)
- [Read for more details](https://docs.sherlock.xyz/audits/watsons)

# Q&A

### Q: On what chains are the smart contracts going to be deployed?
• Ethereum mainnet (chainId = 1)  
• BNB Smart Chain mainnet (chainId = 56)
___

### Q: If you are integrating tokens, are you allowing only whitelisted tokens to work with the codebase or any complying with the standard? Are they assumed to have certain properties, e.g. be non-reentrant? Are there any types of [weird tokens](https://github.com/d-xo/weird-erc20) you want to integrate?
The system interacts only with whitelisted ERC-20 tokens.
All tokens must strictly comply with the ERC-20 standard and are assumed to behave normally (e.g., non-reentrant, standard transfer behavior).
No unusual token traits (such as reentrancy, fee-on-transfer, or deflationary mechanisms) are intended to be integrated.
Tokens with 6 to 18 decimals are acceptable.
___

### Q: Are there any limitations on values set by admins (or other roles) in the codebase, including restrictions on array lengths?
	•	Owner/Admin is trusted to set any values.
	•	There are no enforced backend limitations on values (e.g., no limits on array lengths, percentages, or amounts).
	•	No Keeper or FeeManager roles exist in the current version of the system.
___

### Q: Are there any limitations on values set by admins (or other roles) in protocols you integrate with, including restrictions on array lengths?
No.

___

### Q: Is the codebase expected to comply with any specific EIPs?
The codebase is expected to comply with EIP-20 (ERC-20 Token Standard) to ensure interoperability with existing DeFi protocols and wallets.
No other specific EIPs are targeted for compliance at this stage.

___

### Q: Are there any off-chain mechanisms involved in the protocol (e.g., keeper bots, arbitrage bots, etc.)? We assume these mechanisms will not misbehave, delay, or go offline unless otherwise specified.
Yes — Aegis.im operate an off-chain service that keeps the protocol delta-neutral by adjusting short BTC-perp positions settled through custodial vaults.
___

### Q: What properties/invariants do you want to hold even if breaking them has a low/unknown impact?
The following invariants must always hold:
• The total amount of underlying assets must always match or exceed the total user liabilities or issued tokens.
• Only approved (whitelisted) ERC-20 tokens must be accepted; no interactions with unapproved tokens should occur.
• Administrative privileges must remain correctly assigned and not be accidentally transferred or lost.
___

### Q: Please discuss any design choices you made.
•	We chose to strictly whitelist ERC-20 tokens to avoid interacting with tokens that may have non-standard behavior (such as fee-on-transfer, reentrancy risks, or non-standard decimals).
• Aegis.im operates an off-chain service to maintain delta neutrality and optimize protocol revenue, but this service is not critical for the security or core functionality of the on-chain contracts.
•	Administrative functions were kept minimal to reduce the potential attack surface and minimize governance overhead.
•	Gas optimizations were prioritized only where they did not compromise security, readability, or correctness.
___

### Q: Please provide links to previous audits (if any).
https://hacken.io/audits/aegis/
___

### Q: Please list any relevant protocol resources.

Core
🌐 Website — https://aegis.im/
📖 Docs — https://docs.aegis.im/
🛠️ dApp Dashboard — https://app.aegis.im/
🎯 Points Program — https://app.aegis.im/earn
🔍 Reserve Transparency — https://app.aegis.im/transparency

Contract Addresses:
💠 Ethereum:
🪙 YUSD —
https://etherscan.io/address/0x4274cD7277C7bb0806Bd5FE84b9aDAE466a8DA0a
🔄 Mint/Redeem Contract —https://etherscan.io/address/0xA30644CA67E0A93805c443Df4A6E1856d8Bd815B
🎁 Rewards Claim Contract —
https://etherscan.io/address/0x8aDCFAf1B64Cc514524B80565bCc732273dDEaFD

🔶 BNB:
🪙 YUSD — https://bscscan.com/address/0xAB3dBcD9B096C3fF76275038bf58eAC10D22C61f
🔄 Mint/Redeem Contract — https://bscscan.com/address/0x39dF2D423dF0BDDBA28f23C15c65a86554A2e141
🎁 Rewards Claim Contract —
https://bscscan.com/address/0x93eFAA2d2f6c3600d794233ed7E751d086E5B75E

Code & Development
💻 GitHub (org) — https://github.com/Aegis-im
📜 Smart Contracts — https://github.com/Aegis-im/aegis-contracts

Token & Data
📈 YUSD on CoinMarketCap - https://coinmarketcap.com/currencies/aegis-yusd/
📈 YUSD on CoinGecko — https://www.coingecko.com/en/coins/yusd-aegis
___

### Q: Additional audit information.
1.	Mint / Redeem:
• Check cap logic, rounding, and edge cases (e.g., tiny amounts, simultaneous mint and redeem).
• Verify collateral ratio can’t be bypassed.

2.	Rewards Distribution
• Review rewardsPerShare maths and claim flow for overflow, double-claim, or precision loss.
• Ensure that pausing or unpausing cannot lock or skip rewards.


# Audit scope

[aegis-contracts @ eaaf21ec7f3a9bf30a2aadd7118499b7bcf43681](https://github.com/Aegis-im/aegis-contracts/tree/eaaf21ec7f3a9bf30a2aadd7118499b7bcf43681)
- [aegis-contracts/contracts/AegisConfig.sol](aegis-contracts/contracts/AegisConfig.sol)
- [aegis-contracts/contracts/AegisMinting.sol](aegis-contracts/contracts/AegisMinting.sol)
- [aegis-contracts/contracts/AegisOracle.sol](aegis-contracts/contracts/AegisOracle.sol)
- [aegis-contracts/contracts/AegisRewards.sol](aegis-contracts/contracts/AegisRewards.sol)
- [aegis-contracts/contracts/YUSD.sol](aegis-contracts/contracts/YUSD.sol)
- [aegis-contracts/contracts/lib/ClaimRewardsLib.sol](aegis-contracts/contracts/lib/ClaimRewardsLib.sol)
- [aegis-contracts/contracts/lib/OrderLib.sol](aegis-contracts/contracts/lib/OrderLib.sol)


