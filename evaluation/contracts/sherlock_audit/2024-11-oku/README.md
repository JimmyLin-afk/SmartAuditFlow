
# Oku contest details

- Join [Sherlock Discord](https://discord.gg/MABEWyASkp)
- Submit findings using the issue page in your private contest repo (label issues as med or high)
- [Read for more details](https://docs.sherlock.xyz/audits/watsons)

# Q&A

### Q: On what chains are the smart contracts going to be deployed?
Optimism

___

### Q: If you are integrating tokens, are you allowing only whitelisted tokens to work with the codebase or any complying with the standard? Are they assumed to have certain properties, e.g. be non-reentrant? Are there any types of [weird tokens](https://github.com/d-xo/weird-erc20) you want to integrate?
The owner can manage which tokens can utilize the contract's functionality to prevent non-standard and incompatible tokens from being used. 
USDT should be considered in-scope of this audit
___

### Q: Are there any limitations on values set by admins (or other roles) in the codebase, including restrictions on array lengths?
The only privileged role is the owner.

The owner can:
Withdraw fees from the AutomationMaster
Register oracles 
Set the min order size
Set the max lending orders
Register new sub keeper implementations
___

### Q: Are there any limitations on values set by admins (or other roles) in protocols you integrate with, including restrictions on array lengths?
No 
___

### Q: Is the codebase expected to comply with any specific EIPs?
No specific EIP
___

### Q: Are there any off-chain mechanisms involved in the protocol (e.g., keeper bots, arbitrage bots, etc.)? We assume these mechanisms will not misbehave, delay, or go offline unless otherwise specified.
There will be an off chain mechanism for executing transactions in an automated way. This system has no privileged access to the contracts, and anyone can perform it's duties if desired.
___

### Q: What properties/invariants do you want to hold even if breaking them has a low/unknown impact?
No
___

### Q: Please discuss any design choices you made.
We chose to use a dynamic array to track pending orders by their IDs, utilizing array mutation logic to remove specific indexes from the array as orders are filled. Because the checkUpkeep function must loop through all pending orders, a mapping is not ideal. While this implementation is not the most efficient, we like it's readability, and as this system will primarily operate on later 2, the increased gas usage is not an issue for us.
___

### Q: Please provide links to previous audits (if any).
None
___

### Q: Please list any relevant protocol resources.
None
___

### Q: Additional audit information.
Exchange rate
MinAmountReceived calculation
Array mutation
___



# Audit scope


[oku-custom-order-types @ b84e5725f4d1e0a1ee9048baf44e68d2e53ec971](https://github.com/gfx-labs/oku-custom-order-types/tree/b84e5725f4d1e0a1ee9048baf44e68d2e53ec971)
- [oku-custom-order-types/contracts/automatedTrigger/AutomationMaster.sol](oku-custom-order-types/contracts/automatedTrigger/AutomationMaster.sol)
- [oku-custom-order-types/contracts/automatedTrigger/Bracket.sol](oku-custom-order-types/contracts/automatedTrigger/Bracket.sol)
- [oku-custom-order-types/contracts/automatedTrigger/IAutomation.sol](oku-custom-order-types/contracts/automatedTrigger/IAutomation.sol)
- [oku-custom-order-types/contracts/automatedTrigger/OracleLess.sol](oku-custom-order-types/contracts/automatedTrigger/OracleLess.sol)
- [oku-custom-order-types/contracts/automatedTrigger/StopLimit.sol](oku-custom-order-types/contracts/automatedTrigger/StopLimit.sol)
- [oku-custom-order-types/contracts/libraries/ArrayMutation.sol](oku-custom-order-types/contracts/libraries/ArrayMutation.sol)
- [oku-custom-order-types/contracts/oracle/External/OracleRelay.sol](oku-custom-order-types/contracts/oracle/External/OracleRelay.sol)
- [oku-custom-order-types/contracts/oracle/External/PythOracle.sol](oku-custom-order-types/contracts/oracle/External/PythOracle.sol)


