# Crestal Network contest details

- Join [Sherlock Discord](https://discord.gg/MABEWyASkp)
- Submit findings using the **Issues** page in your private contest repo (label issues as **Medium** or **High**)
- [Read for more details](https://docs.sherlock.xyz/audits/watsons)

# Q&A

### Q: On what chains are the smart contracts going to be deployed?
Base
___

### Q: If you are integrating tokens, are you allowing only whitelisted tokens to work with the codebase or any complying with the standard? Are they assumed to have certain properties, e.g. be non-reentrant? Are there any types of [weird tokens](https://github.com/d-xo/weird-erc20) you want to integrate?
Only standard ERC-20 tokens
___

### Q: Are there any limitations on values set by admins (or other roles) in the codebase, including restrictions on array lengths?
Owner account is trusted

Fee collection address is also trusted

ERC-20 (payment) tokens set are trusted

Gaslass caller (forwarding gateway) is trusted
___

### Q: Are there any limitations on values set by admins (or other roles) in protocols you integrate with, including restrictions on array lengths?
No
___

### Q: Is the codebase expected to comply with any specific EIPs?
There is a gasless feature for many functions, that are meant to use EIP712 signatures to stay compliant with Biconomy's MEE-forwarded contract calls.
___

### Q: Are there any off-chain mechanisms involved in the protocol (e.g., keeper bots, arbitrage bots, etc.)? We assume these mechanisms will not misbehave, delay, or go offline unless otherwise specified.
No
___

### Q: What properties/invariants do you want to hold even if breaking them has a low/unknown impact?
No
___

### Q: Please provide links to previous audits (if any).
N/A
___

### Q: Please list any relevant protocol resources.
This is an overview of what we are doing as a platform: https://x.com/marouen19/status/1876214805926957473

This is a set of slightly outdated docs on how management contracts are being used (ignore the solver terminology) - worker decentralization still applies: https://docs.crestal.network/solvers/worker. Ignore all other pages related to IntentKit or solver or Testnet guide.

Overall direction: https://crestal.network/

Live app on Base using a slightly older version of the contracts: https://nation.fun/
___

### Q: Additional audit information.
No forked contracts. Only external dependencies are standard OZ contracts.

Would like you to look into:
1. Upgradeable integrations of contracts (any loopholes for upgrading)
2. Any function that accepts payments - any non-standard or exploitable calls
3. General function usage of variables - potential passing of unexpected values or getting states stuck at certain values


# Audit scope

[crestal-omni-contracts @ dc45e98af5e247dce5bbe53b0bd5b1f256884f84](https://github.com/crestalnetwork/crestal-omni-contracts/tree/dc45e98af5e247dce5bbe53b0bd5b1f256884f84)
- [crestal-omni-contracts/src/Blueprint.sol](crestal-omni-contracts/src/Blueprint.sol)
- [crestal-omni-contracts/src/BlueprintCore.sol](crestal-omni-contracts/src/BlueprintCore.sol)
- [crestal-omni-contracts/src/BlueprintV5.sol](crestal-omni-contracts/src/BlueprintV5.sol)
- [crestal-omni-contracts/src/EIP712.sol](crestal-omni-contracts/src/EIP712.sol)
- [crestal-omni-contracts/src/Payment.sol](crestal-omni-contracts/src/Payment.sol)


