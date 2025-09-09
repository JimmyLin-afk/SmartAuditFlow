
# Hats Protocol contest details

- Join [Sherlock Discord](https://discord.gg/MABEWyASkp)
- Submit findings using the issue page in your private contest repo (label issues as med or high)
- [Read for more details](https://docs.sherlock.xyz/audits/watsons)

# Q&A

### Q: On what chains are the smart contracts going to be deployed?
Ethereum mainnet
Arbitrum
OP Mainnet
Base
Celo
Polygon POS
Gnosis Chain
Scroll
___

### Q: If you are integrating tokens, are you allowing only whitelisted tokens to work with the codebase or any complying with the standard? Are they assumed to have certain properties, e.g. be non-reentrant? Are there any types of [weird tokens](https://github.com/d-xo/weird-erc20) you want to integrate?
No tokens
___

### Q: Are there any limitations on values set by admins (or other roles) in the codebase, including restrictions on array lengths?
Owner is trusted
___

### Q: Are there any limitations on values set by admins (or other roles) in protocols you integrate with, including restrictions on array lengths?
No
___

### Q: Is the codebase expected to comply with any specific EIPs?
No
___

### Q: Are there any off-chain mechanisms involved in the protocol (e.g., keeper bots, arbitrage bots, etc.)? We assume these mechanisms will not misbehave, delay, or go offline unless otherwise specified.
No
___

### Q: What properties/invariants do you want to hold even if breaking them has a low/unknown impact?
The safe threshold should always be equal to lesser of the "enforced threshold" and the current number of safe owners
HSG should always be the guard of the safe (except when detaching itself)
HSG should always be enabled as a module of the safe (except when detaching itself)
There should never be more than 1 module enabled on the safe
No multisig transactions can be executed when the number of static signers is less than the min threshold
No multisig transactions can be executed when the number of static signers is less than the "enforced threshold"
No multisig transactions can be executed when the number of valid signers is less than the "enforced threshold"
Modules should never be able to change any values in Safe storage
The safe can never change its own number of static signers (unless it is explicitly authorized to do by being made the admin of one of the signerHats)
The safe can never change its own threshold
___

### Q: Please discuss any design choices you made.
Here is a doc (will also include in the repo) with an overview of our approach and some of our design choices, plus differences compared to v1: https://docs.google.com/document/d/1bq9yiuKaSl5qz0KatXxIDwNrkMr0J0dQluO46hiQTJU/edit?usp=sharing
___

### Q: Please provide links to previous audits (if any).
The following audits covered Hats Protocol and HatsSignerGate v1. Since HatsSignerGate v2 has significant changes compared to v1, findings listed here should not invalidate findings in the present contest. But I am including these past audits for reference as it may be helpful to Watsons.

https://github.com/Hats-Protocol/hats-zodiac/tree/main/audits
___

### Q: Please list any relevant protocol resources.
https://docs.hatsprotocol.xyz
https://hatsprotocol.xyz
___

### Q: Additional audit information.
This PR holds the diff between HatsSignerGate v1 and v2: https://github.com/Hats-Protocol/hats-zodiac/pull/45


___



# Audit scope


[hats-zodiac @ 8576776f45d31e1bfde26a72235b2b46a9028b24](https://github.com/Hats-Protocol/hats-zodiac/tree/8576776f45d31e1bfde26a72235b2b46a9028b24)
- [hats-zodiac/src/HatsSignerGate.sol](hats-zodiac/src/HatsSignerGate.sol)
- [hats-zodiac/src/interfaces/IHatsSignerGate.sol](hats-zodiac/src/interfaces/IHatsSignerGate.sol)
- [hats-zodiac/src/lib/SafeManagerLib.sol](hats-zodiac/src/lib/SafeManagerLib.sol)
- [hats-zodiac/src/lib/zodiac-modified/GuardableUnowned.sol](hats-zodiac/src/lib/zodiac-modified/GuardableUnowned.sol)
- [hats-zodiac/src/lib/zodiac-modified/ModifierUnowned.sol](hats-zodiac/src/lib/zodiac-modified/ModifierUnowned.sol)


