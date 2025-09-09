
# Mento Update contest details

- Join [Sherlock Discord](https://discord.gg/MABEWyASkp)
- Submit findings using the issue page in your private contest repo (label issues as med or high)
- [Read for more details](https://docs.sherlock.xyz/audits/watsons)

# Q&A

### Q: On what chains are the smart contracts going to be deployed?
Celo
___

### Q: If you are integrating tokens, are you allowing only whitelisted tokens to work with the codebase or any complying with the standard? Are they assumed to have certain properties, e.g. be non-reentrant? Are there any types of [weird tokens](https://github.com/d-xo/weird-erc20) you want to integrate?
Only whitelisted tokens are used, standard ERC20 is implied. Most of them are protocol created tokens like cUSD, cEUR, cREAL, etc. but with this upgrade we're also integrating with Good$ so the Broker will also be able to mint/burn and interact with the existing Good$ Token https://celoscan.io/address/0x62B8B11039FcfE5aB0C56E502b1C372A3d2a9c7A
___

### Q: Are there any limitations on values set by admins (or other roles) in the codebase, including restrictions on array lengths?
- gooddollar/GoodDollarExchangeProvider::createExchange(PoolExchange _exchange):
  - _exchange.reserveRatio = Between 1 and 1e8
  - _exchange.exitContribution = Between 1 and 1e8

- gooddollar/GoodDollarExpansionController::setExpansionConfig(bytes32 _exchangeId, uint64 expansionRate, uint32 expansionFrequency)
  - expansionRate = Between 1 and 1e8
  - expansionFrequency > 0


___

### Q: Are there any limitations on values set by admins (or other roles) in protocols you integrate with, including restrictions on array lengths?
No
___

### Q: For permissioned functions, please list all checks and requirements that will be made before calling the function.
All permissioned calls either come from Mento governance or GoodDollar governance via the Avatar contract.
___

### Q: Is the codebase expected to comply with any EIPs? Can there be/are there any deviations from the specification?
No
___

### Q: Are there any off-chain mechanisms for the protocol (keeper bots, arbitrage bots, etc.)? We assume they won't misbehave, delay, or go offline unless specified otherwise.
No 
___

### Q: If the codebase is to be deployed on an L2, what should be the behavior of the protocol in case of sequencer issues (if applicable)? Should Sherlock assume that the Sequencer won't misbehave, including going offline?
Sherlock should assume that the Sequencer won't misbehave.
___

### Q: What properties/invariants do you want to hold even if breaking them has a low/unknown impact?
Bancor formula invariant. Price = Reserve / Supply * reserveRatio
___

### Q: Please discuss any design choices you made.
The bancor formula requires power fraction and that results in precision loss depending on where the system is. We've tested and ensured that precision tends to favour the protocol
___

### Q: Please list any known issues and explicitly state the acceptable risks for each known issue.
Same as the above, there are acceptable approximations due to power precision. We've tested to ensure that we're talking about ~1e6 precision on most cases and 1e3 loss on extreme scenarios.
___

### Q: We will report issues where the core protocol functionality is inaccessible for at least 7 days. Would you like to override this value?
It's ok. 
___

### Q: Please provide links to previous audits (if any).
https://docs.mento.org/mento/developers/smart-contracts/audits
___

### Q: Please list any relevant protocol resources.
https://docs.mento.org
___



# Audit scope


[mento-core @ 8722c6da3bb8a161996ca0b8fc2a4d0847b0916c](https://github.com/mento-protocol/mento-core/tree/8722c6da3bb8a161996ca0b8fc2a4d0847b0916c)
- [mento-core/contracts/common/IERC20MintableBurnable.sol](mento-core/contracts/common/IERC20MintableBurnable.sol)
- [mento-core/contracts/common/SafeERC20MintableBurnable.sol](mento-core/contracts/common/SafeERC20MintableBurnable.sol)
- [mento-core/contracts/goodDollar/BancorExchangeProvider.sol](mento-core/contracts/goodDollar/BancorExchangeProvider.sol)
- [mento-core/contracts/goodDollar/BancorFormula.sol](mento-core/contracts/goodDollar/BancorFormula.sol)
- [mento-core/contracts/goodDollar/GoodDollarExchangeProvider.sol](mento-core/contracts/goodDollar/GoodDollarExchangeProvider.sol)
- [mento-core/contracts/goodDollar/GoodDollarExpansionController.sol](mento-core/contracts/goodDollar/GoodDollarExpansionController.sol)
- [mento-core/contracts/interfaces/IBancorExchangeProvider.sol](mento-core/contracts/interfaces/IBancorExchangeProvider.sol)
- [mento-core/contracts/interfaces/IGoodDollarExchangeProvider.sol](mento-core/contracts/interfaces/IGoodDollarExchangeProvider.sol)
- [mento-core/contracts/interfaces/IGoodDollarExpansionController.sol](mento-core/contracts/interfaces/IGoodDollarExpansionController.sol)
- [mento-core/contracts/libraries/TradingLimits.sol](mento-core/contracts/libraries/TradingLimits.sol)
- [mento-core/contracts/swap/Broker.sol](mento-core/contracts/swap/Broker.sol)

