
# Cork Protocol contest details

- Join [Sherlock Discord](https://discord.gg/MABEWyASkp)
- Submit findings using the issue page in your private contest repo (label issues as med or high)
- [Read for more details](https://docs.sherlock.xyz/audits/watsons)

# Q&A

### Q: On what chains are the smart contracts going to be deployed?
The contracts will be primarily deployed on Ethereum.
___

### Q: If you are integrating tokens, are you allowing only whitelisted tokens to work with the codebase or any complying with the standard? Are they assumed to have certain properties, e.g. be non-reentrant? Are there any types of [weird tokens](https://github.com/d-xo/weird-erc20) you want to integrate?
Yes, the Redemption Asset and Pegged Asset will only work after admin initialise pair through config contract (currently token whitelisting is enabled)
Standard ERC-20 tokens with 18 decimals are supported
Rebasing tokens are supported with exchange rate mechanism through Asset contracts

Fee-on-transfer and non-18 decimals tokens are NOT supported now but will support them in future versions subject to additional security review
___

### Q: Are there any limitations on values set by admins (or other roles) in the codebase, including restrictions on array lengths?
Yes, RepurchaseFeeRate, EarlyRedemptionFeeRate, PsmBaseRedemptionFeePrecentage will be between 0 to 5% (0% <= x <= 5%)
___

### Q: Are there any limitations on values set by admins (or other roles) in protocols you integrate with, including restrictions on array lengths?
There are currently no limitations. 
___

### Q: For permissioned functions, please list all checks and requirements that will be made before calling the function.
For all permissioned functions in corkConfig contract, modifier will ensure that caller must have MANAGER or DEFAULT_ADMIN role
___

### Q: Is the codebase expected to comply with any EIPs? Can there be/are there any deviations from the specification?
Our internal Asset.sol contracts should be ERC20 and ERC20Permit strictly compliant
___

### Q: Are there any off-chain mechanisms or off-chain procedures for the protocol (keeper bots, arbitrage bots, etc.)?
No - there will likely be external arbitrage bots operating around our system but they are not within scope and act as any market participant
___

### Q: Are there any hardcoded values that you intend to change before (some) deployments?
No
___

### Q: If the codebase is to be deployed on an L2, what should be the behavior of the protocol in case of sequencer issues (if applicable)? Should Sherlock assume that the Sequencer won't misbehave, including going offline?
N/A
___

### Q: Should potential issues, like broken assumptions about function behavior, be reported if they could pose risks in future integrations, even if they might not be an issue in the context of the scope? If yes, can you elaborate on properties/invariants that should hold?
Any potential misbehaviour regarding DS/CT/FlashSwap mechanism that is not an issue right now but could pose risk in future integrations should be counted, but reported as Low issues.
___

### Q: Please discuss any design choices you made.
Design decisions are discussed in the litepaper: https://corkfi.notion.site/Depeg-Swaps-Litepaper-f21a57d5c19d48209dfa0f0c2ab776c4
___

### Q: Please list any known issues and explicitly state the acceptable risks for each known issue.
Admin will be multisig contract and every multisig transaction from admin multisig will be reviewed by trusted team members, so here we can  assume that multisig owner contract will not act maliciously 
___

### Q: We will report issues where the core protocol functionality is inaccessible for at least 7 days. Would you like to override this value?
N/A
___

### Q: Please provide links to previous audits (if any).
N/A - This is the first audit
___

### Q: Please list any relevant protocol resources.
https://corkfi.notion.site/Depeg-Swaps-Litepaper-f21a57d5c19d48209dfa0f0c2ab776c4
___

### Q: Additional audit information.
N/A
___



# Audit scope


[Depeg-swap @ d4fcdd524deb5d07a7ccfd6eb49ba5157ed42642](https://github.com/Cork-Technology/Depeg-swap/tree/d4fcdd524deb5d07a7ccfd6eb49ba5157ed42642)
- [Depeg-swap/contracts/core/CorkConfig.sol](Depeg-swap/contracts/core/CorkConfig.sol)
- [Depeg-swap/contracts/core/ModuleCore.sol](Depeg-swap/contracts/core/ModuleCore.sol)
- [Depeg-swap/contracts/core/ModuleState.sol](Depeg-swap/contracts/core/ModuleState.sol)
- [Depeg-swap/contracts/core/Psm.sol](Depeg-swap/contracts/core/Psm.sol)
- [Depeg-swap/contracts/core/Vault.sol](Depeg-swap/contracts/core/Vault.sol)
- [Depeg-swap/contracts/core/assets/Asset.sol](Depeg-swap/contracts/core/assets/Asset.sol)
- [Depeg-swap/contracts/core/assets/AssetFactory.sol](Depeg-swap/contracts/core/assets/AssetFactory.sol)
- [Depeg-swap/contracts/core/flash-swaps/FlashSwapRouter.sol](Depeg-swap/contracts/core/flash-swaps/FlashSwapRouter.sol)
- [Depeg-swap/contracts/libraries/DepegSwapLib.sol](Depeg-swap/contracts/libraries/DepegSwapLib.sol)
- [Depeg-swap/contracts/libraries/DsFlashSwap.sol](Depeg-swap/contracts/libraries/DsFlashSwap.sol)
- [Depeg-swap/contracts/libraries/DsSwapperMathLib.sol](Depeg-swap/contracts/libraries/DsSwapperMathLib.sol)
- [Depeg-swap/contracts/libraries/Guard.sol](Depeg-swap/contracts/libraries/Guard.sol)
- [Depeg-swap/contracts/libraries/LvAssetLib.sol](Depeg-swap/contracts/libraries/LvAssetLib.sol)
- [Depeg-swap/contracts/libraries/MathHelper.sol](Depeg-swap/contracts/libraries/MathHelper.sol)
- [Depeg-swap/contracts/libraries/MutexLock.sol](Depeg-swap/contracts/libraries/MutexLock.sol)
- [Depeg-swap/contracts/libraries/Pair.sol](Depeg-swap/contracts/libraries/Pair.sol)
- [Depeg-swap/contracts/libraries/PeggedAssetLib.sol](Depeg-swap/contracts/libraries/PeggedAssetLib.sol)
- [Depeg-swap/contracts/libraries/PermitChecker.sol](Depeg-swap/contracts/libraries/PermitChecker.sol)
- [Depeg-swap/contracts/libraries/PsmLib.sol](Depeg-swap/contracts/libraries/PsmLib.sol)
- [Depeg-swap/contracts/libraries/RedemptionAssetManagerLib.sol](Depeg-swap/contracts/libraries/RedemptionAssetManagerLib.sol)
- [Depeg-swap/contracts/libraries/SignatureHelperLib.sol](Depeg-swap/contracts/libraries/SignatureHelperLib.sol)
- [Depeg-swap/contracts/libraries/State.sol](Depeg-swap/contracts/libraries/State.sol)
- [Depeg-swap/contracts/libraries/UQ112x112.sol](Depeg-swap/contracts/libraries/UQ112x112.sol)
- [Depeg-swap/contracts/libraries/VaultConfig.sol](Depeg-swap/contracts/libraries/VaultConfig.sol)
- [Depeg-swap/contracts/libraries/VaultLib.sol](Depeg-swap/contracts/libraries/VaultLib.sol)
- [Depeg-swap/contracts/libraries/VaultPoolLib.sol](Depeg-swap/contracts/libraries/VaultPoolLib.sol)

