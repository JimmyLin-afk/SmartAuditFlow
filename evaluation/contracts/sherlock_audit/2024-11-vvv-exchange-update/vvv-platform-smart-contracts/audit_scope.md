## VVV Venture Capital Audit Scope

1. **Scope** 
	1. Venture Capital Investment and Claims (`contracts/vc/`)
		1.  `VVVVCInvestmentLedger.sol`
			1. Facilitates user investments in vVv's venture capital projects
		2.  `VVVVCTokenDistributor.sol`
			1. Facilitates users' token claims for vested tokens, based on prior investment
	3. nSLOC
		1. in `contract_metrics_for_audit.md`
2. **Smart Contract Requirements**
	1. Venture Capital Investment and Claims (`contracts/vc/`)
		1. `VVVVCInvestmentLedger.sol`
			1. Role-Restricted
				1. Can pause and unpause calls to the `invest` function via this contract
				2. Can manually add a record of investment contribution to an investment round
			2. User
				1. Can invest in an active investment round **only** when `investmentIsPaused` is `false`, within the contribution limits set in the signature supplied to the `invest` call, within the start and end times of the investment round, and when the input signature is valid, including the condition that it is not expired
				2. A fee (signature parameter defined off-chain) is only taken from the user-invested amount when the fee value is positive. The amount taken is that obtained via the fee calculation (`amount * feeNumerator / FEE_DENOMINATOR`), which has a precision of 1/10000, or 0.01%, since `FEE_DENOMINATOR` is set to 10,000. This level of precision is acceptable for the fee calculation.
				3. A user without the required role cannot access any functions gated via the `onlyAuthorized` modifier.
		3.  `VVVVCTokenDistributor.sol`
			1. Role-restricted
				1. Can pause and unpause calls to the `claim` function via this contract
			1. User
				1. Can claim tokens vested to project proxy wallets via the `claim` function if all checks are passed. Multiple claims of varying amounts from an array of wallets can be performed at once, as defined by the signature parameters `tokenAmountsToClaim` and `projectTokenProxyWallets`, which must be arrays of equal length. Each `claim` signature is assigned a `nonce` which cannot be reused and is incremented off-chain. The supplied signature must otherwise be valid and not expired. Upon successfully passing these checks, the specified amount from `tokenAmountsToClaim` is transferred from each wallet in `projectTokenProxyWallets` to `msg.sender`.
				2. All calls to `claim`, whether containing one or multiple wallets, involve only one ERC20 token address.
	2. Applies to All
		1. All role-restricted functions are gated via the `onlyAuthorized` modifier from `contracts/auth/VVVAuthorizationRegistryChecker.sol`, such that the calling address must possess the specified role for that particular function. Hence, no unauthorized user can call any function meant for role-restricted users.
		2. No aspect of any contract functionality is altered based on chain of deployment, whether Ethereum Mainnet, Avalanche C-Chain, Avalanche Evergreen Subnet, Base, or otherwise.
