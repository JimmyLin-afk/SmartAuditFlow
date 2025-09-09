# Tally ARB Staker collaborative audit details

- Join [Sherlock Discord](https://discord.gg/MABEWyASkp)
- Submit findings using the issue page (label issues as `Medium`, `High`, `Low/Info` or `General Health`)
- [Read for more details](https://docs.sherlock.xyz/audits/watsons)

# Q&A

### Q: On what chains are the smart contracts going to be deployed?
Ethereum, Arbitrum, Rari Chain, zkSync Mainnet, Base, Polygon, OP Mainnet
___

### Q: If you are integrating tokens, are you allowing only whitelisted tokens to work with the codebase or any complying with the standard? Are they assumed to have certain properties, e.g. be non-reentrant? Are there any types of [weird tokens](https://github.com/d-xo/weird-erc20) you want to integrate?
Staker has uses tokens in two ways: staked tokens and reward tokens.

Both staked and reward tokens will be ARB
___

### Q: Are there any limitations on values set by admins (or other roles) in the codebase, including restrictions on array lengths?
Hardcoded variables:

FOR STAKER:
    IERC20 _rewardToken - ARB (https://arbiscan.io/address/0x912CE59144191C1204E64559FE8253a0e49E6548)
    IERC20 _stakeToken - ARB (https://arbiscan.io/address/0x912CE59144191C1204E64559FE8253a0e49E6548)
    IEarningPowerCalculator _earningPowerCalculator - TBD after deployment
    uint256 _maxBumpTip - tentatively 0.01 ARB
    address _admin - Arbitrum DAO Treasury Timelock (https://arbiscan.io/address/0xbFc1FECa8B09A5c5D3EFfE7429eBE24b9c09EF58)

FOR BinaryEligibilityOracleEarningPowerCalculator
    address _owner - Arbitrum DAO Treasury Timelock (https://arbiscan.io/address/0xbFc1FECa8B09A5c5D3EFfE7429eBE24b9c09EF58)
    address _scoreOracle - a Karma oracle, address TBD,
    uint256 _staleOracleWindow - tentatively 7 days
    address _oraclePauseGuardian - a trusted DAO multisig, address TBD
    uint256 _delegateeScoreEligibilityThreshold - tentatively 80
    uint256 _updateEligibilityDelay - 1 day
___

### Q: Are there any limitations on values set by admins (or other roles) in protocols you integrate with, including restrictions on array lengths?
The Staker contract trusts its admin.

The BinaryEligibilityOracleEarningPowerCalculator relies on an oracle.  The calculator should be resilient to oracle liveness and safety failures.
1. During a liveness failure, the EarningPowerCalculator  enters the failsafe state where earning power is based only on amount staked. This state happens after the `STALE_ORACLE_WINDOW`  has elapsed.
2. During a safety failure, the system trusts the pause guardian to put the system into the failsafe state
___

### Q: Is the codebase expected to comply with any specific EIPs?
No
___

### Q: Are there any off-chain mechanisms involved in the protocol (e.g., keeper bots, arbitrage bots, etc.)? We assume these mechanisms will not misbehave, delay, or go offline unless otherwise specified.
Yes:
- searcher bots can call `bumpEarningPower`  to change the earning power of a deposit if the EarningPowerCalculator changed their earning power. Anyone call `bumpEarningPower`
- an earning power oracle can upload earning power scores for each address
___

### Q: What properties/invariants do you want to hold even if breaking them has a low/unknown impact?
No
___

### Q: Please discuss any design choices you made.
Staker:
- We allow a user to claim even if their earning power is zero. This can remove the incentive for bumpers to bump up their earning power

Binary oracle:
- We fall back to stake amount rather than old earnining power.
- We allow a score threshold update to bypass the updateEligibility period
- We allow an eligibility period update to immediately take effect

___

### Q: Please provide links to previous audits (if any).
https://github.com/sherlock-protocol/sherlock-reports/blob/main/audits/2024.12.17%20-%20Final%20-%20Tally%20Collaborative%20Audit%20Report.pdf

Issues from this UniStaker are considered known and OOS, so they won't be validated https://github.com/code-423n4/2024-02-uniswap-foundation-findings/issues
___

### Q: Please list any relevant protocol resources.
https://docs.tally.xyz/knowledge-base/staking-on-tally
https://forum.arbitrum.foundation/c/proposals/7
https://docs.google.com/document/d/1Mu5HMsmnNWhjAiISfnQPznJnErIuyp1kGfAKuK4-GC4/edit?tab=t.0
___

### Q: Additional audit information.
Staker is based on UniStaker. 
- You can find UniStaker's audits here: https://github.com/uniswapfoundation/UniStaker/tree/main/audits
- You can find UniStaker's docs here: https://docs.unistaker.io/
___



# Audit scope


[staker @ fda09a661bbe1b5800fa72f52d6367de46740551](https://github.com/withtally/staker/tree/fda09a661bbe1b5800fa72f52d6367de46740551)
- [staker/src/BinaryEligibilityOracleEarningPowerCalculator.sol](staker/src/BinaryEligibilityOracleEarningPowerCalculator.sol)
- [staker/src/DelegationSurrogate.sol](staker/src/DelegationSurrogate.sol)
- [staker/src/DelegationSurrogateVotes.sol](staker/src/DelegationSurrogateVotes.sol)
- [staker/src/GovernanceStaker.sol](staker/src/GovernanceStaker.sol)
- [staker/src/extensions/GovernanceStakerDelegateSurrogateVotes.sol](staker/src/extensions/GovernanceStakerDelegateSurrogateVotes.sol)
- [staker/src/extensions/GovernanceStakerOnBehalf.sol](staker/src/extensions/GovernanceStakerOnBehalf.sol)
- [staker/src/extensions/GovernanceStakerPermitAndStake.sol](staker/src/extensions/GovernanceStakerPermitAndStake.sol)

