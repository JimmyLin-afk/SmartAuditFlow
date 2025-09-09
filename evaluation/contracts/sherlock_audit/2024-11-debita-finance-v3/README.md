
# Debita Finance V3 contest details

- Join [Sherlock Discord](https://discord.gg/MABEWyASkp)
- Submit findings using the issue page in your private contest repo (label issues as med or high)
- [Read for more details](https://docs.sherlock.xyz/audits/watsons)

# Q&A

### Q: On what chains are the smart contracts going to be deployed?
Sonic (Prev. Fantom), Base, Arbitrum & OP
___

### Q: If you are integrating tokens, are you allowing only whitelisted tokens to work with the codebase or any complying with the standard? Are they assumed to have certain properties, e.g. be non-reentrant? Are there any types of [weird tokens](https://github.com/d-xo/weird-erc20) you want to integrate?
We will interact with :

- any ERC20 that follows exactly the standard (eg. 18/6 decimals)
- Receipt tokens (All the interfaces from "contracts/Non-Fungible-Receipts/..")
- USDC and USDT
- Fee-on-transfer tokens will be used only in TaxTokensReceipt contract

___

### Q: Are there any limitations on values set by admins (or other roles) in the codebase, including restrictions on array lengths?
No
___

### Q: Are there any limitations on values set by admins (or other roles) in protocols you integrate with, including restrictions on array lengths?
  - DebitaV3Aggregator.sol::setValidNFTCollateral(address _collateral, bool status)
        * _collateral = Only receipts created by Debita

 -  DebitaV3Aggregator.sol::setOracleEnabled(address _oracle, bool status) 
    * _oracle = Only Oracles from debita (MixOracle.sol, DebitaChainlink.sol, DebitaPyth.sol)

-  DebitaPyth.sol::setPriceFeeds(address tokenAddress, bytes32 priceId)
     * priceId = priceID attached to the tokenAddress

 - DebitaChainlink.sol::setPriceFeeds(address _token, address _priceFeed)
    * _priceFeed = _priceFeed attached to the _token

 - MixOracle::setAttachedTarotPriceOracle(address uniswapV2Pair)
    * uniswapV2Pair = A pair where token0 is whitelisted on pyth oracle

___

### Q: For permissioned functions, please list all checks and requirements that will be made before calling the function.
   OwnershipsContract
     1. setDebitaContract
  
   AuctionFactoryDebitaContract
   1. setAggregator
   
   DLOFactoryContract
   1. setAggregatorContract

  DBOFactoryContract
   1. setAggregatorContract
   
   incentivesContract
    1. setAggregatorContract

   DebitaV3AggregatorContract
   1. setValidNFTCollateral (for all deployed receipts)
   
   
   


___

### Q: Is the codebase expected to comply with any EIPs? Can there be/are there any deviations from the specification?
None of the current scope is expected to comply with EIPs.


___

### Q: Are there any off-chain mechanisms for the protocol (keeper bots, arbitrage bots, etc.)? We assume they won't misbehave, delay, or go offline unless specified otherwise.
Each oracle will have a MANAGER role. We will have a bot constantly monitoring the price of pairs. If there is a difference greater than 5%, the oracle will be paused until it stabilizes again.

There will be a bot that will constantly call DebitaV3Aggregator.sol::matchOffersV3(), which will listen to the created borrow and lend orders. An important point is that initially we will be the providers of this service but anyone could create a bot or manually accept the orders.
___

### Q: If the codebase is to be deployed on an L2, what should be the behavior of the protocol in case of sequencer issues (if applicable)? Should Sherlock assume that the Sequencer won't misbehave, including going offline?
In the event that the sequencer is down, no additional loans should be created immediately with Chainlink Oracles. For Pyth, if no updates are received within 600 seconds, the transaction will also be reverted.
___

### Q: What properties/invariants do you want to hold even if breaking them has a low/unknown impact?
n/a
___

### Q: Please discuss any design choices you made.
- We have chosen to route all loans through the aggregator to maintain a consistent creation path. Borrowers will always need to create a borrow order, and lenders a lend order. The goal is to replicate the experience of a swap on a centralized exchange, where orders must be fully fulfilled.

- MixOracle.sol oracle - might not be the most precise way to get the price and may have a slippage in the price of +-5%. It will be only be used in edge cases where is the last resource.
___

### Q: Please list any known issues and explicitly state the acceptable risks for each known issue.
N/A
___

### Q: We will report issues where the core protocol functionality is inaccessible for at least 7 days. Would you like to override this value?
7 days is acceptable
___

### Q: Please provide links to previous audits (if any).
--
___

### Q: Please list any relevant protocol resources.
https://debita-finance.gitbook.io/debita-v3/overview/debita-v3
___



# Audit scope


[Debita-V3-Contracts @ bf92c2f839c086be957e3ed6a23b8c11111c7648](https://github.com/DebitaFinance/Debita-V3-Contracts/tree/bf92c2f839c086be957e3ed6a23b8c11111c7648)
- [Debita-V3-Contracts/contracts/DebitaBorrowOffer-Factory.sol](Debita-V3-Contracts/contracts/DebitaBorrowOffer-Factory.sol)
- [Debita-V3-Contracts/contracts/DebitaBorrowOffer-Implementation.sol](Debita-V3-Contracts/contracts/DebitaBorrowOffer-Implementation.sol)
- [Debita-V3-Contracts/contracts/DebitaIncentives.sol](Debita-V3-Contracts/contracts/DebitaIncentives.sol)
- [Debita-V3-Contracts/contracts/DebitaLendOffer-Implementation.sol](Debita-V3-Contracts/contracts/DebitaLendOffer-Implementation.sol)
- [Debita-V3-Contracts/contracts/DebitaLendOfferFactory.sol](Debita-V3-Contracts/contracts/DebitaLendOfferFactory.sol)
- [Debita-V3-Contracts/contracts/DebitaLoanOwnerships.sol](Debita-V3-Contracts/contracts/DebitaLoanOwnerships.sol)
- [Debita-V3-Contracts/contracts/DebitaV3Aggregator.sol](Debita-V3-Contracts/contracts/DebitaV3Aggregator.sol)
- [Debita-V3-Contracts/contracts/DebitaV3Loan.sol](Debita-V3-Contracts/contracts/DebitaV3Loan.sol)
- [Debita-V3-Contracts/contracts/auctions/Auction.sol](Debita-V3-Contracts/contracts/auctions/Auction.sol)
- [Debita-V3-Contracts/contracts/auctions/AuctionFactory.sol](Debita-V3-Contracts/contracts/auctions/AuctionFactory.sol)
- [Debita-V3-Contracts/contracts/oracles/DebitaChainlink.sol](Debita-V3-Contracts/contracts/oracles/DebitaChainlink.sol)

