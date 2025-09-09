# Wasabi Perps

Wasabi Perps are asset-backed perpetual futures on Ethereum Mainnet, where users can long/short ERC-20 tokens with leverage.

A user wanting to open a position will deposit ETH, borrow ETH/ERC20 according to the leverage they select, and open a position. The total position size (user deposit + the loan principal) is swapped into assets backing the position. The swapped amount will be locked in the pool as collateral. The user can close their position at any given time for which the collateral will be sold, and the proceedings will be distributed as principal, interest, fees, and trader payout.

## Trading

A trader can open a position with leverage. They put up a downpayment and borrow funds from the LP pool. The interest for the loans will be calculated off-chain based on the utilization rate and other market factors.

### Long
For Long positions, the trader will borrow ETH. 

$LongPrincipal = (Leverage - 1) * DownPayment$

Then, the downpayment and the loan principal will be used to purchase ERC-20 tokens. These tokens will represent the position and be used as collateral for the loan.

### Short
For Short positions, the trader will borrow the ERC-20 tokens they are shorting.  

$ShortPrincipal = Leverage * DownPayment * MarkPrice$

These tokens will be sold to the market. The ETH received plus the downpayment will represent the position size and be used as collateral for the loan.

### Closing Positions
When a position is being closed, the collateral will be swapped to pay for principal, interest, and fees.

For Long Pools, the total collateral will be sold. The debt and the payout will be in ETH.

For the short pool, some part of the collateral will be used to purchase back the principal ERC20 tokens. The interest, fees and the payout will be paid in ETH.

### Liquidations

The protocol admin (owner) liquidates the positions if the collateral value drops below a certain threshold. The position will be closed, and the collateral will be sold to pay off the loan, interest, fee, and a refund to the trader (for the remaining amount, if any).

The liquidation transactions will only go through if the price is within the liquidation threshold.

## Interest Calculation
An off-chain server calculates interest based on the utilization rate of the pool. The more borrowed, the higher the interest will be for everyone.

A trader will get a signed close position request, including this interest value.

There is an on-chain maximum debt controller that limits the maximum interest a trader will pay.

## Mark Price
Wasabi Perps are asset backed and don't need to have a mark price in order to operate. 

However, an off-chain server needs to monitor the current price of the tokens, the index price, to liquidate positions and avoid bad debt. The off-chain server will monitor Uniswap pools of fractional NFT tokens and NFT marketplaces. This will create a more robust and much less manipulatable liquidation mechanic.

## Setting Up
After cloning the project, install the packages by running.

```shell
npm install
```

## Testing
Tests are created by using Hardhat. To run the unit tests, call the following:

```shell
npx hardhat test
```
