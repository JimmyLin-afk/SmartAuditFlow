# Automated Order System Contracts

This repository contains the smart contracts for an automated trading system, designed to execute orders as they come within range. There are two primary types of orders supported by the system: **Bracket Orders** and **Stop Limit Orders**.

## Automation

The Automation Master contract is designed to be monitored by Chainlink Automation-type systems. Anyone can fill any of the orders as long as the order is eligible and they provide the necessary assets to satisfy the slippage requirements. The token-out assets are sent to the user as part of the upkeep function.

## Order Types

### 1. Bracket Orders
A **Bracket Order** executes an automated swap when either the `takeProfit` or `stopPrice` conditions are met. The purpose of a Bracket Order is to allow traders to define both a profit target (`takeProfit`) and a stop loss (`stopPrice`) in a single transaction. The order is filled when either of these price conditions is reached, swapping the input token (`tokenIn`) for the output token (`tokenOut`).

- **`takeProfit`**: The execution price at which a profit target is reached.
- **`stopPrice`**: The price at which the order is closed to limit losses.

### 2. Stop Limit Orders
A **Stop Limit Order** is used to trigger the creation of a new Bracket Order when the `stopLimitPrice` condition is met. Once the stop limit price is reached, a Bracket Order is automatically created using the same unique `orderId` and parameters such as `takeProfit` and `stopPrice`. 

- **Shared Order ID**: Both the Stop Limit Order and the resulting Bracket Order share the same `orderId` for easy tracking and management.

### 3. Additional Order Types

By manipulating the `stopPrice` or the `takeProfit` in a **Bracket Order**, two more order types can be functionally replicated. 

1. **Limit Order**: By setting the `stopPrice` to 0, the system will have functionally created a standard **limit order**. This order type will only execute when the `takeProfit` is reached.
  
2. **Stop Loss Order**: By setting the `takeProfit` to the maximum possible value (`2 ** 256 - 1`), the system will have functionally created a **stop loss order**. This order type executes when the `stopPrice` is reached to minimize potential losses.

## Example Orders

For all examples, assume `WETH` price is `$3000`

### Bracket Order
1. User holds `1 WETH` and creates a **Bracket Order**, with a `takeProfit` set to `3200` and a `stopPrice` set to `2500`.
2. If either of these are reached, the user's `1 WETH` will be automaticly swapped to how ever much `USDC` can be bought at that price

### Take Profit Order
1. User holds `1 WETH` and creates a **Bracket Order**, with a `takeProfit` set to `3200` and a `stopPrice` set to `0`.
2. In this scenario, the user will never sell their `WETH` until the `takeProfit` is reached

### Stop Loss Order
1. User holds `1 WETH` and creates a **Bracket Order**, with a `takeProfit` set to `(2^256) - 1` and a `stopPrice` set to `2800`.
2. In this scenario, the user will hold their `WETH` until the price has dropped to the `stopPrice`, at which point they will sell for `USDC`

### Stop Limit Order
1. User holds `3000 USDC` and creates a **Stop Limit Order**  with a `stopLimitPrice` set to `2800`
2. Once this price is reached, the **Stop Limit Order** is filled, creating a new **Bracket Order**. This new **Bracket Order** will share the same `orderId` as the **Stop Limit Order**
3. Suppose this new **Bracket Order** has a `stopPrice` at `2500`, and `WETH` continues to fall to this price. 
4. Once this price is reached, the **Bracket Order** will be filled, and the user's `USDC` will be swapped to `WETH`

### Stop Limit Order with 'Swap-On-Fill'
1. User holds `2800 USDC` and creates a **Stop Limit Order**  with a `stopLimitPrice` set to `2800` and `swapOnFill` set to `true`
2. Once this price is reached, the **Stop Limit Order** is filled, swapping the `2800 USDC` for `1 WETH` and creating a new **Bracket Order**. This new **Bracket Order** will share the same `orderId` as the **Stop Limit Order**
3. Suppose this new **Bracket Order** has a `takeProfit` at `3000`, and `WETH` bounces back to this price. 
4. Once this price is reached, the **Bracket Order** will be filled, and the user's `1 WETH` will be swapped back to `3000 USDC`, and the user has profited ~`200 USDC`


## Order Creation

- **Bracket Order Creation**:
    ```solidity
    function createOrder(
        bytes calldata swapPayload,     // Optional data for executing a swap when the Stop Limit order is filled
        uint256 takeProfit,             // Price to trigger take-profit.
        uint256 stopPrice,              // Price to trigger stop-loss.
        uint256 amountIn,               // Amount of tokenIn to sell when conditions are met.
        IERC20 tokenIn,                 // Token to sell
        IERC20 tokenOut,                // Token to buy.
        address recipient,              // Address to receive tokenOut once the order is filled.
        uint16 takeProfitSlippage,      // Slippage tolerance for take-profit price, defined simply in basis points.
        uint16 stopSlippage,            // Slippage tolerance for stop-loss price, defined simply in basis points.
        bool permit,                    // Indicates whether Permit2 is used for token approvals.
        bytes calldata permitPayload    // Permit2 signature payload for approval-less token transfers.
    ) external;
    ```

- **Stop Limit Order Creation**:
    ```solidity
    function createOrder(
        uint256 stopLimitPrice,         // Price to trigger the Stop Limit order.
        uint256 takeProfit,             // Target price for the resulting Bracket Order to take profit.
        uint256 stopPrice,              // Stop-loss price for the resulting Bracket Order.
        uint256 amountIn,               // Amount of tokenIn to sell when conditions are met.
        IERC20 tokenIn,                 // Token to sell.
        IERC20 tokenOut,                // Token to buy.
        address recipient,              // Address to receive tokenOut once the order is filled.
        uint16 takeProfitSlippage,      // Slippage tolerance for the take-profit price in the Bracket Order.
        uint16 stopSlippage,            // Slippage tolerance for the stop-loss price in the Bracket Order.
        uint16 swapSlippage,            // Slippage tolerance for the initial swap when the Stop Limit order is filled.
        bool swapOnFill,                // Determines if the tokens should be swapped immediately after the Stop Limit order is filled.
        bool permit,                    // Indicates whether Permit2 is used for token approvals.
        bytes calldata permitPayload    // Permit2 signature payload for approval-less token transfers.
    ) external;
    ```

## Oracles

Oracles are expected to return a USD price in 1e8 terms, so the price of USDC should be returned as ~1e8 or ~`100000000`

## Testing

In order to run the tests, create a .env file and add a MAINNET_URL and ARB_URL and assign these to the appropriate RPC addresses. Here is an example .env file: 

```
MAINNET_URL="https://rpc.ankr.com/eth"
ARB_URL="https://rpc.ankr.com/arbitrum"
```
Then the tests can then be run by ```npm run test```
