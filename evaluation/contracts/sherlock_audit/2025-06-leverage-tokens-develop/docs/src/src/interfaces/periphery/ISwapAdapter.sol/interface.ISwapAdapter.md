# ISwapAdapter
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/1dbcbcfe9a8bcf9392b2ada63dd8f1827a90783b/src/interfaces/periphery/ISwapAdapter.sol)


## Functions
### swapExactInput

Swap tokens from the `inputToken` to the `outputToken` using the specified provider

*The `outputToken` must be encoded in the `swapContext` path*


```solidity
function swapExactInput(IERC20 inputToken, uint256 inputAmount, uint256 minOutputAmount, SwapContext memory swapContext)
    external
    returns (uint256 outputAmount);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`inputToken`|`IERC20`|Token to swap from|
|`inputAmount`|`uint256`|Amount of tokens to swap|
|`minOutputAmount`|`uint256`|Minimum amount of tokens to receive|
|`swapContext`|`SwapContext`|Swap context to use for the swap (which exchange to use, the swap path, tick spacing, etc.)|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`outputAmount`|`uint256`|Amount of tokens received|


### swapExactOutput

Swap tokens from the `inputToken` to the `outputToken` using the specified provider

*The `outputToken` must be encoded in the `swapContext` path*


```solidity
function swapExactOutput(
    IERC20 inputToken,
    uint256 outputAmount,
    uint256 maxInputAmount,
    SwapContext memory swapContext
) external returns (uint256 inputAmount);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`inputToken`|`IERC20`|Token to swap from|
|`outputAmount`|`uint256`|Amount of tokens to receive|
|`maxInputAmount`|`uint256`|Maximum amount of tokens to swap|
|`swapContext`|`SwapContext`|Swap context to use for the swap (which exchange to use, the swap path, tick spacing, etc.)|

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`inputAmount`|`uint256`|Amount of tokens swapped|


## Errors
### InvalidNumTicks
Error thrown when the number of ticks is invalid


```solidity
error InvalidNumTicks();
```

### InvalidNumFees
Error thrown when the number of fees is invalid


```solidity
error InvalidNumFees();
```

## Structs
### EtherFiSwapContext
Contextual data required for EtherFi swaps using the EtherFi L2 Mode Sync Pool


```solidity
struct EtherFiSwapContext {
    IEtherFiL2ModeSyncPool etherFiL2ModeSyncPool;
    address tokenIn;
    address weETH;
    address referral;
}
```

### ExchangeAddresses
Addresses required to facilitate swaps on the supported exchanges


```solidity
struct ExchangeAddresses {
    address aerodromeRouter;
    address aerodromePoolFactory;
    address aerodromeSlipstreamRouter;
    address uniswapSwapRouter02;
    address uniswapV2Router02;
}
```

### SwapContext
Contextextual data required for a swap


```solidity
struct SwapContext {
    address[] path;
    bytes encodedPath;
    uint24[] fees;
    int24[] tickSpacing;
    Exchange exchange;
    ExchangeAddresses exchangeAddresses;
    bytes additionalData;
}
```

## Enums
### Exchange
The exchanges supported by SwapAdapter


```solidity
enum Exchange {
    AERODROME,
    AERODROME_SLIPSTREAM,
    ETHERFI,
    UNISWAP_V2,
    UNISWAP_V3
}
```

