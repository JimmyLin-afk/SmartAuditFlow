# IUniswapV2Router02
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/1dbcbcfe9a8bcf9392b2ada63dd8f1827a90783b/src/interfaces/periphery/IUniswapV2Router02.sol)

Interface for the Uniswap V2 Router

*https://github.com/Uniswap/v2-core/blob/master/contracts/interfaces/IUniswapV2Router02.sol*


## Functions
### swapExactTokensForTokens

Swaps `amountIn` of one token for as much as possible of another token


```solidity
function swapExactTokensForTokens(
    uint256 amountIn,
    uint256 amountOutMin,
    address[] calldata path,
    address to,
    uint256 deadline
) external returns (uint256[] memory amounts);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`amountIn`|`uint256`|The amount of token to swap|
|`amountOutMin`|`uint256`|The minimum amount of output that must be received|
|`path`|`address[]`|The ordered list of tokens to swap through|
|`to`|`address`|The recipient address|
|`deadline`|`uint256`||

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`amounts`|`uint256[]`|The amounts of the swapped tokens|


### swapTokensForExactTokens

Swaps as little as possible of one token for `amountOut` of another along the specified path (reversed)
that may remain in the router after the swap.


```solidity
function swapTokensForExactTokens(
    uint256 amountOut,
    uint256 amountInMax,
    address[] calldata path,
    address to,
    uint256 deadline
) external returns (uint256[] memory amounts);
```
**Parameters**

|Name|Type|Description|
|----|----|-----------|
|`amountOut`|`uint256`|The amount of token to receive|
|`amountInMax`|`uint256`|The maximum amount of token to swap|
|`path`|`address[]`|The ordered list of tokens to swap through|
|`to`|`address`|The recipient address|
|`deadline`|`uint256`||

**Returns**

|Name|Type|Description|
|----|----|-----------|
|`amounts`|`uint256[]`|The amounts of the swapped tokens|


