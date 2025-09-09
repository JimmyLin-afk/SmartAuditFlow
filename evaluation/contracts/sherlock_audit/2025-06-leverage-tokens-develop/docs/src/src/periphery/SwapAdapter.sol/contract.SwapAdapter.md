# SwapAdapter
[Git Source](https://github.com/seamless-protocol/ilm-v2/blob/1dbcbcfe9a8bcf9392b2ada63dd8f1827a90783b/src/periphery/SwapAdapter.sol)

**Inherits:**
[ISwapAdapter](/src/interfaces/periphery/ISwapAdapter.sol/interface.ISwapAdapter.md)

*The SwapAdapter contract is a periphery contract that facilitates the use of various DEXes for swaps.*


## Functions
### swapExactInput

Swap tokens from the `inputToken` to the `outputToken` using the specified provider

*The `outputToken` must be encoded in the `swapContext` path*


```solidity
function swapExactInput(IERC20 inputToken, uint256 inputAmount, uint256 minOutputAmount, SwapContext memory swapContext)
    external
    returns (uint256);
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
|`<none>`|`uint256`|outputAmount Amount of tokens received|


### swapExactOutput

Swap tokens from the `inputToken` to the `outputToken` using the specified provider

*The `outputToken` must be encoded in the `swapContext` path*


```solidity
function swapExactOutput(
    IERC20 inputToken,
    uint256 outputAmount,
    uint256 maxInputAmount,
    SwapContext memory swapContext
) external returns (uint256);
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
|`<none>`|`uint256`|inputAmount Amount of tokens swapped|


### _swapAerodrome


```solidity
function _swapAerodrome(
    uint256 inputAmount,
    uint256 minOutputAmount,
    address receiver,
    address aerodromeRouter,
    address aerodromePoolFactory,
    address[] memory path
) internal returns (uint256 outputAmount);
```

### _swapExactInputAerodrome


```solidity
function _swapExactInputAerodrome(uint256 inputAmount, uint256 minOutputAmount, SwapContext memory swapContext)
    internal
    returns (uint256 outputAmount);
```

### _swapExactInputAerodromeSlipstream


```solidity
function _swapExactInputAerodromeSlipstream(
    uint256 inputAmount,
    uint256 minOutputAmount,
    SwapContext memory swapContext
) internal returns (uint256 outputAmount);
```

### _swapExactInputEtherFi


```solidity
function _swapExactInputEtherFi(
    IERC20 inputToken,
    uint256 inputAmount,
    uint256 minAmountOut,
    SwapContext memory swapContext
) internal returns (uint256 outputAmount);
```

### _swapExactInputUniV2


```solidity
function _swapExactInputUniV2(uint256 inputAmount, uint256 minOutputAmount, SwapContext memory swapContext)
    internal
    returns (uint256 outputAmount);
```

### _swapExactInputUniV3


```solidity
function _swapExactInputUniV3(uint256 inputAmount, uint256 minOutputAmount, SwapContext memory swapContext)
    internal
    returns (uint256 outputAmount);
```

### _swapExactOutputAerodrome


```solidity
function _swapExactOutputAerodrome(uint256 outputAmount, uint256 maxInputAmount, SwapContext memory swapContext)
    internal
    returns (uint256 inputAmount);
```

### _swapExactOutputAerodromeSlipstream


```solidity
function _swapExactOutputAerodromeSlipstream(
    uint256 outputAmount,
    uint256 maxInputAmount,
    SwapContext memory swapContext
) internal returns (uint256 inputAmount);
```

### _swapExactOutputUniV2


```solidity
function _swapExactOutputUniV2(uint256 outputAmount, uint256 maxInputAmount, SwapContext memory swapContext)
    internal
    returns (uint256 inputAmount);
```

### _swapExactOutputUniV3


```solidity
function _swapExactOutputUniV3(uint256 outputAmount, uint256 maxInputAmount, SwapContext memory swapContext)
    internal
    returns (uint256 inputAmount);
```

### _generateAerodromeRoutes

Generate the array of Routes as required by the Aerodrome router


```solidity
function _generateAerodromeRoutes(address[] memory path, address aerodromePoolFactory)
    internal
    pure
    returns (IAerodromeRouter.Route[] memory routes);
```

### _reversePath

Reverses a path of addresses


```solidity
function _reversePath(address[] memory path) internal pure returns (address[] memory reversedPath);
```

### receive


```solidity
receive() external payable;
```

