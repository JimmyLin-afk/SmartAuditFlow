// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

interface IExactOutSwapper {
    error NotWhitelistedSwapRouter(address target); // 0xc50efd89
    error NotWhitelistedFunctionSelector(bytes4 selector); // 0xd5ee6344
    error InsufficientAmountOutReceived(); // 0x545831f6
    error InsufficientTokenBalance(); // 0xe4455cae
    error InsufficientEthBalance(); // 0xb6d6e7d6

    /// @dev Defines a function call
    struct FunctionCallData {
        address to;
        uint256 value;
        bytes data;
    }

    // @dev Uses two exact input swaps to accomplish an exact output swap
    /// @param tokenIn The address of the input token
    /// @param tokenOut The address of the output token
    /// @param amountOut The amount of output tokens to receive
    /// @param amountInMax The maximum amount of input tokens to spend
    /// @param swapCallData The data for the first swap from tokenIn to tokenOut
    /// @param reverseCallData The data for the second swap, swapping excess tokenOut back to tokenIn
    /// @return amountIn The net amount of input tokens spent
    function swapExactOut(
        address tokenIn,
        address tokenOut,
        uint256 amountOut,
        uint256 amountInMax,
        FunctionCallData calldata swapCallData,
        FunctionCallData calldata reverseCallData
    ) external payable returns (uint256 amountIn);

    /// @dev Sets the whitelist status of a swap router
    /// @param swapRouter The address of the swap router
    /// @param isWhitelisted The whitelist status to set
    function setWhitelistedAddress(address swapRouter, bool isWhitelisted) external;

    /// @dev Sets the whitelist status of a list of function selectors
    /// @param selectors The function selectors to whitelist
    /// @param isWhitelisted The whitelist status to set
    function setWhitelistedFunctionSelectors(bytes4[] calldata selectors, bool isWhitelisted) external;

    /// @dev Recovers ERC20 tokens sent to this contract by mistake
    /// @param token The address of the token to recover
    /// @param to The address to send the recovered tokens to
    /// @param amount The amount of tokens to recover
    function recoverERC20(
        address token,
        address to,
        uint256 amount
    ) external;

    /// @dev Recovers ETH sent to this contract by mistake
    /// @param to The address to send the recovered ETH to
    /// @param amount The amount of ETH to recover
    function recoverETH(
        address to,
        uint256 amount
    ) external;
}