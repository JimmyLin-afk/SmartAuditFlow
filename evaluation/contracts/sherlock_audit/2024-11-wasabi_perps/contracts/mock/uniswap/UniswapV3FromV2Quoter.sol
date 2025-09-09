// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "./IQuoter.sol";
import "./IQuoterV2.sol";

contract UniswapV3FromV2Quoter is IQuoter {
    address immutable quoter;

    constructor(address _quoter) {
        quoter = _quoter;
    }

    function quoteExactInput(
        bytes memory path,
        uint256 amountIn
    ) external returns (uint256 amountOut) {
        (amountOut, , , ) = IQuoterV2(quoter).quoteExactInput(path, amountIn);
    }

    function quoteExactInputSingle(
        address tokenIn,
        address tokenOut,
        uint24 fee,
        uint256 amountIn,
        uint160 sqrtPriceLimitX96
    ) external returns (uint256 amountOut) {
        (amountOut, , , ) = IQuoterV2(quoter).quoteExactInputSingle(
            IQuoterV2.QuoteExactInputSingleParams({
                tokenIn: tokenIn,
                tokenOut: tokenOut,
                fee: fee,
                amountIn: amountIn,
                sqrtPriceLimitX96: sqrtPriceLimitX96
            })
        );
    }

    function quoteExactOutput(
        bytes memory path,
        uint256 amountOut
    ) external returns (uint256 amountIn) {
        (amountIn, , , ) = IQuoterV2(quoter).quoteExactOutput(path, amountOut);
    }

    function quoteExactOutputSingle(
        address tokenIn,
        address tokenOut,
        uint24 fee,
        uint256 amountOut,
        uint160 sqrtPriceLimitX96
    ) external returns (uint256 amountIn) {
        (amountIn, , , ) = IQuoterV2(quoter).quoteExactOutputSingle(
            IQuoterV2.QuoteExactOutputSingleParams({
                tokenIn: tokenIn,
                tokenOut: tokenOut,
                fee: fee,
                amount: amountOut,
                sqrtPriceLimitX96: sqrtPriceLimitX96
            })
        );
    }
}