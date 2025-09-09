// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

interface IMockSwap {
    error SwapReverted();

    event Swap(
        address currencyIn,
        uint256 amountIn,
        address currencyOut,
        uint256 amountOut
    );

    function swap(
        address currencyIn,
        uint256 amountIn,
        address currencyOut
    ) external payable returns (uint256);

    function swapExact(
        address currencyIn,
        uint256 amountIn,
        address currencyOut,
        uint256 amountOut
    ) external payable returns(uint256);

    function swapExactlyOut(
        address currencyIn,
        address currencyOut,
        uint256 amountOut
    ) external payable returns(uint256 amountIn);

    function revertingFunction() external payable;

    function setPrice(address token1, address token2, uint256 price) external;
}