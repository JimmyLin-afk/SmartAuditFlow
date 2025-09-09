// SPDX-License-Identifier: MIT
pragma solidity 0.8.19;

interface IPair {
    /// @dev gives the current twap price measured from amountIn * tokenIn gives amountOut
    function current(address tokenIn, uint256 amountIn) external view returns (uint256 amountOut);

    function getAmountOut(uint256 amountIn, address tokenIn) external view returns (uint256);

    function getReserves() external view returns (uint256 reserve0, uint256 reserve1, uint256 blockTimestampLast);
}
