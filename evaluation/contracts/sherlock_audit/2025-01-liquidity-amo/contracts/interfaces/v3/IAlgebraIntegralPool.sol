// SPDX-License-Identifier: GPL-2.0-or-later
pragma solidity >=0.5.0;

import "./IAlgebraPool.sol";

/// @title The interface for an Algebra Integral Pool
interface IAlgebraIntegralPool is IAlgebraPool {
    /// @notice Burn liquidity from the sender and account tokens owed for the liquidity to the position
    /// @dev Can be used to trigger a recalculation of fees owed to a position by calling with an amount of 0
    /// @dev Fees must be collected separately via a call to #collect
    /// @param bottomTick The lower tick of the position for which to burn liquidity
    /// @param topTick The upper tick of the position for which to burn liquidity
    /// @param amount How much liquidity to burn
    /// @param data Any data that should be passed through to the plugin
    /// @return amount0 The amount of token0 sent to the recipient
    /// @return amount1 The amount of token1 sent to the recipient
    function burn(
        int24 bottomTick,
        int24 topTick,
        uint128 amount,
        bytes calldata data
    ) external returns (uint256 amount0, uint256 amount1);

    /// @notice Returns the information about a position by the position's key
    /// @dev **important security note: caller should check reentrancy lock to prevent read-only reentrancy**
    /// @param key The position's key is a packed concatenation of the owner address, bottomTick and topTick indexes
    /// @return liquidity The amount of liquidity in the position
    /// @return innerFeeGrowth0Token Fee growth of token0 inside the tick range as of the last mint/burn/poke
    /// @return innerFeeGrowth1Token Fee growth of token1 inside the tick range as of the last mint/burn/poke
    /// @return fees0 The computed amount of token0 owed to the position as of the last mint/burn/poke
    /// @return fees1 The computed amount of token1 owed to the position as of the last mint/burn/poke
    function positions(
        bytes32 key
    )
        external
        view
        returns (
            uint256 liquidity,
            uint256 innerFeeGrowth0Token,
            uint256 innerFeeGrowth1Token,
            uint128 fees0,
            uint128 fees1
        );
}
