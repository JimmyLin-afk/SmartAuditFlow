// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

import { IExchangeProvider } from "contracts/interfaces/IExchangeProvider.sol";
import { FixidityLib } from "celo/contracts/common/FixidityLib.sol";

contract MockExchangeProvider is IExchangeProvider {
  using FixidityLib for FixidityLib.Fraction;
  mapping(bytes32 => uint256) private exchangeRate;

  function setRate(bytes32 exchangeId, address base, address quote, uint256 rate) external {
    bytes32 rateId = keccak256(abi.encodePacked(exchangeId, base, quote));
    exchangeRate[rateId] = rate;
    rateId = keccak256(abi.encodePacked(exchangeId, quote, base));
    exchangeRate[rateId] = FixidityLib.fixed1().divide(FixidityLib.wrap(rate)).unwrap();
  }

  function getAmountOut(
    bytes32 exchangeId,
    address tokenIn,
    address tokenOut,
    uint256 amountIn
  ) external view returns (uint256 amountOut) {
    return _getAmountOut(exchangeId, tokenIn, tokenOut, amountIn);
  }

  function getAmountIn(
    bytes32 exchangeId,
    address tokenIn,
    address tokenOut,
    uint256 amountOut
  ) external view returns (uint256 amountIn) {
    return _getAmountIn(exchangeId, tokenIn, tokenOut, amountOut);
  }

  function swapIn(
    bytes32 exchangeId,
    address tokenIn,
    address tokenOut,
    uint256 amountIn
  ) external view returns (uint256 amountOut) {
    return _getAmountOut(exchangeId, tokenIn, tokenOut, amountIn);
  }

  function swapOut(
    bytes32 exchangeId,
    address tokenIn,
    address tokenOut,
    uint256 amountOut
  ) external view returns (uint256 amountIn) {
    return _getAmountIn(exchangeId, tokenIn, tokenOut, amountOut);
  }

  function _getAmountOut(
    bytes32 exchangeId,
    address tokenIn,
    address tokenOut,
    uint256 amountIn
  ) internal view returns (uint256 amountOut) {
    bytes32 rateId = keccak256(abi.encodePacked(exchangeId, tokenOut, tokenIn));
    return FixidityLib.newFixed(amountIn).multiply(FixidityLib.wrap(exchangeRate[rateId])).fromFixed();
  }

  function _getAmountIn(
    bytes32 exchangeId,
    address tokenIn,
    address tokenOut,
    uint256 amountOut
  ) internal view returns (uint256 amountIn) {
    bytes32 rateId = keccak256(abi.encodePacked(exchangeId, tokenIn, tokenOut));
    return FixidityLib.newFixed(amountOut).multiply(FixidityLib.wrap(exchangeRate[rateId])).fromFixed();
  }

  function getExchanges() external view returns (Exchange[] memory exchanges) {}
}
