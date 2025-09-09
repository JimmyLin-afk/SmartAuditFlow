// SPDX-License-Identifier: MIT
pragma solidity 0.8.19;

interface IFactory {
    function isPair(address pair) external view returns (bool);
    function getPair(address tokenA, address token, bool stable) external view returns (address);
    function createPair(address tokenA, address tokenB, bool stable) external returns (address pair);
}
