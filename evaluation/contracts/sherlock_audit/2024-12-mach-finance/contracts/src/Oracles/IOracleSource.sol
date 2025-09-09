// SPDX-License-Identifier: BSD-3-Clause
pragma solidity 0.8.22;

interface IOracleSource {
    function getPrice(address token) external view returns (uint256 price, bool isValid);
}
