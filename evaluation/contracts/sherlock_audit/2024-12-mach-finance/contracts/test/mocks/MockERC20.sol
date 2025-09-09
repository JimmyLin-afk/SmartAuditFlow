// SPDX-License-Identifier: BSD-3-Clause
pragma solidity 0.8.22;

import "@openzeppelin/contracts/mocks/token/ERC20Mock.sol";

contract MockERC20 is ERC20Mock {
    uint8 public _decimals;

    constructor(uint8 decimals_) ERC20Mock() {
        _decimals = decimals_;
    }

    function decimals() public view override returns (uint8) {
        return _decimals;
    }
}
