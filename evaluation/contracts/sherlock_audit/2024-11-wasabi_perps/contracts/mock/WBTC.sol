// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract WBTC is ERC20 {
    constructor() ERC20("Wrapped BTC", "WBTC") {}

    function decimals() public pure override returns (uint8) {
        return 8;
    }

    function mint(address _receiver, uint256 _amount) external {
        _mint(_receiver, _amount);
    }
}