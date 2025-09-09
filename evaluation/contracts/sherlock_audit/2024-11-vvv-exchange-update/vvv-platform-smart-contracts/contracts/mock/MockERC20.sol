//SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract MockERC20 is ERC20 {
    uint8 _decimals;

    constructor(uint256 decimals_) ERC20("MockERC20", "MOCK") {
        _decimals = uint8(decimals_);
        uint256 initialSupply = 1000000000 * 10 ** _decimals;
        _mint(msg.sender, initialSupply);
    }

    function mint(address to, uint256 amount) public {
        _mint(to, amount);
    }

    function decimals() public view virtual override returns (uint8) {
        return _decimals;
    }
}
