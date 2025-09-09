//SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import { ERC20 } from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import { VVVAuthorizationRegistryChecker } from "../auth/VVVAuthorizationRegistryChecker.sol";

contract MockERC20 is ERC20, VVVAuthorizationRegistryChecker {
    uint8 _decimals;

    constructor(
        uint256 decimals_,
        address _authRegistryAddress
    ) ERC20("MockERC20", "MOCK") VVVAuthorizationRegistryChecker(_authRegistryAddress) {
        _decimals = uint8(decimals_);
        uint256 initialSupply = 1000000000 * 10 ** decimals_;
        _mint(msg.sender, initialSupply);
    }

    function mint(address to, uint256 amount) public {
        _mint(to, amount);
    }

    function decimals() public view virtual override returns (uint8) {
        return _decimals;
    }

    function mintWithAuth(address to, uint256 amount) public onlyAuthorized {
        mint(to, amount);
    }
}
