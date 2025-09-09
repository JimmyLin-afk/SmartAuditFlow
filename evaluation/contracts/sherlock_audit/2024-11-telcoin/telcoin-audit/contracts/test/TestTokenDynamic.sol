// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";

/**
 * @title IERC20Withdrawable
 * @author Amir M. Shirif
 * @notice A Telcoin Contract
 * @notice withdraw from wrapped token
 */
interface IERC20Withdrawable {
    /**
     * @dev the exit function
     * @param amount uint256 balance to be returned
     */
    function withdraw(uint256 amount) external;
}

//TESTING ONLY
contract TestTokenDynamic is ERC20, IERC20Withdrawable {
    uint8 public immutable _decimals;

    constructor(
        string memory name,
        string memory symbol,
        uint8 decimals_,
        address recipient,
        uint256 amount
    ) ERC20(name, symbol) {
        _decimals = decimals_;
        _mint(recipient, amount * (10 ** uint256(decimals())));
    }

    function decimals() public view override returns (uint8) {
        return _decimals;
    }

    function mintTo(address recipent, uint256 amount) public {
        _mint(recipent, amount);
    }

    function deposit() public payable {
        _mint(_msgSender(), msg.value);
    }

    function withdraw(uint wad) public {
        require(balanceOf(_msgSender()) >= wad);
        _burn(_msgSender(), wad);
        payable(_msgSender()).transfer(wad);
    }

    fallback() external payable {
        deposit();
    }

    receive() external payable {}
}
