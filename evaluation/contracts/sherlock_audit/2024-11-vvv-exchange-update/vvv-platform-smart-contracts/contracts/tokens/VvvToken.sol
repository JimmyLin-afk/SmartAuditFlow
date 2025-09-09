//SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

/**
 * @title VvvToken
 * @author @vvvfund (@curi0n-s + @c0dejax)
 */
import { ERC20 } from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import { ERC20Capped } from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Capped.sol";
import { VVVAuthorizationRegistryChecker } from "contracts/auth/VVVAuthorizationRegistryChecker.sol";

contract VVVToken is ERC20Capped, VVVAuthorizationRegistryChecker {
    constructor(
        uint256 _cap,
        uint256 _initialSupply,
        address _authorizationRegistryAddress
    )
        ERC20("vVvToken", "VVV")
        ERC20Capped(_cap)
        VVVAuthorizationRegistryChecker(_authorizationRegistryAddress)
    {
        _mint(msg.sender, _initialSupply);
    }

    function mint(address _to, uint256 _amount) public onlyAuthorized {
        _mint(_to, _amount);
    }
}
