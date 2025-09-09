// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import { ERC20 } from "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract FluidToken is ERC20 {
    constructor() ERC20("FLUID Token", "FLUID") { }
}
