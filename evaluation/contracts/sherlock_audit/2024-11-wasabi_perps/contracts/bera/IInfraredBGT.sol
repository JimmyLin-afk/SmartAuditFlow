// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import {IAccessControl} from "@openzeppelin/contracts/access/IAccessControl.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";

interface IERC20Mintable is IERC20 {
    function MINTER_ROLE() external view returns (bytes32);

    function mint(address to, uint256 amount) external;
}

interface IInfraredBGT is IERC20Mintable, IAccessControl {
    function burn(uint256 amount) external;
}