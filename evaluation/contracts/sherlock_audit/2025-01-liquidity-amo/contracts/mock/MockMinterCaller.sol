// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {IERC20Metadata} from "@openzeppelin/contracts/interfaces/IERC20Metadata.sol";
import "@openzeppelin/contracts-upgradeable/token/ERC20/utils/SafeERC20Upgradeable.sol";
import "../interfaces/IMinter.sol";

contract MockMinterCaller {
    using SafeERC20Upgradeable for IERC20Upgradeable;

    address public boostAddress;
    address public collateralAddress;
    address public minterAddress;
    uint8 public boostDecimals;
    uint8 public collateralDecimals;

    constructor(address minterAddress_, address boostAddress_, address collateralAddress_) {
        minterAddress = minterAddress_;
        boostAddress = boostAddress_;
        collateralAddress = collateralAddress_;
        boostDecimals = IERC20Metadata(boostAddress_).decimals();
        collateralDecimals = IERC20Metadata(collateralAddress_).decimals();
    }

    function testMint(address to, uint256 amount) external {
        IERC20Upgradeable(collateralAddress).safeTransferFrom(
            msg.sender,
            address(this),
            amount / (10 ** (boostDecimals - collateralDecimals))
        );
        IERC20Upgradeable(collateralAddress).approve(
            minterAddress,
            amount / (10 ** (boostDecimals - collateralDecimals))
        );
        IMinter(minterAddress).mint(to, amount);
    }

    function testProtocolMint(address to, uint256 amount) external {
        IMinter(minterAddress).protocolMint(to, amount);
    }
}
