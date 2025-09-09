// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol";
import "@openzeppelin/contracts/utils/Address.sol";
// import "hardhat/console.sol";
import "../weth/IWETH.sol";
import "./IMockSwap.sol";

contract MockSwapRouter {
    using Address for address payable;

    error ETHTransferFailed();

    IWETH public immutable WETH9;
    IMockSwap public mockSwap;

    constructor(IMockSwap _mockSwap, IWETH _weth) {
        mockSwap = _mockSwap;
        WETH9 = _weth;
    }

    function swap(
        address currencyIn,
        uint256 amountIn,
        address currencyOut,
        address recipient
    ) external payable returns(uint256 amountOut) {
        // console.log("MockSwapRouter.swap");
        if (msg.value == 0) {
            if (amountIn == 0) {
                amountIn = IERC20(currencyIn).balanceOf(address(this));
            } else {
                IERC20(currencyIn).transferFrom(msg.sender, address(this), amountIn);
            }
            IERC20(currencyIn).approve(address(mockSwap), amountIn);
        }
        amountOut = mockSwap.swap{value: msg.value}(currencyIn, amountIn, currencyOut);
        if (recipient != address(this)) {
            // console.log("Transferring %s %s to %s", amountOut, currencyOut, recipient);
            IERC20(currencyOut).transfer(recipient, amountOut);
        }
    }

    function swapExactlyOut(
        address currencyIn,
        address currencyOut,
        uint256 amountOut,
        uint256 amountInMax,
        address recipient
    ) external payable returns(uint256 amountIn) {
        // console.log("MockSwapRouter.swapExactlyOut");
        if (msg.value == 0) {
            IERC20(currencyIn).transferFrom(msg.sender, address(this), amountInMax);
            IERC20(currencyIn).approve(address(mockSwap), amountInMax);
        }
        amountIn = mockSwap.swapExactlyOut{value: msg.value}(currencyIn, currencyOut, amountOut);
        if (amountIn < amountInMax) {
            if (msg.value == 0) {
                IERC20(currencyIn).transfer(msg.sender, amountInMax - amountIn);
            } else {
                (bool success, ) = payable(msg.sender).call{value: msg.value - amountIn}("");
                if (!success) revert ETHTransferFailed();
            }
        }
        if (recipient != address(this))
            IERC20(currencyOut).transfer(recipient, amountOut);
    }

    function multicall(bytes[] calldata data) public payable returns (bytes[] memory results) {
        // console.log("MockSwapRouter.multicall: data.length = %d", data.length);
        results = new bytes[](data.length);
        for (uint256 i = 0; i < data.length; i++) {
            // console.log("i = %d", i);
            (bool success, bytes memory result) = address(this).delegatecall(data[i]);

            if (!success) {
                // console.log("failed");
                // Next 5 lines from https://ethereum.stackexchange.com/a/83577
                if (result.length < 68) revert();
                assembly {
                    result := add(result, 0x04)
                }
                revert(abi.decode(result, (string)));
            }

            results[i] = result;
        }
    }

    function sweepTokenWithFee(
        address token,
        uint256 amountMinimum,
        address recipient,
        uint256 feeBips,
        address feeRecipient
    ) public payable {
        // console.log("MockSwapRouter.sweepTokenWithFee");
        require(feeBips > 0 && feeBips <= 100);

        uint256 balanceToken = IERC20(token).balanceOf(address(this));
        require(balanceToken >= amountMinimum, 'Insufficient token');

        if (balanceToken > 0) {
            // console.log("Token balance to sweep: %s %s", balanceToken, token);
            uint256 feeAmount = balanceToken * feeBips / 10_000;
            // console.log("Transferring fee amount: %s to %s", feeAmount, feeRecipient);
            if (feeAmount > 0) IERC20(token).transfer(feeRecipient, feeAmount);
            // console.log("Transferring %s to %s", balanceToken - feeAmount, recipient);
            IERC20(token).transfer(recipient, balanceToken - feeAmount);
        }
    }

    function unwrapWETH9WithFee(
        uint256 amountMinimum,
        address recipient,
        uint256 feeBips,
        address feeRecipient
    ) public payable {
        require(feeBips > 0 && feeBips <= 100);

        uint256 balanceWETH9 = WETH9.balanceOf(address(this));
        require(balanceWETH9 >= amountMinimum, 'Insufficient WETH9');

        if (balanceWETH9 > 0) {
            WETH9.withdraw(balanceWETH9);
            uint256 feeAmount = balanceWETH9 * feeBips / 10_000;
            if (feeAmount > 0) payable(feeRecipient).sendValue(feeAmount);
            payable(recipient).sendValue(balanceWETH9 - feeAmount);
        }
    }

    receive() external payable {}
}