// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import {UUPSUpgradeable} from "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import {OwnableUpgradeable} from "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import {Address} from "@openzeppelin/contracts/utils/Address.sol";

import {IExactOutSwapper} from "./IExactOutSwapper.sol";
import {PerpManager} from "../admin/PerpManager.sol";

contract ExactOutSwapper is 
    IExactOutSwapper,
    UUPSUpgradeable,
    OwnableUpgradeable
{
    using Address for address;
    using Address for address payable;
    using SafeERC20 for IERC20;

    /// @dev Maps swap router addresses to their whitelisted status
    mapping(address => bool) public isWhitelistedSwapRouter;

    /// @dev Maps function selectors to their whitelisted status
    mapping(bytes4 => bool) public isWhitelistedFunctionSelector;

    /**
     * @dev Checks if the caller is an admin
     */
    modifier onlyAdmin() {
        _getManager().isAdmin(msg.sender);
        _;
    }

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    function initialize(address _manager) external initializer {
        __Ownable_init(_manager);
        __UUPSUpgradeable_init();
    }

    /// @inheritdoc IExactOutSwapper
    function swapExactOut(
        address tokenIn,
        address tokenOut,
        uint256 amountOut,
        uint256 amountInMax,
        FunctionCallData calldata swapCallData,
        FunctionCallData calldata reverseCallData
    ) external payable returns (uint256 amountIn) {
        // Check if the call targets and functions are whitelisted
        _validateCallData(swapCallData);
        _validateCallData(reverseCallData);

        address sender = msg.sender;

        // Transfer the maximum amount of input tokens to this contract
        IERC20(tokenIn).safeTransferFrom(sender, address(this), amountInMax);

        // Approve the swap router to spend the input tokens
        IERC20(tokenIn).forceApprove(swapCallData.to, amountInMax);
        
        // Perform the initial swap
        swapCallData.to.functionCallWithValue(swapCallData.data, swapCallData.value);
        uint256 amountOutReceived = IERC20(tokenOut).balanceOf(address(this));
        if (amountOutReceived < amountOut) {
            revert InsufficientAmountOutReceived();
        }

        // Send the expected amount of output tokens to the caller
        IERC20(tokenOut).safeTransfer(sender, amountOut);

        // Check for excess output tokens
        uint256 excessAmountOut = amountOutReceived - amountOut;
        uint256 excessAmountIn;
        if (excessAmountOut > 0) {
            // Send the excess output tokens to the swap router, taking advantage of the `hasAlreadyPaid` check
            IERC20(tokenOut).safeTransfer(reverseCallData.to, excessAmountOut);
        
            // Perform the reverse swap to sell the excess (amountIn should be 0 to spend the tokens we just sent)
            reverseCallData.to.functionCall(reverseCallData.data);

            // Get the amount of input tokens received from the reverse swap
            excessAmountIn = IERC20(tokenIn).balanceOf(address(this));
            if (excessAmountIn > 0) {
                // Transfer the excess input tokens back to the caller
                IERC20(tokenIn).safeTransfer(sender, excessAmountIn);
            }
        }

        // Calculate the net amount of input tokens spent
        amountIn = amountInMax - excessAmountIn;
    }

    /// @inheritdoc IExactOutSwapper
    function setWhitelistedAddress(address swapRouter, bool isWhitelisted) external onlyAdmin {
        isWhitelistedSwapRouter[swapRouter] = isWhitelisted;
    }

    /// @inheritdoc IExactOutSwapper
    function setWhitelistedFunctionSelectors(bytes4[] calldata selectors, bool isWhitelisted) external onlyAdmin {
        for (uint256 i; i < selectors.length; ) {
            bytes4 selector = selectors[i];
            isWhitelistedFunctionSelector[selector] = isWhitelisted;
            unchecked {
                ++i;
            }
        }
    }

    /// @inheritdoc IExactOutSwapper
    function recoverERC20(
        address token,
        address to,
        uint256 amount
    ) external onlyAdmin {
        if (amount > IERC20(token).balanceOf(address(this))) {
            revert InsufficientTokenBalance();
        }
        IERC20(token).safeTransfer(to, amount);
    }

    function recoverETH(address to, uint256 amount) external onlyAdmin {
        if (amount > address(this).balance) {
            revert InsufficientEthBalance();
        }
        payable(to).sendValue(amount);
    }

    /// @inheritdoc UUPSUpgradeable
    function _authorizeUpgrade(address) internal view override onlyAdmin {}

    /// @dev returns the manager of the contract
    function _getManager() internal view returns (PerpManager) {
        return PerpManager(owner());
    }

    function _validateCallData(
        FunctionCallData calldata callData
    ) internal view {
        if (!isWhitelistedSwapRouter[callData.to]) {
            revert NotWhitelistedSwapRouter(callData.to);
        }
        bytes4 selector = bytes4(callData.data);
        if (!isWhitelistedFunctionSelector[selector]) {
            revert NotWhitelistedFunctionSelector(selector);
        }
    }
}