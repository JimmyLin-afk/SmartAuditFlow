// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "./weth/IWETH.sol";
import {IWasabiPerps} from "./IWasabiPerps.sol";

library PerpUtils {
    using Address for address;
    using SafeERC20 for IERC20;

    /// @dev Pays ETH to a given address
    /// @param _amount The amount to pay
    /// @param _target The address to pay to
    function payETH(uint256 _amount, address _target) internal {
        if (_amount > 0) {
            (bool sent, ) = payable(_target).call{value: _amount}("");
            if (!sent) {
                revert IWasabiPerps.EthTransferFailed(_amount, _target);
            }
        }
    }

    /// @dev Computes the close fee for a position by looking at the position size
    /// @param _position the position to compute the close fee for
    /// @param _size the size to close
    /// @param _isLong whether the position is long or short
    function computeCloseFee(
        IWasabiPerps.Position calldata _position,
        uint256 _size,
        bool _isLong
    ) internal pure returns(uint256) {
        uint256 denominator = _position.feesToBePaid + (
            _isLong ? _position.downPayment + _position.principal : _position.collateralAmount
        );
        return (_size * _position.feesToBePaid) / denominator;
    }

    /// @dev Receives payment from a given address
    /// @param _currency the currency to receive
    /// @param _amount the amount to receive
    /// @param _wethAddress the WETH address
    /// @param _sender the address to receive from
    function receivePayment(
        address _currency,
        uint256 _amount,
        address _wethAddress,
        address _sender
    ) internal {
        if (msg.value > 0) {
            if (_currency != _wethAddress) revert IWasabiPerps.InvalidCurrency();
            if (msg.value != _amount) revert IWasabiPerps.InsufficientAmountProvided();
            wrapWETH(_wethAddress);
        } else {
            IERC20(_currency).safeTransferFrom(_sender, address(this), _amount);
        }
    }

    /// @dev Wraps the whole ETH in this contract
    function wrapWETH(address _wethAddress) internal {
        IWETH weth = IWETH(_wethAddress);
        weth.deposit{value: address(this).balance}();
    }

    /// @dev Executes a given list of functions and returns the balance changes
    /// @param _marketplaceCallData List of marketplace calldata
    /// @param _tokenIn the token to swap from
    /// @param _tokenOut the token to swap to
    /// @return amountIn the amount of tokenIn swapped
    /// @return amountOut the amount of tokenOut received
    function executeSwapFunctions(
        IWasabiPerps.FunctionCallData[] memory _marketplaceCallData,
        IERC20 _tokenIn,
        IERC20 _tokenOut
    ) internal returns (uint256 amountIn, uint256 amountOut) {
        amountIn = _tokenIn.balanceOf(address(this));
        amountOut = _tokenOut.balanceOf(address(this));
        uint256 length = _marketplaceCallData.length;
        for (uint256 i; i < length; ++i) {
            IWasabiPerps.FunctionCallData memory functionCallData = _marketplaceCallData[i];
            functionCallData.to.functionCallWithValue(functionCallData.data, functionCallData.value);
        }
        amountIn = amountIn - _tokenIn.balanceOf(address(this));
        amountOut = _tokenOut.balanceOf(address(this)) - amountOut;
    }

    /// @dev Deducts the given amount from the total amount
    /// @param _amount the amount to deduct from
    /// @param _deductAmount the amount to deduct
    /// @return remaining the remaining amount
    /// @return deducted the total deducted
    function deduct(uint256 _amount, uint256 _deductAmount) internal pure returns(uint256 remaining, uint256 deducted) {
        if (_amount > _deductAmount) {
            remaining = _amount - _deductAmount;
            deducted = _deductAmount;
        } else {
            remaining = 0;
            deducted = _amount;
        }
    }
}