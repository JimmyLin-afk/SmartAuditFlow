// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol";
// import "hardhat/console.sol";
import "./IMockSwap.sol";

contract MockSwap is IMockSwap {
    uint256 public constant PRICE_DENOMINATOR = 10_000;
    uint8 public constant ETH_DECIMALS = 18;
    mapping (address => mapping(address => uint256)) prices;

    constructor() payable {}

    function swap(
        address currencyIn,
        uint256 amountIn,
        address currencyOut
    ) external payable returns(uint256 amountOut) {
        // console.log('MockSwap.swap');
        uint256 price = prices[currencyIn][currencyOut];
        require(price > 0, 'Price Not Set');

        amountOut = amountIn * price / PRICE_DENOMINATOR;

        // check if we need to convert the amount to different decimals
        uint8 tokenInDecimals = currencyIn == address(0) 
            ? ETH_DECIMALS 
            : IERC20Metadata(currencyIn).decimals();
        uint8 tokenOutDecimals = currencyOut == address(0) 
            ? ETH_DECIMALS 
            : IERC20Metadata(currencyOut).decimals();
        if (tokenInDecimals != ETH_DECIMALS) {
            amountOut = tokenToWad(tokenInDecimals, amountOut);
        }
        if (tokenOutDecimals != ETH_DECIMALS) {
            amountOut = wadToToken(tokenOutDecimals, amountOut);
        }

        if (msg.value != 0) {
            // console.log("Checking that msg.value (%s) == amountIn (%s)", msg.value, amountIn);
            require(msg.value == amountIn, 'Not enough ETH supplied');
            // console.log("Payment received");
        } else {
            // console.log("Transferring %s %s", amountIn, currencyIn);
            IERC20(currencyIn).transferFrom(msg.sender, address(this), amountIn);
            // console.log('Payment received');
        }

        if (currencyOut == address(0)) {
            payETH(amountOut, msg.sender);
        } else {
            // console.log("Transferring %s %s. Current balance %s", amountOut, currencyOut, IERC20(currencyOut).balanceOf(address(this)));
            IERC20(currencyOut).transfer(msg.sender, amountOut);
            // console.log('Payment sent');
        }

        emit Swap(currencyIn, amountIn, currencyOut, amountOut);
    }

    function swapExact(
        address currencyIn,
        uint256 amountIn,
        address currencyOut,
        uint256 amountOut
    ) external payable returns(uint256) {
        if (currencyIn == address(0)) {
            require(msg.value == amountIn, 'Not enough ETH supplied');
        } else {
            IERC20(currencyIn).transferFrom(msg.sender, address(this), amountIn);
            // console.log('Payment received');
        }
        
        if (currencyOut == address(0)) {
            payETH(amountOut, msg.sender);
        } else {
            // console.log("Transferring %s %s. Current balance %s", amountOut, currencyOut, IERC20(currencyOut).balanceOf(address(this)));
            IERC20(currencyOut).transfer(msg.sender, amountOut);
            // console.log('Payment sent');
        }

        emit Swap(currencyIn, amountIn, currencyOut, amountOut);

        return amountOut;
    }

    function swapExactlyOut(
        address currencyIn,
        address currencyOut,
        uint256 amountOut
    ) external payable returns(uint256 amountIn) {
        // console.log("MockSwap.swapExactlyOut");
        uint256 price = prices[currencyIn][currencyOut];
        require(price > 0, 'Price Not Set');

        amountIn = amountOut * PRICE_DENOMINATOR / price;

        // check if we need to convert the amount to different decimals
        uint8 tokenInDecimals = currencyIn == address(0) 
            ? ETH_DECIMALS 
            : IERC20Metadata(currencyIn).decimals();
        uint8 tokenOutDecimals = currencyOut == address(0) 
            ? ETH_DECIMALS 
            : IERC20Metadata(currencyOut).decimals();
        if (tokenInDecimals != ETH_DECIMALS) {
            amountIn = wadToToken(tokenInDecimals, amountIn);
        }
        if (tokenOutDecimals != ETH_DECIMALS) {
            amountIn = tokenToWad(tokenOutDecimals, amountIn);
        }

        if (msg.value != 0) {
            // console.log("Checking that msg.value (%s) >= amountIn (%s)", msg.value, amountIn);
            require(msg.value >= amountIn, 'Not enough ETH supplied');
            // console.log("Payment received");
            if (msg.value > amountIn) {
                // console.log('Returning %s wei to %s', msg.value - amountIn, msg.sender);
                payETH(msg.value - amountIn, msg.sender);
                // console.log("Excess ETH returned");
            }
        } else {
            // console.log("Transferring %s %s", amountIn, currencyIn);
            IERC20(currencyIn).transferFrom(msg.sender, address(this), amountIn);
            // console.log('Payment received');
        }

        if (currencyOut == address(0)) {
            payETH(amountOut, msg.sender);
        } else {
            // console.log("Transferring %s %s. Current balance %s", amountOut, currencyOut, IERC20(currencyOut).balanceOf(address(this)));
            IERC20(currencyOut).transfer(msg.sender, amountOut);
            // console.log('Payment sent');
        }

        emit Swap(currencyIn, amountIn, currencyOut, amountOut);
    }

    function revertingFunction() external payable {
        revert SwapReverted();
    }

    function setPrice(address token1, address token2, uint256 price) external {
        prices[token1][token2] = price;
        prices[token2][token1] = PRICE_DENOMINATOR * PRICE_DENOMINATOR / price;
    }
    
    function payETH(uint256 _amount, address target) private {
        (bool sent, ) = payable(target).call{value: _amount}("");
        require(sent, 'Couldnt send eth');
    }

    /// @notice Convert amount from 'tokenDecimals' to 18 decimals precision
    /// @param tokenDecimals Decimals of the token. 8 decimals uint like in the ERC20 standard
    /// @param tokenAmount Amount with 'tokenDecimals' precision
    /// @return wadAmount Scaled amount to 18 decimals
    function tokenToWad(uint8 tokenDecimals, uint256 tokenAmount) internal pure returns (uint256) {
        if (tokenDecimals == ETH_DECIMALS) {
            return tokenAmount;
        } else if (tokenDecimals < ETH_DECIMALS) {
            return tokenAmount * (10 ** (ETH_DECIMALS - tokenDecimals));
        }

        return tokenAmount / (10 ** (tokenDecimals - ETH_DECIMALS));
    }

    /// @notice Convert amount from 18 decimals to 'tokenDecimals' precision
    /// @param tokenDecimals Decimals of the token. 8 decimals uint like in the ERC20 standard
    /// @param wadAmount Amount with 18 decimals precision
    /// @return amount Amount scaled to 'tokenDecimals' precision
    function wadToToken(uint8 tokenDecimals, uint256 wadAmount) internal pure returns (uint256) {
        if (tokenDecimals == ETH_DECIMALS) {
            return wadAmount;
        } else if (tokenDecimals < ETH_DECIMALS) {
            return wadAmount / (10 ** (ETH_DECIMALS - tokenDecimals));
        }

        return wadAmount * 10 ** (tokenDecimals - ETH_DECIMALS);
    }

    receive() external payable virtual {}

    fallback() external payable {
        require(false, "No fallback");
    }
}