// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/utils/cryptography/EIP712.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract DynamicSwap is EIP712, Ownable {
    bytes32 private constant _SWAP_REQUEST_DATA_HASH =
        keccak256("SwapRequest(address currencyIn,address currencyOut,uint256 amount,bool exactIn,uint256 expiration,uint256 price,uint256 priceDenominator)");

    error SwapReverted();
    error InvalidSignature();

    event Swap(
        address currencyIn,
        uint256 amountIn,
        address currencyOut,
        uint256 amountOut
    );

    struct Signature {
        uint8 v;
        bytes32 r;
        bytes32 s;
    }

    struct SwapRequest {
        address currencyIn;
        address currencyOut;
        uint256 amount;
        bool exactIn;
        uint256 expiration;
        uint256 price;
        uint256 priceDenominator;
    }

    constructor() EIP712("DynamicSwap", "1") Ownable(msg.sender) payable {}

    function swap(
        SwapRequest calldata _request, Signature calldata _signature
    ) external payable returns(uint256 amountIn, uint256 amountOut) {
        // Validate Signature
        bytes32 typedDataHash = _hashTypedDataV4(hash(_request));
        address signer = ecrecover(typedDataHash, _signature.v, _signature.r, _signature.s);
        if (owner() != signer) {
            revert InvalidSignature();
        }

        // Validate Expiration
        if (_request.expiration < block.timestamp) {
            revert SwapReverted();
        }

        // Compute Amounts
        if (_request.exactIn) {
            amountIn = _request.amount;
            amountOut = amountIn * _request.price / _request.priceDenominator;
        } else {
            amountOut = _request.amount;
            amountIn = amountOut * _request.priceDenominator / _request.price;
        }

        if (_request.currencyIn == address(0)) {
            require(msg.value == amountIn, 'Not enough ETH supplied');
        } else {
            IERC20(_request.currencyIn).transferFrom(msg.sender, address(this), amountIn);
        }

        if (_request.currencyOut == address(0)) {
            payETH(amountOut, msg.sender);
        } else {
            IERC20(_request.currencyOut).transfer(msg.sender, amountOut);
        }

        emit Swap(_request.currencyIn, amountIn, _request.currencyOut, amountOut);
    }

    function revertingFunction() external payable {
        revert SwapReverted();
    }
    
    function payETH(uint256 _amount, address target) private {
        (bool sent, ) = payable(target).call{value: _amount}("");
        require(sent, 'Couldnt send eth');
    }

    function hash(SwapRequest calldata _request) public pure returns (bytes32) {
        return keccak256(
            abi.encode(
                _SWAP_REQUEST_DATA_HASH,
                _request.currencyIn,
                _request.currencyOut,
                _request.amount,
                _request.exactIn,
                _request.expiration,
                _request.price,
                _request.priceDenominator
            )
        );
    }

    receive() external payable virtual {}

    fallback() external payable {
        require(false, "No fallback");
    }
}
