// SPDX-License-Identifier: BSD-3-Clause
pragma solidity 0.8.22;

import {Ownable2Step, Ownable} from "@openzeppelin/contracts/access/Ownable2Step.sol";
import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import {IOracleSource} from "../IOracleSource.sol";
import {IApi3ReaderProxy} from "@api3/contracts/interfaces/IApi3ReaderProxy.sol";

contract API3Oracle is IOracleSource, Ownable2Step {
    uint256 public constant PRICE_SCALE = 36;
    uint256 public constant API3_SCALE_FACTOR = 18;
    uint256 public constant NATIVE_DECIMALS = 18;
    address public constant NATIVE_ASSET = 0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE;

    event UnderlyingTokenApi3ProxyAddressSet(address indexed token, address api3ProxyAddress);

    // Mapping between underlying token and API3 proxy address
    mapping(address => address) public tokenToApi3ProxyAddress;

    constructor(address _owner, address[] memory _underlyingTokens, address[] memory _api3ProxyAddresses)
        Ownable(_owner)
    {
        require(
            _underlyingTokens.length == _api3ProxyAddresses.length,
            "API3Oracle: Lengths of tokens and API3 proxy addresses must match"
        );

        for (uint256 i = 0; i < _underlyingTokens.length; i++) {
            tokenToApi3ProxyAddress[_underlyingTokens[i]] = _api3ProxyAddresses[i];
        }
    }

    /**
     * @notice Get the price of a token from API3 proxy
     * @param token The address of the token to get the price for
     * @return price Price of the token in USD as an unsigned integer scaled up by 10 ^ (36 - token decimals)
     * @return isValid Boolean indicating if the price is valid
     */
    function getPrice(address token) external view returns (uint256 price, bool isValid) {
        uint256 price = _getLatestPrice(token);
        uint256 decimals = _getDecimals(token);

        uint256 scaledPrice;

        // Price from API3 is always multiplied by 1e18 base
        if (API3_SCALE_FACTOR + decimals <= PRICE_SCALE) {
            uint256 scale = 10 ** (PRICE_SCALE - API3_SCALE_FACTOR - decimals);
            scaledPrice = price * scale;
        } else {
            uint256 scale = 10 ** (API3_SCALE_FACTOR + decimals - PRICE_SCALE);
            scaledPrice = price / scale;
        }

        if (scaledPrice == 0) {
            return (0, false);
        }

        return (scaledPrice, true);
    }

    /**
     */
    /**
     * @notice Gets the latest price for a token from the API3 proxy
     * @dev Returns 0 if the token has no proxy address configured or if the price is not positive
     * @param token The address of the token to get the price for
     * @return price token price in USD with 18 decimals of precision
     */
    function _getLatestPrice(address token) internal view returns (uint256) {
        address proxyAddress = tokenToApi3ProxyAddress[token];
        if (proxyAddress == address(0)) {
            return 0;
        }

        IApi3ReaderProxy api3Proxy = IApi3ReaderProxy(proxyAddress);

        // API3 returns prices with scaled up by 1e18 base
        // https://docs.api3.org/dapps/integration/contract-integration.html#using-value
        (int224 price,) = api3Proxy.read();

        // Ensure price is positive, negative & zero prices are not valid
        if (price <= 0) {
            return 0;
        }

        return uint256(int256(price));
    }

    function _getDecimals(address token) internal view returns (uint256) {
        if (token == NATIVE_ASSET) {
            return NATIVE_DECIMALS;
        } else {
            return ERC20(token).decimals();
        }
    }

    /// Admin functions to set API3 oracle proxy address for a token ////
    function _setApi3ProxyAddress(address underlyingToken, address api3ProxyAddress) internal {
        require(api3ProxyAddress != address(0), "API3Oracle: API3 proxy address cannot be zero");
        // Attempt to check if the API3 proxy address is valid, ignore return value, should revert if invalid
        IApi3ReaderProxy(api3ProxyAddress).read();
        tokenToApi3ProxyAddress[underlyingToken] = api3ProxyAddress;
        emit UnderlyingTokenApi3ProxyAddressSet(underlyingToken, api3ProxyAddress);
    }

    function setApi3ProxyAddress(address underlyingToken, address api3ProxyAddress) public onlyOwner {
        _setApi3ProxyAddress(underlyingToken, api3ProxyAddress);
    }

    function bulkSetApi3ProxyAddresses(address[] memory _underlyingTokens, address[] memory _api3ProxyAddresses)
        external
        onlyOwner
    {
        require(
            _underlyingTokens.length == _api3ProxyAddresses.length,
            "API3Oracle: Lengths of tokens and API3 proxy addresses must match"
        );

        for (uint256 i = 0; i < _underlyingTokens.length; i++) {
            _setApi3ProxyAddress(_underlyingTokens[i], _api3ProxyAddresses[i]);
        }
    }
}
