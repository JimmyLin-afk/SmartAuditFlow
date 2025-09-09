// SPDX-License-Identifier: BSD-3-Clause
pragma solidity 0.8.22;

import {Ownable2Step, Ownable} from "@openzeppelin/contracts/access/Ownable2Step.sol";
import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import {IStdReference} from "../Band/IStdReference.sol";
import {PriceOracle} from "../../PriceOracle.sol";
import {CErc20} from "../../CErc20.sol";
import {CToken} from "../../CToken.sol";
import {IOracleSource} from "../IOracleSource.sol";

contract BandOracle is IOracleSource, Ownable2Step {
    uint256 public constant PRICE_SCALE = 36;
    uint256 public constant BAND_SCALE_FACTOR = 18;
    uint256 public constant NATIVE_DECIMALS = 18;
    address public constant NATIVE_ASSET = 0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE;

    event UnderlyingSymbolSet(address indexed token, string symbol);

    /// @notice BAND reference
    IStdReference public immutable bandReference;

    /// @notice The mapping records the cToken and its underlying symbol that we use for BAND reference
    ///         It's not necessarily equals to the symbol in the underlying contract
    mapping(address => string) public tokenToBandSymbol;

    /// @notice Quote symbol we used for BAND reference contract
    string public constant QUOTE_SYMBOL = "USD";

    constructor(
        address _owner,
        address _bandReference,
        address[] memory _underlyingTokens,
        string[] memory _bandSymbols
    ) Ownable(_owner) {
        require(_underlyingTokens.length == _bandSymbols.length, "BandOracle: Length mismatch");

        bandReference = IStdReference(_bandReference);
        for (uint256 i = 0; i < _underlyingTokens.length; i++) {
            _setUnderlyingSymbol(_underlyingTokens[i], _bandSymbols[i]);
        }
    }

    /**
     * @notice Get the price of a token from Band oracle
     * @dev The price is scaled to maintain precision based on token decimals. Band prices are always
     *      multiplied by 1e18. The function scales the price to match PRICE_SCALE (36) decimals.
     * @param token The address of the token to get the price for
     * @return A tuple containing:
     *         - The price of the token in USD scaled by 10^(36 - token decimals)
     *         - A boolean indicating if the price is valid (true) or not (false)
     */
    function getPrice(address token) external view returns (uint256, bool) {
        uint256 price = _getLatestPrice(token);
        uint256 decimals = _getDecimals(token);

        uint256 scaledPrice;

        // Price from BAND is always multiplied by 1e18 base
        if (BAND_SCALE_FACTOR + decimals <= PRICE_SCALE) {
            uint256 scale = 10 ** (PRICE_SCALE - BAND_SCALE_FACTOR - decimals);
            scaledPrice = price * scale;
        } else {
            uint256 scale = 10 ** (BAND_SCALE_FACTOR + decimals - PRICE_SCALE);
            scaledPrice = price / scale;
        }

        if (scaledPrice == 0) {
            return (0, false);
        }

        return (scaledPrice, true);
    }

    /**
     * @notice Gets the latest price for a token from the BAND oracle
     * @dev The price is scaled to maintain precision based on token decimals
     * @param token The address of the token to get the price for
     * @return The token price in USD scaled by 10^(36 - token decimals)
     */
    function _getLatestPrice(address token) internal view returns (uint256) {
        // Return 0 if underlying symbol is not set, reverts are handled by caller
        if (bytes(tokenToBandSymbol[token]).length == 0) return 0;

        IStdReference.ReferenceData memory data = bandReference.getReferenceData(tokenToBandSymbol[token], QUOTE_SYMBOL);

        return data.rate;
    }

    function _getDecimals(address token) internal view returns (uint256) {
        if (token == NATIVE_ASSET) {
            return NATIVE_DECIMALS;
        } else {
            return ERC20(token).decimals();
        }
    }

    /// Owner functions
    function _setUnderlyingSymbol(address token, string memory symbol) internal {
        require(bytes(symbol).length > 0, "BandOracle: Symbol cannot be empty");
        // Attempt to check if the symbol is valid, ignore return value, should revert if invalid
        bandReference.getReferenceData(symbol, QUOTE_SYMBOL);
        tokenToBandSymbol[token] = symbol;
        emit UnderlyingSymbolSet(token, symbol);
    }

    function setUnderlyingSymbol(address token, string memory symbol) external onlyOwner {
        _setUnderlyingSymbol(token, symbol);
    }

    function bulkSetUnderlyingSymbols(address[] memory tokens, string[] memory symbols) external onlyOwner {
        for (uint256 i = 0; i < tokens.length; i++) {
            _setUnderlyingSymbol(tokens[i], symbols[i]);
        }
    }
}
