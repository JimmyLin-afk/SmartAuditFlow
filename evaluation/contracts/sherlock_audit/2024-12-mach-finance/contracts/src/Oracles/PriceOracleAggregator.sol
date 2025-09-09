// SPDX-License-Identifier: BSD-3-Clause
pragma solidity 0.8.22;

import {PriceOracle} from "../PriceOracle.sol";
import {IOracleSource} from "./IOracleSource.sol";
import {CErc20} from "../CErc20.sol";
import {CToken} from "../CToken.sol";
import {Ownable2StepUpgradeable} from "@openzeppelin/contracts-upgradeable/access/Ownable2StepUpgradeable.sol";
import {Initializable} from "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import {UUPSUpgradeable} from "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";

contract PriceOracleAggregator is PriceOracle, Initializable, Ownable2StepUpgradeable, UUPSUpgradeable {
    address public constant NATIVE_ASSET = 0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE;

    /// @notice Emitted when the oracle list for a token is updated
    event TokenOraclesUpdated(address indexed token, IOracleSource[] newOracles);

    // Mapping between underlying token and oracle sources
    mapping(address => IOracleSource[]) public tokenToOracleSources;

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    function initialize(address initialOwner) public initializer {
        __Ownable_init(initialOwner);
        __UUPSUpgradeable_init();
    }

    /**
     * @notice Get the price of the underlying asset of a cToken
     * @dev Iterates through oracle sources until it finds a valid price. Returns 0 if no valid price is found.
     * @param cToken The cToken contract to get the underlying price for
     * @return The price of the underlying asset in USD, scaled by 10^(36 - underlying decimals).
     *         Returns 0 if no valid price is found from any oracle source.
     */
    function getUnderlyingPrice(CToken cToken) public view override returns (uint256) {
        address underlying = _getUnderlyingAddress(cToken);
        IOracleSource[] memory oracles = tokenToOracleSources[underlying];

        for (uint256 i; i < oracles.length; i++) {
            (uint256 price, bool isValid) = oracles[i].getPrice(underlying);

            if (isValid) {
                return price;
            }
        }
        return 0;
    }

    function updateTokenOracles(address token, IOracleSource[] memory oracles) external onlyOwner {
        tokenToOracleSources[token] = oracles;
        emit TokenOraclesUpdated(token, oracles);
    }

    function _getUnderlyingAddress(CToken cToken) internal view returns (address) {
        address asset;
        if (compareStrings(cToken.symbol(), "cSonic")) {
            asset = NATIVE_ASSET;
        } else {
            asset = address(CErc20(address(cToken)).underlying());
        }
        return asset;
    }

    function compareStrings(string memory a, string memory b) internal pure returns (bool) {
        return (keccak256(abi.encodePacked((a))) == keccak256(abi.encodePacked((b))));
    }

    function _authorizeUpgrade(address newImplementation) internal override onlyOwner {}
}
