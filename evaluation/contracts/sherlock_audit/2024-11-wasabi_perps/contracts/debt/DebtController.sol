// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/access/Ownable.sol";

import "./IDebtController.sol";

contract DebtController is Ownable, IDebtController {
    error InvalidValue();

    uint256 public constant LEVERAGE_DENOMINATOR = 100;
    uint256 public constant APY_DENOMINATOR = 100;

    uint256 public maxApy; // 300% APR will be 300
    uint256 public maxLeverage; // e.g. 3x leverage = 300
    uint256 public liquidationThreshold;
    uint256 public liquidationFeeBps;

    /// @dev creates a new DebtController
    /// @param _maxApy the max apy
    /// @param _maxLeverage the max leverage
    constructor(uint256 _maxApy, uint256 _maxLeverage) Ownable(msg.sender) {
        maxApy = _maxApy;
        maxLeverage = _maxLeverage;
        liquidationFeeBps = 500; // 5%
        // liquidationThreshold = _liquidationThreshold;
    }

    /// @inheritdoc IDebtController
    function computeMaxInterest(
        address,
        uint256 _principal,
        uint256 _lastFundingTimestamp
    ) public view returns(uint256 maxInterestToPay) {
        uint256 secondsSince = block.timestamp - _lastFundingTimestamp;
        maxInterestToPay = _principal * maxApy * secondsSince / (APY_DENOMINATOR * (365 days));
    }

    /// @inheritdoc IDebtController
    function computeMaxPrincipal(
        address,
        address,
        uint256 _downPayment
    ) external view returns (uint256 maxPrincipal) {
        maxPrincipal = _downPayment * (maxLeverage - LEVERAGE_DENOMINATOR) / LEVERAGE_DENOMINATOR;
    }

    /// @inheritdoc IDebtController
    function getLiquidationFeeBps(address, address) external view returns (uint256) {
        return liquidationFeeBps;
    }

    /// @inheritdoc IDebtController
    function getLiquidationFee(uint256 _downPayment, address, address) external view returns (uint256) {
        return (_downPayment * liquidationFeeBps) / 10000;
    }

    /// @dev sets the maximum leverage
    /// @param _maxLeverage the max leverage 
    function setMaxLeverage(uint256 _maxLeverage) external onlyOwner {
        if (_maxLeverage == 0) revert InvalidValue();
        if (_maxLeverage > 100 * LEVERAGE_DENOMINATOR) revert InvalidValue(); // 100x leverage
        maxLeverage = _maxLeverage;
    }

    /// @dev sets the maximum apy
    /// @param _maxApy the max APY 
    function setMaxAPY(uint256 _maxApy) external onlyOwner {
        if (_maxApy == 0) revert InvalidValue();
        if (_maxApy > 1000 * APY_DENOMINATOR) revert InvalidValue(); // 1000% APR
        maxApy = _maxApy;
    }

    /// @dev sets the liquidation threshold
    /// @param _liquidationThreshold the liquidation threshold
    function setLiquidationThreshold(uint256 _liquidationThreshold) external onlyOwner {
        if (_liquidationThreshold == 0) revert InvalidValue();
        liquidationThreshold = _liquidationThreshold;
    }

    /// @dev sets the liquidation fee bps
    /// @param _liquidationFeeBps the liquidation fee bps
    function setLiquidationFeeBps(uint256 _liquidationFeeBps) external onlyOwner {
        if (_liquidationFeeBps == 0) revert InvalidValue();
        if (_liquidationFeeBps > 1000) revert InvalidValue(); // 10%
        liquidationFeeBps = _liquidationFeeBps;
    }
}