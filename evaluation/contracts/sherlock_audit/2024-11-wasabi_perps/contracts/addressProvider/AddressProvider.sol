// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/access/Ownable.sol";

import "./IAddressProvider.sol";
import "../debt/IDebtController.sol";
import "../vaults/IWasabiVault.sol";
import "../router/IWasabiRouter.sol";

contract AddressProvider is Ownable, IAddressProvider {
    error InvalidAddress();
    error InvalidLiquidationFee();

    IDebtController public debtController;
    IWasabiRouter public wasabiRouter;
    address public feeReceiver;
    address public immutable wethAddress;
    address public liquidationFeeReceiver;
    address public stakingAccountFactory;
    IPartnerFeeManager public partnerFeeManager;
    constructor(
        IDebtController _debtController,
        IWasabiRouter _wasabiRouter,
        address _feeReceiver,
        address _wethAddress,
        address _liquidationFeeReceiver,
        address _stakingAccountFactory,
        IPartnerFeeManager _partnerFeeManager
    ) Ownable(msg.sender) {
        debtController = _debtController;
        wasabiRouter = _wasabiRouter;
        feeReceiver = _feeReceiver;
        wethAddress = _wethAddress;
        liquidationFeeReceiver = _liquidationFeeReceiver;
        stakingAccountFactory = _stakingAccountFactory;
        partnerFeeManager = _partnerFeeManager;
    }

    /// @inheritdoc IAddressProvider
    function getDebtController()
        external
        view
        override
        returns (IDebtController)
    {
        return debtController;
    }

    /// @inheritdoc IAddressProvider
    function getWasabiRouter()
        external
        view
        override
        returns (IWasabiRouter)
    {
        return wasabiRouter;
    }

    /// @inheritdoc IAddressProvider
    function getFeeReceiver()
        external
        view
        override
        returns (address)
    {
        return feeReceiver;
    }

    /// @inheritdoc IAddressProvider
    function getLiquidationFeeReceiver()
        external
        view
        override
        returns (address)
    {
        return liquidationFeeReceiver;
    }

    /// @inheritdoc IAddressProvider
    function getWethAddress() external view returns (address) {
        return wethAddress;
    }

    /// @inheritdoc IAddressProvider
    function getStakingAccountFactory() external view returns (address) {
        return stakingAccountFactory;
    }

    /// @inheritdoc IAddressProvider
    function getPartnerFeeManager() external view returns (IPartnerFeeManager) {
        return partnerFeeManager;
    }

    /// @dev sets the debt controller
    /// @param _debtController the debt controller
    function setDebtController(IDebtController _debtController) external onlyOwner {
        debtController = _debtController;
    }

    /// @dev sets the Wasabi router
    /// @param _wasabiRouter the Wasabi router
    function setWasabiRouter(IWasabiRouter _wasabiRouter) external onlyOwner {
        wasabiRouter = _wasabiRouter;
    }

    /// @dev sets the fee controller
    /// @param _feeReceiver the fee receiver
    function setFeeReceiver(address _feeReceiver) external onlyOwner {
        if (_feeReceiver == address(0)) revert InvalidAddress();
        feeReceiver = _feeReceiver;
    }

    /// @dev sets the fee controller
    /// @param _liquidationFeeReceiver the fee receiver
    function setLiquidationFeeReceiver(address _liquidationFeeReceiver) external onlyOwner {
        if (_liquidationFeeReceiver == address(0)) revert InvalidAddress();
        liquidationFeeReceiver = _liquidationFeeReceiver;
    }

    /// @dev sets the staking account factory
    /// @param _stakingAccountFactory the staking account factory
    function setStakingAccountFactory(address _stakingAccountFactory) external onlyOwner {
        if (_stakingAccountFactory == address(0)) revert InvalidAddress();
        stakingAccountFactory = _stakingAccountFactory;
    }

    /// @dev sets the partner fee manager
    /// @param _partnerFeeManager the partner fee manager
    function setPartnerFeeManager(address _partnerFeeManager) external onlyOwner {
        if (_partnerFeeManager == address(0)) revert InvalidAddress();
        partnerFeeManager = IPartnerFeeManager(_partnerFeeManager);
    }
}