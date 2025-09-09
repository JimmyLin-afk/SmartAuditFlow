// SPDX-License-Identifier: MIT
pragma solidity 0.8.19;

import "@openzeppelin/contracts-upgradeable/access/AccessControlEnumerableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/security/PausableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/token/ERC20/utils/SafeERC20Upgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import {IERC20Metadata} from "@openzeppelin/contracts/interfaces/IERC20Metadata.sol";
import {IMinter} from "./interfaces/IMinter.sol";
import {IBoostStablecoin} from "./interfaces/IBoostStablecoin.sol";

contract Minter is Initializable, AccessControlEnumerableUpgradeable, PausableUpgradeable, IMinter {
    using SafeERC20Upgradeable for IERC20Upgradeable;

    address public override boostAddress;
    address public override collateralAddress;
    address public override treasury;
    uint8 public override boostDecimals;
    uint8 public override collateralDecimals;

    bytes32 public constant WITHDRAWER_ROLE = keccak256("WITHDRAWER_ROLE");
    bytes32 public constant UNPAUSER_ROLE = keccak256("UNPAUSER_ROLE");
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant AMO_ROLE = keccak256("AMO_ROLE");

    error ZeroAddress();
    error NonContractSender();

    modifier onlyContract() {
        if (msg.sender.code.length == 0) revert NonContractSender();
        _;
    }

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    function initialize(address boostAddress_, address collateralAddress_, address treasury_) external initializer {
        __AccessControl_init();
        __Pausable_init();
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        if (boostAddress_ == address(0) || collateralAddress_ == address(0) || treasury_ == address(0))
            revert ZeroAddress();
        boostAddress = boostAddress_;
        collateralAddress = collateralAddress_;
        treasury = treasury_;
        boostDecimals = IERC20Metadata(boostAddress).decimals();
        collateralDecimals = IERC20Metadata(collateralAddress).decimals();
    }

    function pause() external onlyRole(PAUSER_ROLE) {
        _pause();
    }

    function unpause() external onlyRole(UNPAUSER_ROLE) {
        _unpause();
    }

    function setTokens(address boostAddress_, address collateralAddress_) external onlyRole(ADMIN_ROLE) {
        if (boostAddress_ == address(0) || collateralAddress_ == address(0)) revert ZeroAddress();
        boostAddress = boostAddress_;
        collateralAddress = collateralAddress_;
        boostDecimals = IERC20Metadata(boostAddress).decimals();
        collateralDecimals = IERC20Metadata(collateralAddress).decimals();
        emit TokenAddressesUpdated(boostAddress_, collateralAddress_);
    }

    function setTreasury(address treasury_) external onlyRole(ADMIN_ROLE) {
        if (treasury_ == address(0)) revert ZeroAddress();
        treasury = treasury_;
        emit TreasuryUpdated(treasury_);
    }

    function mint(address to, uint256 amount) external whenNotPaused onlyContract onlyRole(MINTER_ROLE) {
        IERC20Upgradeable(collateralAddress).safeTransferFrom(
            msg.sender,
            treasury,
            amount / (10 ** (boostDecimals - collateralDecimals))
        );
        IBoostStablecoin(boostAddress).mint(to, amount);
        emit TokenMinted(msg.sender, to, amount);
    }

    function protocolMint(address to, uint256 amount) external whenNotPaused onlyContract onlyRole(AMO_ROLE) {
        IBoostStablecoin(boostAddress).mint(to, amount);
        emit TokenProtocolMinted(msg.sender, to, amount);
    }

    function withdrawToken(address token, uint256 amount) external onlyRole(WITHDRAWER_ROLE) {
        IERC20Upgradeable(token).safeTransfer(treasury, amount);
        emit TokenWithdrawn(token, amount);
    }
}
