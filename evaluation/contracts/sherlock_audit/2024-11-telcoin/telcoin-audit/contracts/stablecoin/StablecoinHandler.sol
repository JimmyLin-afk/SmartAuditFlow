// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {AccessControlUpgradeable} from "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import {PausableUpgradeable} from "@openzeppelin/contracts-upgradeable/utils/PausableUpgradeable.sol";
import {Stablecoin, SafeERC20, ERC20PermitUpgradeable} from "../stablecoin/Stablecoin.sol";

/**
 * @title StablecoinHandler
 * @author Amir M. Shirif
 * @notice A Telcoin Contract
 *
 * @notice This handles the minting and burning of stablecoins
 */
abstract contract StablecoinHandler is
    AccessControlUpgradeable,
    PausableUpgradeable
{
    using SafeERC20 for ERC20PermitUpgradeable;

    struct StablecoinSwap {
        // address provides intermediary stablecoin or eXYZs that are not mintable
        address liquiditySafe;
        // recipient of the target currency
        address destination;
        // the originating currency
        address origin;
        // the amount of currency being provided
        uint256 oAmount;
        // the target currency
        address target;
        // the amount of currency to be provided
        uint256 tAmount;
        // currency that the fee should be paid out in
        address stablecoinFeeCurrency;
        // deposit address for stablecoin fee
        address stablecoinFeeSafe;
        // stablecoin fee amount
        uint256 feeAmount;
    }

    struct eXYZ {
        // status of address as stablecoin
        bool validity;
        // the max mint limit
        uint256 maxSupply;
        // the min burn limit
        uint256 minSupply;
    }

    /// @custom:storage-location erc7201:telcoin.storage.StablecoinHandler
    struct StablecoinHandlerStorage {
        mapping(address => eXYZ) _eXYZs;
    }

    // keccak256(abi.encode(uint256(keccak256("erc7201.telcoin.storage.StablecoinHandler")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant StablecoinHandlerStorageLocation =
        0x38361881985b0f585e6124dca158a3af102bffba0feb9c42b0b40825f41a3300;

    function _getStablecoinHandlerStorage()
        private
        pure
        returns (StablecoinHandlerStorage storage $)
    {
        assembly {
            $.slot := StablecoinHandlerStorageLocation
        }
    }

    // Authorized Roles
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    bytes32 public constant SWAPPER_ROLE = keccak256("SWAPPER_ROLE");
    bytes32 public constant MAINTAINER_ROLE = keccak256("MAINTAINER_ROLE");

    // revert with zero value
    error ZeroValueInput(string value);
    // revert with mint/burn action outside of range
    error InvalidMintBurnBoundry(address token, uint256 amount);

    // status change to eXYZ
    event XYZUpdated(address token, bool validity, uint256 max, uint256 min);

    /************************************************
     *   initializers
     ************************************************/

    function __StablecoinHandler_init() public onlyInitializing {
        __StablecoinHandler_init_unchained();
    }

    function __StablecoinHandler_init_unchained() public onlyInitializing {
        __Pausable_init();
    }

    /************************************************
     *   modifiers
     ************************************************/

    /**
     * @notice Ensures all inputs to a stablecoin swap are non-zero
     * @param ss The stablecoin swap details including origin, target, and amounts.
     */
    modifier nonZero(StablecoinSwap memory ss) {
        if (ss.origin == address(0)) revert ZeroValueInput("ORIGIN CURRENCY");
        if (ss.oAmount == 0) revert ZeroValueInput("ORIGIN AMOUNT");
        if (ss.destination == address(0)) revert ZeroValueInput("DESTINATION");
        if (ss.target == address(0)) revert ZeroValueInput("TARGET CURRENCY");
        if (ss.tAmount == 0) revert ZeroValueInput("TARGET AMOUNT");
        _;
    }

    /************************************************
     *   swap function
     ************************************************/

    /**
     * @notice Initiates the stablecoin swap process, moving funds between origin and target currencies.
     * @dev Verifies the swap parameters and invokes the internal function to execute the swap.
     * This function is only accessible to accounts with the SWAPPER_ROLE and when the contract is not paused.
     * @param wallet The address of the user initiating the swap.
     * @param ss The StablecoinSwap structure containing the swap details (origin, target, amounts, fees, etc.).
     */
    function stablecoinSwap(
        address wallet,
        StablecoinSwap memory ss
    ) external onlyRole(SWAPPER_ROLE) whenNotPaused {
        // Verify the swap details before proceeding with the operation.
        _verifyStablecoinSwap(wallet, ss);
        // Perform the stablecoin swap after validation.
        _stablecoinSwap(wallet, ss);
    }

    /************************************************
     *   internal functions
     ************************************************/

    /**
     * @notice Handles the internal execution of the stablecoin swap, transferring tokens between the origin and target.
     * @dev This function manages fee deduction, burns/mints tokens when required, and performs transfers for both origin and target.
     * It supports both external XYZ tokens and other ERC20 tokens.
     * @param wallet The address of the user initiating the swap.
     * @param ss The StablecoinSwap structure with all the details for executing the swap.
     */
    function _stablecoinSwap(
        address wallet,
        StablecoinSwap memory ss
    ) internal {
        if (
            ss.stablecoinFeeCurrency != address(0) &&
            ss.stablecoinFeeSafe != address(0)
        )
            ERC20PermitUpgradeable(ss.stablecoinFeeCurrency).safeTransferFrom(
                wallet,
                ss.stablecoinFeeSafe,
                ss.feeAmount
            );

        // Handle the transfer or burning of the origin currency:
        // If the origin is a recognized stablecoin (XYZ), burn the specified amount from the wallet.
        if (isXYZ(ss.origin)) {
            Stablecoin(ss.origin).burnFrom(wallet, ss.oAmount);
        } else {
            ERC20PermitUpgradeable(ss.origin).safeTransferFrom(
                wallet,
                ss.liquiditySafe,
                ss.oAmount
            );
        }

        // Handle the minting or transferring of the target currency:
        // If the target is a recognized stablecoin (XYZ), mint the required amount to the destination address.
        if (isXYZ(ss.target)) {
            Stablecoin(ss.target).mintTo(ss.destination, ss.tAmount);
        } else {
            ERC20PermitUpgradeable(ss.target).safeTransferFrom(
                ss.liquiditySafe,
                ss.destination,
                ss.tAmount
            );
        }
    }

    /************************************************
     *   view functions
     ************************************************/

    /**
     * @notice Validates the parameters of the stablecoin swap, ensuring non-zero values and proper minting/burning limits.
     * @dev This function checks that all inputs are valid, verifies limits for minting and burning of XYZ tokens,
     * and confirms liquidity safe addresses for ERC20 tokens.
     * @param wallet The address of the user initiating the swap.
     * @param ss The StablecoinSwap structure with all swap details (origin, target, amounts, fees, etc.).
     */
    function _verifyStablecoinSwap(
        address wallet,
        StablecoinSwap memory ss
    ) internal view nonZero(ss) {
        // Ensure the wallet address is not zero.
        if (wallet == address(0)) revert ZeroValueInput("WALLET");

        // For the origin currency:
        if (isXYZ(ss.origin)) {
            // Ensure the total supply does not drop below the minimum limit after burning the specified amount.
            if (
                Stablecoin(ss.origin).totalSupply() - ss.oAmount <
                getMinLimit(ss.origin)
            ) revert InvalidMintBurnBoundry(ss.origin, ss.oAmount);
        } else if (ss.liquiditySafe == address(0)) {
            // Ensure the liquidity safe is provided for ERC20 origin tokens.
            revert ZeroValueInput("LIQUIDITY SAFE");
        }

        // For the target currency:
        if (isXYZ(ss.target)) {
            // Ensure the total supply does not exceed the maximum limit after minting the specified amount.
            if (
                Stablecoin(ss.target).totalSupply() + ss.tAmount >
                getMaxLimit(ss.target)
            ) revert InvalidMintBurnBoundry(ss.target, ss.tAmount);
        } else if (ss.liquiditySafe == address(0)) {
            // Ensure the liquidity safe is provided for ERC20 target tokens.
            revert ZeroValueInput("LIQUIDITY SAFE");
        }
    }

    /**
     * @notice Checks if a given token address is recognized as a valid external XYZ token.
     * @dev Reads from the contract's storage to determine the validity of the token address.
     * @param token The address of the token to check.
     * @return bool True if the token is a valid external XYZ token, false otherwise.
     */
    function isXYZ(address token) public view virtual returns (bool) {
        StablecoinHandlerStorage storage $ = _getStablecoinHandlerStorage();
        return $._eXYZs[token].validity;
    }

    /**
     * @notice Retrieves the maximum supply limit for a specified external XYZ token.
     * @dev Reads the maximum supply limit set for the token from the contract's storage.
     * This function provides visibility into the operational constraints of external XYZ tokens,
     * specifically the upper bound of the token's supply within the system.
     * @param token The address of the external XYZ token whose maximum supply limit is being queried.
     * @return uint256 The maximum supply limit for the specified token. This value represents the
     * upper limit on the total supply of the token that can be managed by the contract.
     */
    function getMaxLimit(address token) public view virtual returns (uint256) {
        StablecoinHandlerStorage storage $ = _getStablecoinHandlerStorage();
        return $._eXYZs[token].maxSupply;
    }

    /**
     * @notice Retrieves the minimum supply limit for a specified external XYZ token.
     * @dev Reads the minimum supply limit set for the token from the contract's storage.
     * This function is essential for understanding the operational constraints of external XYZ tokens,
     * highlighting the lower bound of the token's supply that is considered acceptable within the system.
     * @param token The address of the external XYZ token whose minimum supply limit is being queried.
     * @return uint256 The minimum supply limit for the specified token. This value indicates the
     * minimum amount of the token that should be maintained or is allowable within the contract's management scope.
     */
    function getMinLimit(address token) public view virtual returns (uint256) {
        StablecoinHandlerStorage storage $ = _getStablecoinHandlerStorage();
        return $._eXYZs[token].minSupply;
    }

    /************************************************
     *   support functions
     ************************************************/

    /**
     * @notice Updates the configuration for an external XYZ token.
     * @dev Modifies the validity status and supply limits of the specified token.
     * Can only be executed by addresses with the MAINTAINER_ROLE.
     * This method is crucial for maintaining the operational parameters of external XYZ tokens within the system.
     * @param token The address of the external XYZ token to update.
     * @param validity A boolean indicating whether the token should be considered valid.
     * @param maxLimit The maximum supply limit for the token.
     * @param minLimit The minimum supply limit for the token.
     *
     * Emits an `XYZUpdated` event upon successfully updating the token's parameters.
     */
    function UpdateXYZ(
        address token,
        bool validity,
        uint256 maxLimit,
        uint256 minLimit
    ) external virtual onlyRole(MAINTAINER_ROLE) {
        require(
            maxLimit > minLimit,
            "StablecoinHandler: upperbound must be greater than lowerbound"
        );

        StablecoinHandlerStorage storage $ = _getStablecoinHandlerStorage();
        $._eXYZs[token].validity = validity;
        $._eXYZs[token].maxSupply = maxLimit;
        $._eXYZs[token].minSupply = minLimit;
        emit XYZUpdated(token, validity, maxLimit, minLimit);
    }

    /**
     * @notice Pauses all pause-sensitive operations within the contract.
     * @dev Can only be called by addresses with the PAUSER_ROLE, halting certain functionalities.
     */
    function pause() external onlyRole(PAUSER_ROLE) {
        _pause();
    }

    /**
     * @notice Unpauses the contract, allowing previously paused operations to resume.
     * @dev Only callable by addresses with the PAUSER_ROLE, reenabling functionalities halted by pausing.
     */
    function unpause() external onlyRole(PAUSER_ROLE) {
        _unpause();
    }
}
