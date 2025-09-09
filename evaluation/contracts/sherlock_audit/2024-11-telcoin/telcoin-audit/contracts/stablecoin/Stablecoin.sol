// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {ERC20PermitUpgradeable} from "@openzeppelin/contracts-upgradeable/token/ERC20/extensions/ERC20PermitUpgradeable.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import {StorageSlot} from "@openzeppelin/contracts/utils/StorageSlot.sol";
import {Blacklist} from "../util/abstract/Blacklist.sol";

/**
 * @title Stablecoin
 * @author Amir M. Shirif
 * @notice A Telcoin Contract
 *
 * @notice This is an ERC20 standard coin with advanced capabilities to allow for
 * minting and burning. This coin is pegged to a fiat currency and its value is
 * intended to reflect the value of its native currency
 * @dev Blacklisting has been included to prevent this currency from being used for illicit or nefarious activities
 */
contract Stablecoin is ERC20PermitUpgradeable, Blacklist {
    using SafeERC20 for ERC20PermitUpgradeable;

    /**
     * @dev Storage slot with the address of the current implementation.
     * This is the keccak-256 hash of "eip1967.telcoin.Stablecoin.decimals" subtracted by 1.
     */
    // solhint-disable-next-line private-vars-leading-underscore
    bytes32 internal constant DECIMALS_SLOT =
        0x86386409a65c1a7f963bc51852fa7ecbdb9cad2cec464de22ee4591e1622b46b;

    //Authorized Roles
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant BURNER_ROLE = keccak256("BURNER_ROLE");
    bytes32 public constant SUPPORT_ROLE = keccak256("SUPPORT_ROLE");

    /************************************************
     *   initializer
     ************************************************/

    /**
     * @notice initializes the contract
     * @dev this function is called with proxy deployment to update state data
     * @dev uses initializer modifier to only allow one initialization per proxy
     * @param name_ is a string representing the token name
     * @param symbol_ is a string representing the token symbol
     * @param decimals_ is an int representing the number of decimals for the token
     */
    function initialize(
        string memory name_,
        string memory symbol_,
        uint8 decimals_
    ) external initializer {
        __ERC20_init(name_, symbol_);
        __ERC20Permit_init(name_);
        _grantRole(DEFAULT_ADMIN_ROLE, _msgSender());
        StorageSlot.getUint256Slot(DECIMALS_SLOT).value = decimals_;
    }

    /**
     * @notice Returns the number of decimal places
     */
    function decimals() public view override returns (uint8) {
        return uint8(StorageSlot.getUint256Slot(DECIMALS_SLOT).value);
    }

    /************************************************
     *   supply functions
     ************************************************/

    /**
     * @notice Mints `value` tokens to the caller's account.
     * @dev Only accounts with MINTER_ROLE can call this function.
     * @param value The amount of tokens to mint.
     */
    function mint(uint256 value) public onlyRole(MINTER_ROLE) {
        _mint(_msgSender(), value);
    }

    /**
     * @notice Mints `value` tokens to a specified `account`.
     * @dev Only accounts with MINTER_ROLE can call this function.
     * @param account The account to which tokens will be minted.
     * @param value The amount of tokens to mint.
     */
    function mintTo(
        address account,
        uint256 value
    ) public onlyRole(MINTER_ROLE) {
        _mint(account, value);
    }

    /**
     * @notice Burns `value` tokens from the caller's account.
     * @dev Only accounts with BURNER_ROLE can call this function.
     * @param value The amount of tokens to burn.
     */
    function burn(uint256 value) public onlyRole(BURNER_ROLE) {
        _burn(_msgSender(), value);
    }

    /**
     * @notice Burns `value` tokens from a specified `account`, deducting from the caller's allowance.
     * @dev Only accounts with BURNER_ROLE can call this function.
     * @param account The account from which tokens will be burned.
     * @param value The amount of tokens to burn.
     */
    function burnFrom(
        address account,
        uint256 value
    ) public onlyRole(BURNER_ROLE) {
        _spendAllowance(account, _msgSender(), value);
        _burn(account, value);
    }

    /************************************************
     *   internal functions
     ************************************************/

    /**
     * @notice Overrides Blacklist function to transfer balance of a blacklisted user to the caller.
     * @dev This function is called internally when an account is blacklisted.
     * @param user The blacklisted user whose balance will be transferred.
     */
    function _onceBlacklisted(address user) internal override {
        _transfer(user, _msgSender(), balanceOf(user));
    }

    function _update(
        address from,
        address to,
        uint256 value
    ) internal override {
        if (blacklisted(from)) revert Blacklisted(from);
        if (blacklisted(to)) revert Blacklisted(to);

        super._update(from, to, value);
    }

    /************************************************
     *   support functions
     ************************************************/

    /**
     * @notice sends tokens accidently sent to contract
     * @dev restricted to SUPPORT_ROLE
     * @param token currency stuck in contract
     * @param destination address where funds are returned
     * @param amount is the amount being transferred
     */
    function erc20Rescue(
        ERC20PermitUpgradeable token,
        address destination,
        uint256 amount
    ) external onlyRole(SUPPORT_ROLE) {
        token.safeTransfer(destination, amount);
    }
}
