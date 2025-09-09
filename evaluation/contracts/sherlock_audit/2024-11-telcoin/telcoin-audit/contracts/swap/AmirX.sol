// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {AccessControlUpgradeable, SafeERC20, Stablecoin, StablecoinHandler} from "../stablecoin/StablecoinHandler.sol";
import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import {ISimplePlugin} from "./interfaces/ISimplePlugin.sol";

/**
 * @title AmirX
 * @author Amir M. Shirif
 * @notice A Telcoin Contract
 *
 * @dev Extends StablecoinHandler to implement a DeFi swap and fee buyback mechanism.
 * It facilitates token swaps and uses collected fees for buyback operations.
 */
contract AmirX is StablecoinHandler {
    using SafeERC20 for Stablecoin;
    using SafeERC20 for ERC20;

    struct DefiSwap {
        // Address for fee deposit
        address defiSafe;
        // Address of the swap aggregator or router
        address aggregator;
        // Plugin for handling referral fees
        ISimplePlugin plugin;
        // Token collected as fees
        ERC20 feeToken;
        // Address to receive referral fees
        address referrer;
        // Amount of referral fee
        uint256 referralFee;
        // Data for wallet interaction, if any
        bytes walletData;
        // Data for performing the swap, if any
        bytes swapData;
    }

    // Telcoin token address on Polygon network
    ERC20 public constant TELCOIN =
        ERC20(0xdF7837DE1F2Fa4631D716CF2502f8b230F1dcc32);
    // POL address on Polygon network
    address public constant POL = 0x0000000000000000000000000000000000001010;
    // Authorized Role
    bytes32 public constant SUPPORT_ROLE = keccak256("SUPPORT_ROLE");

    /************************************************
     *   initializer
     ************************************************/

    /**
     * @notice Initializes the contract and assigns the deployer as the default administrator.
     * @dev Grants the `DEFAULT_ADMIN_ROLE` to the deployer, allowing them to manage roles and perform
     * admin-level actions. Calls the initializer for `StablecoinHandler` to set up internal configurations.
     */
    function initialize() public initializer {
        _grantRole(DEFAULT_ADMIN_ROLE, _msgSender());
        __StablecoinHandler_init();
    }

    /************************************************
     *   swap functions
     ************************************************/

    /**
     * @notice Handles stablecoin swaps and triggers DeFi swap operations.
     * @dev Validates stablecoin swap parameters, performs swaps, and handles DeFi interactions based on provided DefiSwap details.
     * @param wallet Address initiating the swap.
     * @param directional dictates the direction of multiple swaps.
     * @param ss StablecoinSwap structure with swap details.
     * @param defi DefiSwap structure with DeFi swap details.
     */
    function swap(
        address wallet,
        bool directional,
        StablecoinSwap memory ss,
        DefiSwap memory defi
    ) external payable onlyRole(SWAPPER_ROLE) whenNotPaused {
        // checks if it will fail
        if (ss.destination != address(0)) _verifyStablecoinSwap(wallet, ss);
        if (defi.walletData.length != 0) _verifyDefiSwap(wallet, defi);

        if (directional) {
            // if only defi swap
            if (ss.destination == address(0)) _defiSwap(wallet, defi);
            else {
                // if defi then stablecoin swap
                //check balance to adjust second swap
                uint256 iBalance = ERC20(ss.origin).balanceOf(wallet);
                if (defi.walletData.length != 0) _defiSwap(wallet, defi);
                uint256 fBalance = ERC20(ss.origin).balanceOf(wallet);
                //change balance to reflect change
                if (fBalance - iBalance != 0) ss.oAmount = fBalance - iBalance;
                _stablecoinSwap(wallet, ss);
            }
        } else {
            // if stablecoin swap
            _stablecoinSwap(wallet, ss);
            // if only stablecoin swap
            if (defi.walletData.length != 0) _defiSwap(wallet, defi);
        }
    }

    /**
     * @notice performs a defiswap that feeds into a stablecoin swap
     * @dev Validates stablecoin swap parameters, performs swaps, and handles DeFi interactions based on provided DefiSwap details.
     * @param wallet Address initiating the swap.
     * @param ss StablecoinSwap structure with swap details.
     * @param defi DefiSwap structure with DeFi swap details.
     */
    function defiToStablecoinSwap(
        address wallet,
        StablecoinSwap memory ss,
        DefiSwap memory defi
    ) external payable onlyRole(SWAPPER_ROLE) whenNotPaused {
        // checks if defi will fail
        _verifyDefiSwap(wallet, defi);
        // checks if stablecoin swap will fail
        _verifyStablecoinSwap(wallet, ss);

        //check balance to adjust second swap
        uint256 iBalance = ERC20(ss.origin).balanceOf(wallet);
        _defiSwap(wallet, defi);
        uint256 fBalance = ERC20(ss.origin).balanceOf(wallet);
        ss.oAmount = fBalance - iBalance;
        //change balance to reflect change
        _stablecoinSwap(wallet, ss);
    }

    /**
     * @notice performs a stablecoin swap that feeds into a defiswap
     * @dev Validates stablecoin swap parameters, performs swaps, and handles DeFi interactions based on provided DefiSwap details.
     * @param wallet Address initiating the swap.
     * @param ss StablecoinSwap structure with swap details.
     * @param defi DefiSwap structure with DeFi swap details.
     */
    function stablecoinToDefiSwap(
        address wallet,
        StablecoinSwap memory ss,
        DefiSwap memory defi
    ) external payable onlyRole(SWAPPER_ROLE) whenNotPaused {
        // checks if it will fail
        _verifyStablecoinSwap(wallet, ss);
        _verifyDefiSwap(wallet, defi);

        _stablecoinSwap(wallet, ss);
        _defiSwap(wallet, defi);
    }

    /**
     * @notice Performs a DeFi swap using the provided DefiSwap details.
     * @dev Executes wallet transaction and fee dispersal as part of the swap process.
     * @param wallet Address initiating the swap.
     * @param defi DefiSwap structure with DeFi swap details.
     */
    function defiSwap(
        address wallet,
        DefiSwap memory defi
    ) external payable onlyRole(SWAPPER_ROLE) whenNotPaused {
        //revert if no user wallet
        if (wallet == address(0)) revert ZeroValueInput("WALLET");
        _verifyDefiSwap(wallet, defi);
        _defiSwap(wallet, defi);
    }

    function _defiSwap(address wallet, DefiSwap memory defi) internal {
        //user's interaction
        (bool walletResult, ) = wallet.call{value: 0}(defi.walletData);
        require(walletResult, "AmirX: wallet transaction failed");

        _feeDispersal(defi);
    }

    /************************************************
     *   internal functions
     ************************************************/

    /**
     * @notice Handles the dispersal of fees collected during a DeFi swap.
     * @dev Executes the buyback of fee tokens and handles referral fees if applicable.
     * @param defi DefiSwap structure with DeFi swap details.
     */
    function _feeDispersal(DefiSwap memory defi) internal {
        // must buy into TEL
        if (defi.feeToken != TELCOIN)
            _buyBack(
                defi.feeToken,
                defi.aggregator,
                defi.defiSafe,
                defi.swapData
            );

        // distribute reward
        if (defi.referrer != address(0) && defi.referralFee != 0) {
            TELCOIN.forceApprove(address(defi.plugin), 0);
            TELCOIN.safeIncreaseAllowance(
                address(defi.plugin),
                defi.referralFee
            );
            require(
                defi.plugin.increaseClaimableBy(
                    defi.referrer,
                    defi.referralFee
                ),
                "AmirX: balance was not adjusted"
            );
        }
        // retain remainder
        if (TELCOIN.balanceOf(address(this)) > 0)
            TELCOIN.safeTransfer(
                defi.defiSafe,
                TELCOIN.balanceOf(address(this))
            );
    }

    /**
     * @notice Performs a token buyback using the collected fees.
     * @dev Supports buyback for ERC20 tokens and POL, handling the swap via the specified aggregator.
     * @param feeToken The token to be bought back.
     * @param aggregator The swap aggregator address.
     * @param safe The fee destination.
     * @param swapData Data required to perform the swap.
     */
    function _buyBack(
        ERC20 feeToken,
        address aggregator,
        address safe,
        bytes memory swapData
    ) internal {
        if (address(feeToken) == address(0)) return;
        if (address(feeToken) == POL) {
            (bool polSwap, ) = aggregator.call{value: msg.value}(swapData);
            require(polSwap, "AmirX: POL swap transaction failed");

            if (address(this).balance > 0) {
                (bool success, ) = safe.call{value: address(this).balance}("");
                require(success, "AmirX: POL send transaction failed");
            }
        } else {
            // zero out approval
            feeToken.forceApprove(aggregator, 0);
            feeToken.safeIncreaseAllowance(
                aggregator,
                feeToken.balanceOf(address(this))
            );

            (bool ercSwap, ) = aggregator.call{value: 0}(swapData);
            require(ercSwap, "AmirX: token swap transaction failed");

            uint256 remainder = feeToken.balanceOf(address(this));
            if (remainder > 0) feeToken.safeTransfer(safe, remainder);
        }
    }

    /************************************************
     *   view functions
     ************************************************/

    /**
     * @notice Performs additional validations for the DefiSwap parameters.
     * @dev Ensures feeToken, aggregator, and swapData are valid for buyback operations.
     * @param defi DefiSwap structure with DeFi swap details.
     */
    function _verifyDefiSwap(
        address wallet,
        DefiSwap memory defi
    ) internal pure {
        if (wallet == address(0)) revert ZeroValueInput("WALLET");
        // validate pathway
        if (defi.feeToken != TELCOIN && address(defi.feeToken) != address(0)) {
            if (defi.aggregator == address(0) || defi.swapData.length == 0)
                revert ZeroValueInput("BUYBACK");
        }
        // determines if there is a referrer increase
        if (defi.referrer != address(0)) {
            if (address(defi.plugin) == address(0))
                revert ZeroValueInput("PLUGIN");
        }
    }

    /************************************************
     *   support functions
     ************************************************/

    /**
     * @notice Rescues crypto assets mistakenly sent to the contract.
     * @dev Allows for the recovery of both ERC20 tokens and native POL sent to the contract.
     * @param token The token to rescue.
     * @param amount The amount of the token to rescue.
     */
    function rescueCrypto(
        ERC20 token,
        uint256 amount
    ) external onlyRole(SUPPORT_ROLE) {
        if (address(token) != POL) {
            // ERC20s
            token.safeTransfer(_msgSender(), amount);
        } else {
            // POL
            (bool sent, ) = _msgSender().call{value: amount}("");
            require(sent, "AmirX: POL send failed");
        }
    }

    receive() external payable {}
}
