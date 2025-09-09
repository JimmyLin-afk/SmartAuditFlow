// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title ISimplePlugin
 * @notice Interface for a plugin contract used in the AmirX ecosystem.
 *
 * @dev Provides an interface for handling claimable amounts, typically for managing referral or reward balances.
 * Contracts implementing this interface should define the logic for increasing the claimable amount for an account.
 */
interface ISimplePlugin {
    /**
     * @notice Increases the claimable amount for a specific account.
     * @dev Called to adjust the claimable balance for an account, typically after certain operations like swaps.
     * @param account The address of the account to increase claimable amount.
     * @param amount The amount by which to increase the claimable balance.
     * @return bool Returns true if the claimable balance was successfully increased.
     */
    function increaseClaimableBy(
        address account,
        uint256 amount
    ) external returns (bool);
}
