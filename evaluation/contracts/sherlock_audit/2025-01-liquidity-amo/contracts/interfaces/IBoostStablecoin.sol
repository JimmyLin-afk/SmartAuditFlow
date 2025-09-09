// SPDX-License-Identifier: MIT
pragma solidity 0.8.19;

import "@openzeppelin/contracts-upgradeable/interfaces/IERC20Upgradeable.sol";

interface IBoostStablecoin is IERC20Upgradeable {
    /**
     * @dev Destroys a `value` amount of tokens from the caller.
     *
     * See {ERC20-_burn}.
     */
    function burn(uint256 value) external;

    /**
     * @dev Destroys a `value` amount of tokens from `account`, deducting from
     * the caller's allowance.
     *
     * See {ERC20-_burn} and {ERC20-allowance}.
     *
     * Requirements:
     *
     * - the caller must have allowance for ``accounts``'s tokens of at least
     * `value`.
     */
    function burnFrom(address account, uint256 value) external;

    /**
     * @dev Mints a `amount` amount of tokens for `to`.
     *
     * Requirements:
     *
     * - Can only be called by an account with the MINTER_ROLE.
     */
    function mint(address to, uint256 amount) external;
}
