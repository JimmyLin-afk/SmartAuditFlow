// SPDX-License-Identifier: GPL-3.0-or-later
// solhint-disable gas-custom-errors, immutable-vars-naming
pragma solidity 0.8.18;

import { ERC20Burnable, ERC20 } from "openzeppelin-contracts-next/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import { Ownable } from "openzeppelin-contracts-next/contracts/access/Ownable.sol";
import { Pausable } from "openzeppelin-contracts-next/contracts/security/Pausable.sol";

/**
 * @title Mento Token
 * @author Mento Labs
 * @notice This contract represents the Mento Protocol Token which is a Burnable ERC20 token.
 */
contract MentoToken is Ownable, Pausable, ERC20Burnable {
  /// @notice The address of the locking contract that has the capability to transfer tokens
  /// even when the contract is paused.
  address public immutable locking;

  /// @notice The address of the emission contract that has the capability to emit new tokens
  /// and transfer even when the contract is paused.
  address public immutable emission;

  /// @notice The total amount of tokens that can be minted by the emission contract.
  uint256 public immutable emissionSupply;

  /// @notice The total amount of tokens that have been minted by the emission contract so far.
  uint256 public emittedAmount;

  // solhint-disable max-line-length
  /**
   * @dev Constructor for the MentoToken contract.
   * @notice It mints and allocates the initial token supply among several contracts.
   * @param allocationRecipients_ The addresses of the initial token recipients.
   * @param allocationAmounts_ The percentage of tokens to be allocated to each recipient.
   * @param emission_ The address of the emission contract where the rest of the supply will be emitted.
   * @param locking_ The address of the locking contract where the tokens will be locked.
   */
  // solhint-enable max-line-length
  constructor(
    address[] memory allocationRecipients_,
    uint256[] memory allocationAmounts_,
    address emission_,
    address locking_
  ) ERC20("Mento Token", "MENTO") Ownable() {
    require(emission_ != address(0), "MentoToken: emission is zero address");
    require(locking_ != address(0), "MentoToken: locking is zero address");
    require(
      allocationRecipients_.length == allocationAmounts_.length,
      "MentoToken: recipients and amounts length mismatch"
    );

    locking = locking_;
    emission = emission_;

    uint256 supply = 1_000_000_000 * 10 ** decimals();

    // slither-disable-next-line uninitialized-local
    uint256 totalAllocated;
    for (uint256 i = 0; i < allocationRecipients_.length; i++) {
      require(allocationRecipients_[i] != address(0), "MentoToken: allocation recipient is zero address");

      if (allocationAmounts_[i] == 0) continue;

      totalAllocated += allocationAmounts_[i];
      _mint(allocationRecipients_[i], (supply * allocationAmounts_[i]) / 1000);
    }
    require(totalAllocated <= 1000, "MentoToken: total allocation exceeds 100%");
    emissionSupply = (supply * (1000 - totalAllocated)) / 1000;

    _pause();
  }

  /**
   * @notice Unpauses all token transfers.
   * @dev See {Pausable-_unpause}
   * Requirements: caller must be the owner
   */
  function unpause() public virtual onlyOwner {
    require(paused(), "MentoToken: token is not paused");
    _unpause();
  }

  /**
   * @dev Allows the emission contract to mint new tokens up to the emission supply limit.
   * @notice This function can only be called by the emission contract and
   * only if the total emitted amount hasn't exceeded the emission supply.
   * @param target Address to which the newly minted tokens will be sent.
   * @param amount Amount of tokens to be minted.
   */
  function mint(address target, uint256 amount) external {
    require(msg.sender == emission, "MentoToken: only emission contract");
    require(emittedAmount + amount <= emissionSupply, "MentoToken: emission supply exceeded");

    emittedAmount += amount;
    _mint(target, amount);
  }

  /*
   * @dev See {ERC20-_beforeTokenTransfer}
   * Requirements: the contract must not be paused OR transfer must be initiated by owner
   * @param from The account that is sending the tokens
   * @param to The account that should receive the tokens
   * @param amount Amount of tokens that should be transferred
   */
  function _beforeTokenTransfer(address from, address to, uint256 amount) internal virtual override {
    super._beforeTokenTransfer(from, to, amount);

    require(to != address(this), "MentoToken: cannot transfer tokens to token contract");
    // Token transfers are only possible if the contract is not paused
    // OR if triggered by the owner of the contract
    // OR if triggered by the locking contract
    // OR if triggered by the emission contract
    require(
      !paused() || owner() == _msgSender() || locking == _msgSender() || emission == _msgSender(),
      "MentoToken: token transfer while paused"
    );
  }
}
