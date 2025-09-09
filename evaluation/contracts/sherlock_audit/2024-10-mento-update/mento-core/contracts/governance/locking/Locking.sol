// SPDX-License-Identifier: MIT
pragma solidity 0.8.18;
// solhint-disable func-name-mixedcase, gas-custom-errors

import "./LockingBase.sol";
import "./LockingRelock.sol";
import "./LockingVotes.sol";
import "./interfaces/ILocking.sol";

/**
 * @title Locking
 * @notice Implements locking mechanism for tokens to enable voting power accumulation
 * @notice https://github.com/rarible/locking-contracts/tree/4f189a96b3e85602dedfbaf69d9a1f5056d835eb
 */
contract Locking is ILocking, LockingBase, LockingRelock, LockingVotes {
  using LibBrokenLine for LibBrokenLine.BrokenLine;

  /**
   * @notice Initializes the locking contract.
   * @dev Sets up the base locking parameters and initializes ownership and context setup
   * @param _token Address of the ERC20 that will be locked. (Mento Token)
   * @param _startingPointWeek Origin week number for the week-based time system
   * @param _minCliffPeriod Minimum cliff period for locks
   * @param _minSlopePeriod Minimum slope period for locks
   */
  function __Locking_init(
    IERC20Upgradeable _token,
    uint32 _startingPointWeek,
    uint32 _minCliffPeriod,
    uint32 _minSlopePeriod
  ) external initializer {
    __LockingBase_init_unchained(_token, _startingPointWeek, _minCliffPeriod, _minSlopePeriod);
    __Ownable_init_unchained();
    __Context_init_unchained();
  }

  /**
   * @notice Locks a specified amount of tokens for a given period
   * @dev Delegate is not optional, it can be set to the lock owner if no delegate is desired
   * @param account Account for which tokens are being locked
   * @param _delegate Address that will receive the voting power from the locked tokens
   * @param amount Amount of tokens to lock
   * @param slopePeriod Period over which the tokens will unlock
   * @param cliff Initial period during which tokens remain locked and do not start unlocking
   * @return Id for the created lock
   */
  function lock(
    address account,
    address _delegate,
    uint96 amount,
    uint32 slopePeriod,
    uint32 cliff
  ) external override returns (uint256) {
    require(amount >= 1e18, "amount is less than minimum");
    require(cliff <= MAX_CLIFF_PERIOD, "cliff too big");
    require(slopePeriod <= MAX_SLOPE_PERIOD, "period too big");
    require(account != address(0), "account is zero");
    require(_delegate != address(0), "delegate is zero");

    counter++;

    uint32 currentBlock = getBlockNumber();
    uint32 time = roundTimestamp(currentBlock);
    addLines(account, _delegate, amount, slopePeriod, cliff, time, currentBlock);
    accounts[account].amount = accounts[account].amount + (amount);

    // slither-disable-next-line reentrancy-events
    require(token.transferFrom(msg.sender, address(this), amount), "transfer failed");

    emit LockCreate(counter, account, _delegate, time, amount, slopePeriod, cliff);
    return counter;
  }

  /**
   * @notice Withdraws unlocked tokens for the caller
   */
  function withdraw() external {
    uint96 value = getAvailableForWithdraw(msg.sender);
    if (value > 0) {
      accounts[msg.sender].amount = accounts[msg.sender].amount - (value);
      // slither-disable-next-line reentrancy-events
      require(token.transfer(msg.sender, value), "transfer failed");
    }
    emit Withdraw(msg.sender, value);
  }

  /**
   * @notice Calculates the amount available for withdrawal by an account
   * @param account The account to check the withdrawable amount for
   * @return The amount of tokens available for withdrawal
   */
  function getAvailableForWithdraw(address account) public view returns (uint96) {
    uint96 value = accounts[account].amount;
    uint32 currentBlock = getBlockNumber();
    uint32 time = roundTimestamp(currentBlock);
    uint96 bias = accounts[account].locked.actualValue(time, currentBlock);
    value = value - (bias);
    return value;
  }

  /**
   * @notice Returns the total amount of tokens locked for an account
   * @param account The account to check locked amount for
   * @return The locked amount for the account
   */
  function locked(address account) external view returns (uint256) {
    return accounts[account].amount;
  }

  /**
   * @notice Retrieves the account and delegate associated with a given lock ID
   * @param id The id of the lock
   * @return _account The account that owns the lock
   * @return _delegate The account that owns the voting power
   */
  function getAccountAndDelegate(uint256 id) external view returns (address _account, address _delegate) {
    _account = locks[id].account;
    _delegate = locks[id].delegate;
  }

  /**
   * @notice Returns "current week" of the contract. The Locking contract works with a week-based time system
   * for managing locks and voting power. The current week number is calculated based on the number of weeks passed
   * since the starting point week. The starting point is set during the contract initialization.
   */
  function getWeek() external view returns (uint256) {
    return roundTimestamp(getBlockNumber());
  }

  /**
   * @notice Changes the delegate for a specific lock
   * @dev Updates the delegation and adjusts the voting power accordingly
   * @param id The unique identifier for the lock whose delegate is to be changed
   * @param newDelegate The address to which the delegation will be transferred
   */
  function delegateTo(uint256 id, address newDelegate) external {
    require(newDelegate != address(0), "delegate is zero");

    address account = verifyLockOwner(id);
    address _delegate = locks[id].delegate;
    uint32 currentBlock = getBlockNumber();
    uint32 time = roundTimestamp(currentBlock);
    accounts[_delegate].balance.update(time);
    (uint96 bias, uint96 slope, uint32 cliff) = accounts[_delegate].balance.remove(id, time, currentBlock);
    LibBrokenLine.Line memory line = LibBrokenLine.Line(time, bias, slope, cliff);
    accounts[newDelegate].balance.update(time);
    accounts[newDelegate].balance.addOneLine(id, line, currentBlock);
    locks[id].delegate = newDelegate;
    emit Delegate(id, account, newDelegate, time);
  }

  /**
   * @notice Returns the current total supply of veMENTO tokens
   * @return The total supply of veMENTO tokens
   */
  function totalSupply() external view returns (uint256) {
    if (totalSupplyLine.initial.bias == 0) {
      return 0;
    }
    uint32 currentBlock = getBlockNumber();
    uint32 time = roundTimestamp(currentBlock);
    return totalSupplyLine.actualValue(time, currentBlock);
  }

  /**
   * @notice Retrieves the veMENTO balance of an account
   * @param account The account to check the balance for
   * @return The accounts balance of veMENTO tokens
   */
  function balanceOf(address account) external view returns (uint256) {
    if (accounts[account].balance.initial.bias == 0) {
      return 0;
    }
    uint32 currentBlock = getBlockNumber();
    uint32 time = roundTimestamp(currentBlock);
    return accounts[account].balance.actualValue(time, currentBlock);
  }

  /**
   * @notice Returns the name of the token
   */
  function name() public view virtual returns (string memory) {
    return "Mento Vote-Escrow";
  }

  /**
   * @notice Returns the symbol of the token
   */
  function symbol() public view virtual returns (string memory) {
    return "veMENTO";
  }

  /**
   * @notice Returns the decimal points of the token
   */
  function decimals() public view virtual returns (uint8) {
    return 18;
  }

  uint256[50] private __gap;
}
