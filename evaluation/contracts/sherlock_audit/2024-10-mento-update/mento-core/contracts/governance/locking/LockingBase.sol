// SPDX-License-Identifier: MIT
pragma solidity 0.8.18;
// solhint-disable state-visibility, func-name-mixedcase, gas-custom-errors

import "openzeppelin-contracts-upgradeable/contracts/token/ERC20/IERC20Upgradeable.sol";
import "openzeppelin-contracts-upgradeable/contracts/access/OwnableUpgradeable.sol";
import "openzeppelin-contracts-upgradeable/contracts/governance/utils/IVotesUpgradeable.sol";
import "./libs/LibBrokenLine.sol";

/**
 * @title LockingBase
 * @dev This abstract contract provides the foundational functionality
 * for locking ERC20 tokens to accrue voting power.
 * @dev It utilizes the Broken Line library to represent the decay of voting power as tokens unlock.
 * @notice https://github.com/rarible/locking-contracts/tree/4f189a96b3e85602dedfbaf69d9a1f5056d835eb
 */
abstract contract LockingBase is OwnableUpgradeable, IVotesUpgradeable {
  using LibBrokenLine for LibBrokenLine.BrokenLine;
  /**
   * @dev Duration of a week in blocks on the CELO blockchain assuming 5 seconds per block
   */
  uint32 public constant WEEK = 120_960;
  /**
   * @dev Maximum allowable cliff period for token locks in weeks
   */
  uint32 constant MAX_CLIFF_PERIOD = 103;
  /**
   * @dev Maximum allowable slope period for token locks in weeks
   */
  uint32 constant MAX_SLOPE_PERIOD = 104;
  /**
   * @dev Basis for locking formula calculations
   */
  uint32 constant ST_FORMULA_BASIS = 1 * (10 ** 8);
  /**
   * @dev ERC20 token that will be locked
   */
  IERC20Upgradeable public token;
  /**
   * @dev Counter for Lock identifiers
   */
  uint256 public counter;
  /**
   * @dev Minimum cliff period in weeks
   */
  uint256 public minCliffPeriod;
  /**
   * @dev Minimum slope period in weeks
   */
  uint256 public minSlopePeriod;
  /**
   * @dev Starting point week for the locking week-based time system
   */
  uint256 public startingPointWeek;
  /**
   * @dev Struct used to represent a lock
   * account - Address owning the lock
   * delegate - Address that will receive the voting power from the locked tokens
   */
  struct Lock {
    address account;
    address delegate;
  }
  /**
   * @dev Mapping of lock identifiers to Lock structs
   */
  mapping(uint256 => Lock) locks;
  /**
   * @dev Struct used to represent an account's locked and unlocked token balances
   * balance - BrokenLine representing the linear function of the veMento balance
   * locked - BrokenLine representing the linear function of the locked token balance
   * amount - amount of locked tokens
   */
  struct Account {
    LibBrokenLine.BrokenLine balance;
    LibBrokenLine.BrokenLine locked;
    uint96 amount;
  }
  /**
   * @dev Mapping of addresses to Account structs
   */
  mapping(address => Account) accounts;
  /**
   * @dev Total supply line of veMento
   */
  LibBrokenLine.BrokenLine public totalSupplyLine;
  /**
   * @dev Emitted when create Lock with parameters (account, delegate, amount, slopePeriod, cliff)
   */
  event LockCreate(
    uint256 indexed id,
    address indexed account,
    address indexed delegate,
    uint256 time,
    uint256 amount,
    uint256 slopePeriod,
    uint256 cliff
  );
  /**
   * @dev Emitted when change Lock parameters (newDelegate, newAmount, newSlopePeriod, newCliff) for Lock with given id
   */
  event Relock(
    uint256 indexed id,
    address indexed account,
    address indexed delegate,
    uint256 counter,
    uint256 time,
    uint256 amount,
    uint256 slopePeriod,
    uint256 cliff
  );
  /**
   * @dev Emitted when to set newDelegate address for Lock with given id
   */
  event Delegate(uint256 indexed id, address indexed account, address indexed delegate, uint256 time);
  /**
   * @dev Emitted when withdraw amount of Rari, account - msg.sender, amount - amount Rari
   */
  event Withdraw(address indexed account, uint256 amount);
  /**
   * @dev set newMinCliffPeriod
   */
  event SetMinCliffPeriod(uint256 indexed newMinCliffPeriod);
  /**
   * @dev set newMinSlopePeriod
   */
  event SetMinSlopePeriod(uint256 indexed newMinSlopePeriod);

  /**
   * @dev Initializes the contract with token, starting point week, and minimum cliff and slope periods.
   * @param _token ERC20 token to be locked. (Mento Token)
   * @param _startingPointWeek Origin week number for the week-based time system.
   * @param _minCliffPeriod Minimum cliff period for locks.
   * @param _minSlopePeriod Minimum slope period for locks.
   */
  function __LockingBase_init_unchained(
    IERC20Upgradeable _token,
    uint32 _startingPointWeek,
    uint32 _minCliffPeriod,
    uint32 _minSlopePeriod
  ) internal onlyInitializing {
    token = _token;
    startingPointWeek = _startingPointWeek;

    //setting min cliff and slope
    require(_minCliffPeriod <= MAX_CLIFF_PERIOD, "cliff too big");
    require(_minSlopePeriod <= MAX_SLOPE_PERIOD, "period too big");
    minCliffPeriod = _minCliffPeriod;
    minSlopePeriod = _minSlopePeriod;
  }

  /**
   * @notice Adds a new locking line for an account, initializing the lock with specified parameters.
   * @param account  Address for which tokens are being locked.
   * @param _delegate Address that will receive the voting power from the locked tokens.
   * @param amount Amount of tokens to lock.
   * @param slopePeriod Period over which the tokens will unlock.
   * @param cliff Initial period during which tokens remain locked and do not start unlocking.
   * @param time Week number when the line is added.
   * @param currentBlock Current block number.
   */
  function addLines(
    address account,
    address _delegate,
    uint96 amount,
    uint32 slopePeriod,
    uint32 cliff,
    uint32 time,
    uint32 currentBlock
  ) internal {
    require(slopePeriod <= amount, "Wrong value slopePeriod");
    updateLines(account, _delegate, time);
    (uint96 stAmount, uint96 stSlope) = getLock(amount, slopePeriod, cliff);
    LibBrokenLine.Line memory line = LibBrokenLine.Line(time, stAmount, stSlope, cliff);
    totalSupplyLine.addOneLine(counter, line, currentBlock);
    accounts[_delegate].balance.addOneLine(counter, line, currentBlock);
    {
      uint96 slope = divUp(amount, slopePeriod);
      line = LibBrokenLine.Line(time, amount, slope, cliff);
    }
    accounts[account].locked.addOneLine(counter, line, currentBlock);
    locks[counter].account = account;
    locks[counter].delegate = _delegate;
  }

  /**
   * @notice Updates broken lines for account, delegate and total supply
   * @param account address of account that locked tokens
   * @param _delegate address of delegate that owns the voting power
   * @param time week number till which to update lines
   */
  function updateLines(address account, address _delegate, uint32 time) internal {
    totalSupplyLine.update(time);
    accounts[_delegate].balance.update(time);
    accounts[account].locked.update(time);
  }

  /**
   * @notice Calculates lockAmount and lockSlope for given lock parameters
   * @dev Сalculate and return (lockAmount, lockSlope), using formula:
   * P = t * min(c/c_max + s/s_max, 1),
   *
   * The formula has the following properties:
   * - the voting power can't exceed the amount of tokens locked.
   * - a voter can reach 100% voting power by relying on either the slope or the cliff,
   *   or a combination of both.
   * - there is a parameter space above a diagonal on the (c, s) plane where the
   *   voting power is capped at 100%, moving past that diagonal is disadvantageous
   *   but the contract doesn't forbid it.
   *
   *
   * The formula roughly translates to solidity as:
   * votingPower = (
   *   tokens *
   *   min(
   *    (ST_FORMULA_BASIS * cliffPeriod) / MAX_CLIFF_PERIOD +
   *    (ST_FORMULA_BASIS * slopePeriod) / MAX_SLOPE_PERIOD,
   *    ST_FORMULA_BASIS
   *   )
   * ) / ST_FORMULA_BASIS
   * @param amount of tokens to lock
   * @param slopePeriod period over which the tokens will unlock
   * @param cliff initial period during which tokens remain locked and do not start unlocking
   **/
  function getLock(
    uint96 amount,
    uint32 slopePeriod,
    uint32 cliff
  ) public view returns (uint96 lockAmount, uint96 lockSlope) {
    require(cliff >= minCliffPeriod, "cliff period < minimal lock period");
    require(slopePeriod >= minSlopePeriod, "slope period < minimal lock period");

    uint96 cliffSide = (uint96(cliff) * ST_FORMULA_BASIS) / MAX_CLIFF_PERIOD;
    uint96 slopeSide = (uint96(slopePeriod) * ST_FORMULA_BASIS) / MAX_SLOPE_PERIOD;
    uint96 multiplier = cliffSide + slopeSide;

    if (multiplier > ST_FORMULA_BASIS) {
      multiplier = ST_FORMULA_BASIS;
    }

    uint256 amountMultiplied = uint256(amount) * uint256(multiplier);
    lockAmount = uint96(amountMultiplied / (ST_FORMULA_BASIS));
    require(lockAmount > 0, "voting power is 0");
    lockSlope = divUp(lockAmount, slopePeriod);
  }

  /**
   * @notice Calculates a divided by b rounded up
   * @param a numerator
   * @param b denominator
   * @return ⌈a/b⌉
   */
  function divUp(uint96 a, uint96 b) internal pure returns (uint96) {
    return ((a - 1) / b) + 1;
  }

  /**
   * @notice Calculates the week number for a given blocknumber
   * @param ts block number
   * @return week number the block number belongs to
   */
  function roundTimestamp(uint32 ts) public view returns (uint32) {
    if (ts < getEpochShift()) {
      return 0;
    }
    uint32 shifted = ts - (getEpochShift());
    return shifted / WEEK - uint32(startingPointWeek);
  }

  /**
   * @notice method returns the amount of blocks to shift locking epoch to.
   * we move it to 00-00 UTC Wednesday (approx) by shifting 89964 blocks (CELO)
   */
  function getEpochShift() internal view virtual returns (uint32) {
    return 89964;
  }

  /**
   * @notice Verifies msg.sender is lock owner
   * @param id lock id to verify
   * @return account address of lock owner
   */
  function verifyLockOwner(uint256 id) internal view returns (address account) {
    account = locks[id].account;
    require(account == msg.sender, "caller not a lock owner");
  }

  /**
   * @notice Returns the current block number as a uint32
   * @return current block number
   */
  function getBlockNumber() internal view virtual returns (uint32) {
    return uint32(block.number);
  }

  /**
   * @notice Sets the minimum cliff period
   * @param newMinCliffPeriod new minimum cliff period
   */
  function setMinCliffPeriod(uint32 newMinCliffPeriod) external onlyOwner {
    require(newMinCliffPeriod <= MAX_CLIFF_PERIOD, "new cliff period > 2 years");
    minCliffPeriod = newMinCliffPeriod;

    emit SetMinCliffPeriod(newMinCliffPeriod);
  }

  /**
   * @notice Sets the minimum slope period
   * @param newMinSlopePeriod new minimum slope period
   */
  function setMinSlopePeriod(uint32 newMinSlopePeriod) external onlyOwner {
    require(newMinSlopePeriod <= MAX_SLOPE_PERIOD, "new slope period > 2 years");
    minSlopePeriod = newMinSlopePeriod;

    emit SetMinSlopePeriod(newMinSlopePeriod);
  }

  /**
   * @notice Updates the broken lines for an account until a given week number
   * @param account address of account to update
   * @param time week number until which to update lines
   */
  function updateAccountLines(address account, uint32 time) public onlyOwner {
    accounts[account].balance.update(time);
    accounts[account].locked.update(time);
  }

  /**
   * @notice updates the total supply line until a given week number
   * @param time week number until which to update lines
   */
  function updateTotalSupplyLine(uint32 time) public onlyOwner {
    totalSupplyLine.update(time);
  }

  /**
   * @notice updates the broken lines for an account until a given block number
   * @param account address of account to update
   * @param blockNumber block number until which to update lines
   */
  function updateAccountLinesBlockNumber(address account, uint32 blockNumber) external onlyOwner {
    uint32 time = roundTimestamp(blockNumber);
    updateAccountLines(account, time);
  }

  /**
   * @notice Updates the total supply line until a given block number
   * @param blockNumber block number until which to update line
   */
  function updateTotalSupplyLineBlockNumber(uint32 blockNumber) external onlyOwner {
    uint32 time = roundTimestamp(blockNumber);
    updateTotalSupplyLine(time);
  }

  uint256[50] private __gap;
}
