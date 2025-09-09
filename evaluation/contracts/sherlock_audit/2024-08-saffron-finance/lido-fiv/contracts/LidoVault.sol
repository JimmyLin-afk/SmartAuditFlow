// SPDX-License-Identifier: MIT

pragma solidity 0.8.18;

import "hardhat/console.sol";

import "@openzeppelin/contracts/utils/math/Math.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

import "./interfaces/ILidoVault.sol";
import "./interfaces/ILido.sol";
import "./interfaces/ILidoWithdrawalQueueERC721.sol";
import "./interfaces/ILidoVaultInitializer.sol";

/// @title Saffron Fixed Income Lido Vault
contract LidoVault is ILidoVaultInitializer, ILidoVault {
  using Math for uint256;
  using SafeERC20 for IERC20;

  /// @notice Vault ID set by the factory
  uint256 public id;

  /// @notice Length of the earning period of the vault in seconds
  uint256 public duration;

  /// @notice blocks the initialize() call to the contract received from vaultFactory
  bool private isFactoryCreated;

  /// @notice Start of vault
  /// @dev Calculated when vault starts via block.timestamp
  uint256 public startTime;

  /// @notice End of duration
  /// @dev Calculated when vault starts via (block.timestamp + duration)
  uint256 public endTime;

  /// @notice Total capacity of the fixed side
  uint256 public fixedSideCapacity;

  /// @notice Total capacity of the variable side
  uint256 public variableSideCapacity;

  /// @notice Saffron protocol fee in basis points
  uint256 public protocolFeeBps;

  /// @notice ETH amount that tracks the Saffron protocol fee applied to withdrawn Lido staking earnings. It is reset when the fee reciever withdraws
  uint256 public appliedProtocolFee;

  /// @notice ETH amount that tracks the total Saffron protocol fee from variable withdrawal staking earnings while the vault is still ongoing
  uint256 public totalProtocolFee;

  /// @notice Address that collects the Saffron protocol fee
  address public protocolFeeReceiver;

  /// @notice Penalty fee in basis points for fixed side early withdrawals that is quadratically scaled based off of the amount of time that has elapsed since the vault started
  uint256 public earlyExitFeeBps;

   /// @notice Total stETH capacity of the fixed side on vault start
  uint256 public fixedSidestETHOnStartCapacity;

  /// @notice Minimum amount of ETH that can be deposited for variable or fixed side users
  uint256 public immutable minimumDepositAmount = 0.01 ether;

  /// @notice Minimum amount of the fixed capacity that can be deposited for fixed side users on a single deposit in basis points
  uint256 public immutable minimumFixedDepositBps = 500; // default 5%

  /// @notice Mock ERC20 bearer token that entitles owner to a portion of the fixed side bearer tokens and the variable side premium payment
  /// It represents the amount of StETH shares generated from Lido staking submit on fixed side deposits
  /// @dev If the vault hasn't started, this is used to return the fixed side deposit
  mapping(address => uint256) public fixedClaimToken;
  uint256 public fixedClaimTokenTotalSupply;

  /// @notice Mock ERC20 bearer token that entitles owner to a portion of the fixed side deposits after the vault has ended and the ETH has been unstaked from Lido
  mapping(address => uint256) public fixedBearerToken;
  uint256 public fixedBearerTokenTotalSupply;

  /// @notice Mock ERC20 bearer token that entitles owner to a portion of the fixed side bearer tokens and the variable side premium payment
  /// It represents total amount of StETH shares generated from Lido staking submit on fixed side deposits
  uint256 public fixedIntialTokenTotalSupply;

  /// @notice Mock ERC20 bearer token that tracks fixed user's ETH deposits in order to determine when the fixed capacity is reached
  mapping(address => uint256) public fixedETHDepositToken;
  uint256 public fixedETHDepositTokenTotalSupply;

  /// @notice Mock ERC20 bearer token that entitles owner to a portion of the vault earnings
  mapping(address => uint256) public variableBearerToken;
  uint256 public variableBearerTokenTotalSupply;

  uint256 private constant FIXED = 0;
  uint256 private constant VARIABLE = 1;

  /// @notice Variable to block smart contract replenishment
  bool private unlockReceive = false;

  /// @notice Amount of earnings from fixed side early exit fees
  uint256 public feeEarnings;

  /// @notice Fee earnings that have been withdrawn from the variable side
  uint256 public withdrawnFeeEarnings;

  /// @notice Amount of earnings from Lido Staking in lido stakes already withdrawn by variable depositors
  uint256 public withdrawnStakingEarningsInStakes;

  /// @notice Amount of earnings from Lido Staking in ETH already withdrawn by variable depositors
  uint256 public withdrawnStakingEarnings;

  /// @notice Lido withdraw request ids for after the vault has ended
  uint256[] public vaultEndedWithdrawalRequestIds;

  /// @notice The vault ended Lido withdrawal requests have been claimed and all the staked Lido ETH has been withdrawn
  /// @dev True if the remaining Lido ETH funds have been withdrawn from Lido
  bool public vaultEndedWithdrawalsFinalized;

  /// @notice Amount of earnings in ETH from Lido Staking after the vault has ended
  uint256 public vaultEndedStakingEarnings;

  /// @notice Amount of lido stETH stakes in Lido Vault at the moment of vault has ending
  uint256 public vaultEndingStakesAmount;

  /// @notice Balance of Lido Vault in ETH at the moment of vault has ending
  uint256 public vaultEndingETHBalance;

  /// @notice Amount of ETH used to cover the returning of fixed user's initial principal
  uint256 public vaultEndedFixedDepositsFunds;

  /// @notice Mapping from user addresses to the upfront premium a fixed depositor received from the variable side
  mapping(address => uint256) public userToFixedUpfrontPremium;

  /// @notice Mapping from user addresses to their withdrawal request ids before the vault has started
  mapping(address => uint256[]) public fixedToVaultNotStartedWithdrawalRequestIds;

  /// @notice Struct to store the Lido withdrawal requestIds and the timestamp at which the withdrawal was requested - this will be used to calculate the early exit penalty fees
  struct WithdrawalRequest {
    uint256 timestamp;
    uint256[] requestIds;
  }

  /// @notice Mapping from fixed side user addresses to their withdrawal request ids after the vault has started but before the vault has ended
  mapping(address => WithdrawalRequest) public fixedToVaultOngoingWithdrawalRequestIds;

  /// @notice Add getter functions for each field in the struct b/c the compiler doesn't know how to construct the "free" getter that usually comes with public
  function getFixedOngoingWithdrawalRequestTimestamp(address user) public view returns (uint256) {
    return fixedToVaultOngoingWithdrawalRequestIds[user].timestamp;
  }

  function getFixedOngoingWithdrawalRequestIds(address user) public view returns (uint256[] memory) {
    return fixedToVaultOngoingWithdrawalRequestIds[user].requestIds;
  }

  function getFixedToVaultNotStartedWithdrawalRequestIds(address user) public view returns (uint256[] memory) {
    return fixedToVaultNotStartedWithdrawalRequestIds[user];
  }

  function getVariableToVaultOngoingWithdrawalRequestIds(address user) public view returns (uint256[] memory) {
    return variableToVaultOngoingWithdrawalRequestIds[user];
  }

  /// @notice Store array of all fixed users with pending ongoing withdrawals to iterate through fixedToVaultOngoingWithdrawalRequestIds mapping if the vault ends
  address[] public fixedOngoingWithdrawalUsers;

  /// @notice When the vault ends ongoing fixed withdrawals may be claimed by another user
  /// this stores the amount they should be given when the user eventually withdraws
  mapping(address => uint256) public fixedToPendingWithdrawalAmount;

  /// @notice Mapping from variable side user addresses to their withdrawal amount after the vault has started
  /// this stores the amount they should be given when the user eventually withdraws
  mapping(address => uint256) public variableToPendingWithdrawalAmount;

  /// @notice Mapping from variable side user addresses to their withdrawal request ids after the vault has started but before the vault has ended
  mapping(address => uint256[]) public variableToVaultOngoingWithdrawalRequestIds;

  /// @notice Mapping from variable side user addresses to their total withdrawn staking earnings amount right after their Lido withdrawal request is submitted
  mapping(address => uint256) public variableToWithdrawnStakingEarnings;

  /// @notice Mapping from variable side user addresses to their total withdrawn staking earnings amount in lido shares right after their Lido withdrawal request is submitted
  mapping(address => uint256) public variableToWithdrawnStakingEarningsInShares;

  /// @notice Mapping from variable side user addresses to their total payed protocol fee amount right after their Lido withdrawal request is submitted
  mapping(address => uint256) public variableToWithdrawnProtocolFee;

  /// @notice Mapping from variable side user addresses to their total withdrawn fees amount right after their Lido withdrawal request is claimed
  mapping(address => uint256) public variableToWithdrawnFees;

  /// @notice Emitted when a new vault is initialized
  /// @param vaultId ID of vault
  /// @param duration How long the vault will be locked once started, in seconds
  /// @param variableSideCapacity Maximum capacity of variable side
  /// @param fixedSideCapacity Maximum capacity of fixed side
  /// @param earlyExitFeeBps Fixed depositor early exit fee in basis points
  /// @param protocolFeeBps Protocol fee in basis points
  /// @param protocolFeeReceiver Address that collects protocol fee
  event VaultInitialized(
    uint256 vaultId,
    uint256 duration,
    uint256 variableSideCapacity,
    uint256 fixedSideCapacity,
    uint256 earlyExitFeeBps,
    uint256 protocolFeeBps,
    address protocolFeeReceiver
  );

  /// @notice Emitted when variable funds are deposited into the vault
  /// @param amount ETH amount deposited
  /// @param user Address of user
  event VariableFundsDeposited(uint256 amount, address indexed user);

  /// @notice Emitted when fixed funds are deposited into the vault
  /// @param amount ETH amount deposited
  /// @param shares stETH shares issued for Lido submit
  /// @param user Address of user
  event FixedFundsDeposited(uint256 amount, uint256 shares, address indexed user);

  /// @notice Emitted when a fixed user claims their upfront premium
  /// @param premium Fixed upfront premium transferred to the fixed user
  /// @param shares stETH shares claimed
  /// @param user Address of user
  event FixedPremiumClaimed(uint256 premium, uint256 shares, address indexed user);

  /// @notice Emitted when variable funds are withdrawn from the vault
  /// @param amount Amount withdrawn
  /// @param user Address of user
  /// @param isStarted Indicates whether the vault has started when the funds were withdrawn
  /// @param isEnded Indicates whether the vault has ended when the funds were withdrawn
  event VariableFundsWithdrawn(uint256 amount, address indexed user, bool indexed isStarted, bool indexed isEnded);

  /// @notice Emitted when fixed funds are withdrawn from the vault
  /// @param amount Amount withdrawn
  /// @param user Address of user
  /// @param isStarted Indicates whether the vault has started when the funds were withdrawn
  /// @param isEnded Indicates whether the vault has ended when the funds were withdrawn
  event FixedFundsWithdrawn(uint256 amount, address indexed user, bool indexed isStarted, bool indexed isEnded);

  /// @notice Emitted when the vault capacities have been met and the vault has been moved into the started phase
  /// @param timeStarted Time the vault started
  /// @param user Address of user that triggered the start of the vault
  event VaultStarted(uint256 timeStarted, address indexed user);

  /// @notice Emitted when the vault has passed its end time and moved into the ended phase
  /// @param timeEnded Time the vault ended
  /// @param user Address of user that triggered the end of the vault
  event VaultEnded(uint256 timeEnded, address indexed user);

  /// @notice Emitted when stakings funds are requested to be withdrawn from Lido
  /// @param user Address of user
  /// @param requestIds Request ids of the Lido withdrawal requests
  /// @param side Indicates fixed or variable side requested
  /// @param isStarted Indicates whether the vault has started when the funds were requested to be withdrawn
  /// @param isEnded Indicates whether the vault has ended when the funds were requested to be withdrawn
  event LidoWithdrawalRequested(
    address indexed user,
    uint256[] requestIds,
    uint256 side,
    bool indexed isStarted,
    bool indexed isEnded
  );

  /// @notice Emitted when a Lido withdrawal is claimed and the ETH is transferred to the adapter contract
  /// @param user Address of user
  /// @param requestIds Request ids of the Lido withdrawal requests
  /// @param side Indicates fixed or variable side requested
  /// @param isStarted Indicates whether the vault has started when the funds were requested to be withdrawn
  /// @param isEnded Indicates whether the vault has ended when the funds were submitted for withdrawal
  event LidoWithdrawalFinalized(
    address indexed user,
    uint256[] requestIds,
    uint256 side,
    bool indexed isStarted,
    bool indexed isEnded
  );

  constructor(bool _initialize) {
    isFactoryCreated = _initialize;
  }

  /// @inheritdoc ILidoVault
  function initialize(InitializationParams memory params) external {
    require(isFactoryCreated != true, "MBF");
    // Only run once by vault factory in atomic operation right after cloning, then fixedSideCapacity is set
    require(fixedSideCapacity == 0, "ORO");

    // Validate args
    require(params.vaultId != 0, "NEI");
    require(params.duration != 0, "NEI");
    require(params.fixedSideCapacity != 0, "NEI");
    require(params.variableSideCapacity != 0, "NEI");
    require(params.earlyExitFeeBps != 0, "NEI");
    require(params.protocolFeeReceiver != address(0), "NEI");

    require(params.fixedSideCapacity.mulDiv(minimumFixedDepositBps, 10_000) >= minimumDepositAmount, "IFC");

    // Initialize contract state variables
    id = params.vaultId;
    duration = params.duration;
    fixedSideCapacity = params.fixedSideCapacity;
    variableSideCapacity = params.variableSideCapacity;
    earlyExitFeeBps = params.earlyExitFeeBps;
    protocolFeeBps = params.protocolFeeBps;
    protocolFeeReceiver = params.protocolFeeReceiver;

    emit VaultInitialized(
      id,
      duration,
      variableSideCapacity,
      fixedSideCapacity,
      earlyExitFeeBps,
      protocolFeeBps,
      protocolFeeReceiver
    );
  }

  /// @notice Function to receive Ether. msg.data must be empty
  receive() external payable {
    require(unlockReceive, "LRC");
  }

  /// @notice True if the vault has started
  function isStarted() public view returns (bool) {
    return startTime > 0;
  }

  /// @notice True if the vault has ended
  function isEnded() public view returns (bool) {
    return (endTime > 0 && block.timestamp > endTime);
  }

  /// @notice Deposit ETH into the vault
  /// @param side ID of side to deposit into
  function deposit(uint256 side) external payable {
    // fixedSideCapacity will not be zero of initialized
    require(fixedSideCapacity != 0, "NI");
    require(!isStarted(), "DAS");
    require(side == FIXED || side == VARIABLE, "IS");
    require(msg.value >= minimumDepositAmount, "MDA");

    uint256 amount = msg.value;
    if (side == FIXED) {
      // Fixed side deposits

      uint256 minimumFixedDeposit = fixedSideCapacity.mulDiv(minimumFixedDepositBps, 10_000);
      require(amount >= minimumFixedDeposit, "MFD");

      // no refunds allowed
      require(amount <= fixedSideCapacity - fixedETHDepositTokenTotalSupply, "OED");
      // do not allow remaining capacity to be less than minimum fixed deposit bps
      uint256 remainingCapacity = fixedSideCapacity - fixedETHDepositTokenTotalSupply - amount;
      require(remainingCapacity == 0 || remainingCapacity >= minimumFixedDeposit, "RC");

      // Stake on Lido
      /// returns stETH, and returns amount of Lido shares issued for the staked ETH
      uint256 stETHBalanceBefore = stakingBalance();
      uint256 shares = lido.submit{value: amount}(address(0)); // _referral address argument is optional use zero address
      require(shares > 0, "ISS");
      // stETH transfered from Lido != ETH deposited to Lido - some rounding error
      uint256 stETHReceived = (stakingBalance() - stETHBalanceBefore);
      require((stETHReceived >= amount) || (amount - stETHReceived <= LIDO_ERROR_TOLERANCE_ETH), "ULD");
      emit FundsStaked(amount, shares, msg.sender);

      // Mint claim tokens
      fixedClaimToken[msg.sender] += shares;
      fixedClaimTokenTotalSupply += shares;
      fixedETHDepositToken[msg.sender] += amount;
      fixedETHDepositTokenTotalSupply += amount;

      emit FixedFundsDeposited(amount, shares, msg.sender);
    } else {
      // Variable side deposits

      // no refunds allowed
      require(amount <= variableSideCapacity - variableBearerTokenTotalSupply, "OED");
      uint256 remainingCapacity = variableSideCapacity - variableBearerTokenTotalSupply - amount;
      require(remainingCapacity == 0 || remainingCapacity >= minimumDepositAmount, "RC");

      // Mint bearer tokens
      variableBearerToken[msg.sender] += amount;
      variableBearerTokenTotalSupply += amount;

      emit VariableFundsDeposited(amount, msg.sender);
    }

    // Start the vault if we're at capacity
    if (
      fixedETHDepositTokenTotalSupply == fixedSideCapacity &&
      variableBearerTokenTotalSupply == variableSideCapacity
    ) {
      startTime = block.timestamp;
      endTime = block.timestamp + duration;
      fixedSidestETHOnStartCapacity = stakingBalance();
      fixedIntialTokenTotalSupply = fixedClaimTokenTotalSupply;
      emit VaultStarted(block.timestamp, msg.sender);
    }
  }

  /// @notice Claim fixed side bearer tokens with fixed side claim tokens
  /// and transfer to the user their upfront premium from variable deposits
  function claimFixedPremium() external {
    require(isStarted(), "CBS");

    // Check and cache balance for gas savings
    uint256 claimBal = fixedClaimToken[msg.sender];
    require(claimBal > 0, "NCT");

    // Send a proportional share of the total variable side deposits (premium) to the fixed side depositor
    uint256 sendAmount = claimBal.mulDiv(variableSideCapacity, fixedIntialTokenTotalSupply);

    // Track premiums
    userToFixedUpfrontPremium[msg.sender] = sendAmount;

    // Mint bearer token
    fixedBearerToken[msg.sender] += claimBal;
    fixedBearerTokenTotalSupply += claimBal;

    // Burn claim tokens
    fixedClaimToken[msg.sender] = 0;
    fixedClaimTokenTotalSupply -= claimBal;

    (bool sent, ) = msg.sender.call{value: sendAmount}("");
    require(sent, "ETF");
    emit FixedPremiumClaimed(sendAmount, claimBal, msg.sender);
  }

  /// @notice Withdraw from the vault
  /// @param side ID of side to withdraw from
  function withdraw(uint256 side) external {
    require(side == FIXED || side == VARIABLE, "IS");

    // Vault has not started
    if (!isStarted()) {
      if (side == FIXED) {
        require(fixedToVaultNotStartedWithdrawalRequestIds[msg.sender].length == 0, "WAR");

        // need to have claim tokens
        uint256 claimBalance = fixedClaimToken[msg.sender];
        //uint256 claimBalance = fixedClaimToken.balanceOf(msg.sender);
        require(claimBalance > 0, "NCT");

        fixedClaimToken[msg.sender] = 0;
        fixedClaimTokenTotalSupply -= claimBalance;
        // need to burn ETH deposit tokens so their deposit does not count towards reaching the fixed capacity
        fixedETHDepositTokenTotalSupply -= fixedETHDepositToken[msg.sender];
        fixedETHDepositToken[msg.sender] = 0;

        // Request withdrawal of their stETH shares
        // Add the requestIds associated with the user who submitted the withdrawal
        fixedToVaultNotStartedWithdrawalRequestIds[msg.sender] = requestWithdrawViaShares(
          msg.sender,
          claimBalance
        );

        emit LidoWithdrawalRequested(
          msg.sender,
          fixedToVaultNotStartedWithdrawalRequestIds[msg.sender],
          FIXED,
          false,
          false
        );
        return;
      } else {
        uint256 sendAmount = variableBearerToken[msg.sender];
        require(sendAmount > 0, "NBT");

        variableBearerToken[msg.sender] -= sendAmount;
        variableBearerTokenTotalSupply -= sendAmount;

        (bool sent, ) = msg.sender.call{value: sendAmount}("");
        require(sent, "ETF");

        emit VariableFundsWithdrawn(sendAmount, msg.sender, false, false);
        return;
      }

      // Vault started and in progress
    } else if (!isEnded()) {
      if (side == FIXED) {
        require(
          fixedToVaultOngoingWithdrawalRequestIds[msg.sender].requestIds.length == 0 &&
            fixedToVaultNotStartedWithdrawalRequestIds[msg.sender].length == 0,
          "WAR"
        );

        // require that they have claimed their upfront premium to simplify this flow
        // Also prevents completion of withdrawals initiated before the vault start
        // if the vault was initiated before the user's request was minted
        uint256 bearerBalance = fixedBearerToken[msg.sender];
        require(bearerBalance > 0, "NBT");

        // since the vault has started only withdraw their initial deposit equivalent in stETH  at the start of the vault- unless we are in a loss
        uint256 fixedETHDeposits = fixedSidestETHOnStartCapacity;
        uint256 withdrawAmount = fixedETHDeposits.mulDiv(fixedBearerToken[msg.sender], fixedLidoSharesTotalSupply());
        uint256 lidoStETHBalance = stakingBalance(); 

        if (fixedETHDeposits > lidoStETHBalance) {
          // our staking balance if less than our  stETH deposits at the start of the vault - only return a proportional amount of the balance to the fixed user
          withdrawAmount = lidoStETHBalance.mulDiv(fixedBearerToken[msg.sender], fixedLidoSharesTotalSupply());
        }

        fixedBearerToken[msg.sender] = 0;
        fixedBearerTokenTotalSupply -= bearerBalance;
        fixedSidestETHOnStartCapacity -= withdrawAmount;
        fixedToVaultOngoingWithdrawalRequestIds[msg.sender] = WithdrawalRequest({
          requestIds: requestWithdrawViaETH(msg.sender, withdrawAmount),
          timestamp: block.timestamp
        });
        fixedOngoingWithdrawalUsers.push(msg.sender);

        emit LidoWithdrawalRequested(
          msg.sender,
          fixedToVaultOngoingWithdrawalRequestIds[msg.sender].requestIds,
          FIXED,
          true,
          false
        );
        return;
      } else {
        if (msg.sender == protocolFeeReceiver && appliedProtocolFee > 0) {
          require(variableToVaultOngoingWithdrawalRequestIds[msg.sender].length == 0, "WAR");
          return protocolFeeReceiverWithdraw();
        }

        if (variableToVaultOngoingWithdrawalRequestIds[msg.sender].length == 0) {
          uint256 lidoStETHBalance = stakingBalance();
          uint256 fixedETHDeposits = fixedSidestETHOnStartCapacity;

        // staking earnings have accumulated on Lido
        if (lidoStETHBalance > fixedETHDeposits + minStETHWithdrawalAmount()) {
          uint256 currentStakes = stakingShares();
          (uint256 currentState, uint256 ethAmountOwed) = calculateVariableWithdrawState(
            (lidoStETHBalance.mulDiv(currentStakes + withdrawnStakingEarningsInStakes, currentStakes) - fixedETHDeposits),
            variableToWithdrawnStakingEarningsInShares[msg.sender].mulDiv(lidoStETHBalance, currentStakes)
          );
          if (ethAmountOwed >= minStETHWithdrawalAmount()) {
            // estimate protocol fee and update total - will actually be applied on withdraw finalization
            uint256 protocolFee = ethAmountOwed.mulDiv(protocolFeeBps, 10000);
            totalProtocolFee += protocolFee;
            uint256 stakesAmountOwed = lido.getSharesByPooledEth(ethAmountOwed);

            withdrawnStakingEarnings += ethAmountOwed - protocolFee;
            withdrawnStakingEarningsInStakes += stakesAmountOwed;

            variableToWithdrawnStakingEarnings[msg.sender] += ethAmountOwed - protocolFee;
            variableToWithdrawnStakingEarningsInShares[msg.sender] += stakesAmountOwed;
            variableToWithdrawnProtocolFee[msg.sender] += protocolFee;
            variableToVaultOngoingWithdrawalRequestIds[msg.sender] = requestWithdrawViaETH(
              msg.sender,
              ethAmountOwed
            );

              emit LidoWithdrawalRequested(
                msg.sender,
                variableToVaultOngoingWithdrawalRequestIds[msg.sender],
                VARIABLE,
                true,
                false
              );
            }
          }
        }

        // there are no staking earnings that can be withdrawn but there are fixed side early withdrawal fees
        if (feeEarnings > 0) {
          uint256 feeEarningsShare = calculateVariableFeeEarningsShare();
          if (feeEarningsShare > 0) {
            transferWithdrawnFunds(msg.sender, feeEarningsShare);

            emit VariableFundsWithdrawn(feeEarningsShare, msg.sender, true, false);
          }
        }
        return;
      }

      // Vault ended
    } else {
      return vaultEndedWithdraw(side);
    }
  }

  /// @notice Finalize a fixed withdrawal that was requested before the vault started
  function finalizeVaultNotStartedFixedWithdrawals() external {
    uint256[] memory requestIds = fixedToVaultNotStartedWithdrawalRequestIds[msg.sender];
    require(requestIds.length != 0, "WNR");

    delete fixedToVaultNotStartedWithdrawalRequestIds[msg.sender];

    // give fixed depositor all of their principal + any staking earnings
    uint256 sendAmount = claimWithdrawals(msg.sender, requestIds);

    bool _isStarted = isStarted();
    bool _isEnded = isEnded();
    transferWithdrawnFunds(msg.sender, sendAmount);

    emit LidoWithdrawalFinalized(msg.sender, requestIds, FIXED, _isStarted, _isEnded);
    emit FixedFundsWithdrawn(sendAmount, msg.sender, _isStarted, _isEnded);
  }

  /// @notice Finalize a fixed withdrawal that was requested after the vault has started
  function finalizeVaultOngoingFixedWithdrawals() external {
    uint256 sendAmount = claimFixedVaultOngoingWithdrawal(msg.sender);
    bool _isEnded = isEnded();
    uint256 arrayLength = fixedOngoingWithdrawalUsers.length;
    for (uint i = 0; i < arrayLength; i++) {
      if (fixedOngoingWithdrawalUsers[i] == msg.sender) {
        delete fixedOngoingWithdrawalUsers[i];
      }
    }
    transferWithdrawnFunds(msg.sender, sendAmount);

    emit FixedFundsWithdrawn(sendAmount, msg.sender, true, _isEnded);
  }

  /// @notice Finalize a variable withdrawal that was requested after the vault has started
  function finalizeVaultOngoingVariableWithdrawals() external {
    uint256[] memory requestIds = variableToVaultOngoingWithdrawalRequestIds[msg.sender];
    if(variableToPendingWithdrawalAmount[msg.sender] != 0) {
      withdrawAmountVariablePending();
      if(requestIds.length == 0) {
        return;
      }
    }
    require(requestIds.length != 0, "WNR");

    delete variableToVaultOngoingWithdrawalRequestIds[msg.sender];

    uint256 amountWithdrawn = claimWithdrawals(msg.sender, requestIds);

    uint256 protocolFee = applyProtocolFee(amountWithdrawn);

    uint256 sendAmount = amountWithdrawn + calculateVariableFeeEarningsShare() - protocolFee;

    bool _isEnded = isEnded();
    transferWithdrawnFunds(msg.sender, sendAmount);

    emit LidoWithdrawalFinalized(msg.sender, requestIds, VARIABLE, true, _isEnded);
    emit VariableFundsWithdrawn(sendAmount, msg.sender, true, _isEnded);
  }

  /// @notice Protocol Fee Reciever only Finalize a variable withdrawal that was requested after the vault has started
  /// @param user for whom the reward amount is due
  function feeReceiverFinalizeVaultOngoingVariableWithdrawals(address user) external {
    require(msg.sender == protocolFeeReceiver, "IFR");
    uint256[] memory requestIds = variableToVaultOngoingWithdrawalRequestIds[user];
    require(requestIds.length != 0, "WNR");

    delete variableToVaultOngoingWithdrawalRequestIds[user];

    uint256 amountWithdrawn = claimWithdrawals(user, requestIds);

    uint256 protocolFee = applyProtocolFee(amountWithdrawn);

    uint256 sendAmount = amountWithdrawn + calculateVariableFeeEarningsShareWithUser(user) - protocolFee;
    variableToPendingWithdrawalAmount[user] += sendAmount;
  }

  /// @notice withdrawal of funds for Variable side
  function withdrawAmountVariablePending() public {
    uint256 amount = variableToPendingWithdrawalAmount[msg.sender];
    variableToPendingWithdrawalAmount[msg.sender] = 0;
    payable(msg.sender).transfer(amount);
  }

  /// @notice Finalize the vault ended withdrawals
  /// @param side ID of side
  function finalizeVaultEndedWithdrawals(uint256 side) external {
    require(side == FIXED || side == VARIABLE, "IS");
    if (vaultEndedWithdrawalsFinalized) {
      return vaultEndedWithdraw(side);
    }
    require(vaultEndedWithdrawalRequestIds.length != 0 && !vaultEndedWithdrawalsFinalized, "WNR");

    vaultEndedWithdrawalsFinalized = true;

    // claim any ongoing fixed withdrawals too
    claimOngoingFixedWithdrawals();

    uint256 amountWithdrawn = claimWithdrawals(msg.sender, vaultEndedWithdrawalRequestIds);
    uint256 fixedETHDeposit = fixedSidestETHOnStartCapacity;
    if (amountWithdrawn > fixedETHDeposit) {
      vaultEndedStakingEarnings = amountWithdrawn - fixedETHDeposit;
      vaultEndedFixedDepositsFunds = fixedETHDeposit;
    } else {
      vaultEndedFixedDepositsFunds = amountWithdrawn;
    }

    uint256 protocolFee = applyProtocolFee(vaultEndedStakingEarnings);
    vaultEndedStakingEarnings -= protocolFee;

    emit LidoWithdrawalFinalized(msg.sender, vaultEndedWithdrawalRequestIds, side, true, true);

    return vaultEndedWithdraw(side);
  }

  function claimOngoingFixedWithdrawals() internal {
    uint256 arrayLength = fixedOngoingWithdrawalUsers.length;
    for (uint i = 0; i < arrayLength; i++) {
      address fixedUser = fixedOngoingWithdrawalUsers[i];
      fixedToPendingWithdrawalAmount[fixedUser] = claimFixedVaultOngoingWithdrawal(fixedUser);
      delete fixedOngoingWithdrawalUsers[i];
    }
  }

  /// @notice Helper function to apply Saffron's protocol fee to Lido staking earnings
  /// @param stakingEarnings the amount of ETH staking earnings on Lido
  function applyProtocolFee(uint256 stakingEarnings) internal returns (uint256) {
    uint256 protocolFee = stakingEarnings.mulDiv(protocolFeeBps, 10000);
    appliedProtocolFee += protocolFee;
    return protocolFee;
  }

  /// @notice Helper function to do the accounting for a withdraw after the vault has ended
  /// @param side ID of side
  function vaultEndedWithdraw(uint256 side) internal {
    if (vaultEndedWithdrawalRequestIds.length == 0 && !vaultEndedWithdrawalsFinalized) {
      emit VaultEnded(block.timestamp, msg.sender);
      if (stakingBalance() < minStETHWithdrawalAmount()) {

        claimOngoingFixedWithdrawals();
        vaultEndingStakesAmount = stakingShares();
        vaultEndingETHBalance = stakingBalance();
        // not enough staking ETH to withdraw just override vault ended state and continue the withdraw
        vaultEndedWithdrawalsFinalized = true;
      } else {
        vaultEndingStakesAmount = stakingShares();
        vaultEndingETHBalance = stakingBalance();
        vaultEndedWithdrawalRequestIds = requestEntireBalanceWithdraw(msg.sender);

        emit LidoWithdrawalRequested(msg.sender, vaultEndedWithdrawalRequestIds, side, true, true);
        // need to call finalizeVaultEndedWithdrawals once request is processed
        return;
      }
    }

    // have to call finalizeVaultEndedWithdrawals first
    require(vaultEndedWithdrawalsFinalized, "WNF");
    if (side == FIXED) {
      require(
        fixedToVaultOngoingWithdrawalRequestIds[msg.sender].requestIds.length == 0 &&
          fixedToVaultNotStartedWithdrawalRequestIds[msg.sender].length == 0,
        "WAR"
      );

      uint256 sendAmount = fixedToPendingWithdrawalAmount[msg.sender];

      // they submitted a withdraw before the vault had ended and the vault ending should have claimed it
      if (sendAmount > 0) {
        delete fixedToPendingWithdrawalAmount[msg.sender];
      } else {
        uint256 bearerBalance = fixedBearerToken[msg.sender];
        //uint256 bearerBalance = fixedBearerToken.balanceOf(msg.sender);
        require(bearerBalance > 0, "NBT");
        sendAmount = fixedBearerToken[msg.sender].mulDiv(
          vaultEndedFixedDepositsFunds,
          fixedLidoSharesTotalSupply()
        );

        fixedBearerToken[msg.sender] = 0;
        fixedBearerTokenTotalSupply -= bearerBalance;
        vaultEndedFixedDepositsFunds -= sendAmount;
      }

      transferWithdrawnFunds(msg.sender, sendAmount);

      emit FixedFundsWithdrawn(sendAmount, msg.sender, true, true);
      return;
    } else {
      require(variableToVaultOngoingWithdrawalRequestIds[msg.sender].length == 0, "WAR");

      if (msg.sender == protocolFeeReceiver && appliedProtocolFee > 0) {
        return protocolFeeReceiverWithdraw();
      }

      uint256 bearerBalance = variableBearerToken[msg.sender];
      require(bearerBalance > 0, "NBT");

      // Return proportional share of both earnings to caller
      uint256 stakingShareAmount = 0;

      uint256 totalEarnings = vaultEndingETHBalance.mulDiv(withdrawnStakingEarningsInStakes,vaultEndingStakesAmount) - totalProtocolFee  + vaultEndedStakingEarnings;

      if (totalEarnings > 0) {
        (uint256 currentState, uint256 stakingEarningsShare) = calculateVariableWithdrawState(
          totalEarnings,
          variableToWithdrawnStakingEarningsInShares[msg.sender].mulDiv(vaultEndingETHBalance, vaultEndingStakesAmount)
        );
        stakingShareAmount = stakingEarningsShare;
        variableToWithdrawnStakingEarningsInShares[msg.sender] = currentState.mulDiv(vaultEndingStakesAmount,vaultEndingETHBalance);
        variableToWithdrawnStakingEarnings[msg.sender] = currentState;
      }

      uint256 feeShareAmount = 0;
      if (withdrawnFeeEarnings + feeEarnings > 0) {
        feeShareAmount = calculateVariableFeeEarningsShare();
      }

      variableBearerToken[msg.sender] -= bearerBalance;
      variableBearerTokenTotalSupply -= bearerBalance;

      uint256 sendAmount = stakingShareAmount + feeShareAmount;
      transferWithdrawnFunds(msg.sender, sendAmount);

      emit VariableFundsWithdrawn(sendAmount, msg.sender, true, true);
      return;
    }
  }

  /// @notice Returns all fixed side Lido shares - claimed or unclaimed
  /// @return totalSupply Total supply of the fixed bearer and claim tokens
  function fixedLidoSharesTotalSupply() internal view returns (uint256) {
	return fixedBearerTokenTotalSupply + fixedClaimTokenTotalSupply;
  }

  /// @notice Helper function to claim a Lido fixed withdrawal that was requested after the vault has started
  /// @param user User that requested the withdrawal
  /// @return amount Amount of ETH withdrawn from Lido after the withdraw has been claimed minus the early exit fees
  function claimFixedVaultOngoingWithdrawal(address user) internal returns (uint256) {
    if (user == address(0)) return 0;

    WithdrawalRequest memory request = fixedToVaultOngoingWithdrawalRequestIds[user];
    uint256[] memory requestIds = request.requestIds;
    require(requestIds.length != 0, "WNR");

    uint256 upfrontPremium = userToFixedUpfrontPremium[user];

    delete userToFixedUpfrontPremium[user];
    delete fixedToVaultOngoingWithdrawalRequestIds[user];

    // uint256 arrayLength = fixedOngoingWithdrawalUsers.length;
    // for (uint i = 0; i < arrayLength; i++) {
    //   if (fixedOngoingWithdrawalUsers[i] == user) {
    //     delete fixedOngoingWithdrawalUsers[i];
    //   }
    // }

    uint256 amountWithdrawn = claimWithdrawals(msg.sender, requestIds);

    uint256 earlyExitFees = calculateFixedEarlyExitFees(upfrontPremium, request.timestamp);
    // make sure, that earlyExitFee cant be higher than initial deposit
    earlyExitFees = Math.min(earlyExitFees, amountWithdrawn);

    // add earlyExitFees to earnings for variable side
    feeEarnings += earlyExitFees;

    emit LidoWithdrawalFinalized(user, requestIds, FIXED, true, isEnded());

    return amountWithdrawn - earlyExitFees;
  }

  /// @notice Helper function to withdraw any accrued protocol fees
  function protocolFeeReceiverWithdraw() internal {
    uint256 protocolFee = appliedProtocolFee;
    appliedProtocolFee = 0;
    transferWithdrawnFunds(msg.sender, protocolFee);

    emit VariableFundsWithdrawn(protocolFee, msg.sender, isStarted(), isEnded());
  }

  /// @notice Helper function to calculate the ongoing variable withdraw state
  /// The vault must track a variable user's withdrawals during the duration of the vault since withdrawals can be executed at any time
  /// @param totalEarnings Amount of staking or fee earnings
  /// @param previousWithdrawnAmount The total amount of earnings already withdrawn
  /// @return (currentState, amountOwed) The new total amount of earnings withdrawn, the amount to withdraw
  function calculateVariableWithdrawState(
    uint256 totalEarnings,
    uint256 previousWithdrawnAmount
  ) internal view returns (uint256, uint256) {

    uint256 bearerBalance = variableBearerToken[msg.sender];
    require(bearerBalance > 0, "NBT");

    uint256 totalOwed = bearerBalance.mulDiv(totalEarnings, variableSideCapacity);
    uint256 ethAmountOwed = 0;
    if (previousWithdrawnAmount < totalOwed) {
      ethAmountOwed = totalOwed - previousWithdrawnAmount;
    }

    return (ethAmountOwed + previousWithdrawnAmount, ethAmountOwed);
  }
  
  /// @notice Helper function to calculate the ongoing variable withdraw state
  /// The vault must track a variable user's withdrawals during the duration of the vault since withdrawals can be executed at any time
  /// @param user for whom the withdrawal variable amount is calculated
  /// @return (ethAmountOwed) The amount to withdraw
  function getCalculateVariableWithdrawStateWithStakingBalance(address user) public view returns (uint256) {
    uint256 lidoStETHBalance = stakingBalance();
    uint256 fixedETHDeposits = fixedETHDepositTokenTotalSupply;
    require(lidoStETHBalance > fixedETHDeposits, "LBL");
    uint256 totalEarnings = (lidoStETHBalance - fixedETHDeposits) + withdrawnStakingEarnings + totalProtocolFee;
    uint256 previousWithdrawnAmount = variableToWithdrawnStakingEarnings[user].mulDiv(10000, 10000 - protocolFeeBps);

    uint256 bearerBalance = variableBearerToken[user];
    require(bearerBalance > 0, "NBT");

    uint256 totalOwed = bearerBalance.mulDiv(totalEarnings, variableSideCapacity);
    uint256 ethAmountOwed = 0;
    if (previousWithdrawnAmount < totalOwed) {
      ethAmountOwed = totalOwed - previousWithdrawnAmount;
    }
    return ethAmountOwed;
  }
  
  /// @notice Helper function to calculate the ongoing variable withdraw state for user
  /// The vault must track a variable user's withdrawals during the duration of the vault since withdrawals can be executed at any time
  /// @param totalEarnings Amount of staking or fee earnings
  /// @param previousWithdrawnAmount The total amount of earnings already withdrawn
  /// @param user for whom the calculation
  /// @return (currentState, amountOwed) The new total amount of earnings withdrawn, the amount to withdraw
  function calculateVariableWithdrawStateWithUser(
    uint256 totalEarnings,
    uint256 previousWithdrawnAmount,
    address user
  ) internal view returns (uint256, uint256) {

    uint256 bearerBalance = variableBearerToken[user];
    require(bearerBalance > 0, "NBT");

    uint256 totalOwed = bearerBalance.mulDiv(totalEarnings, variableSideCapacity);
    uint256 ethAmountOwed = 0;
    if (previousWithdrawnAmount < totalOwed) {
      ethAmountOwed = totalOwed - previousWithdrawnAmount;
    }

    return (ethAmountOwed + previousWithdrawnAmount, ethAmountOwed);
  }
  
  /// @notice Helper function to calculate a variable user's proportional amount of fees that they are owed
  /// @param user for which the variable amount of the user proportional to the commission due to him is calculated
  /// @return (currentState, feeEarningsShare) The new total amount of earnings withdrawn, the amount to fee share
  function getCalculateVariableFeeEarningsShareWithUser(address user) public view returns (uint256, uint256) {
    uint256 totalEarnings = feeEarnings + withdrawnFeeEarnings;
    uint256 previousWithdrawnAmount = variableToWithdrawnFees[user];

    uint256 bearerBalance = variableBearerToken[user];
    require(bearerBalance > 0, "NBT");

    uint256 totalOwed = bearerBalance.mulDiv(totalEarnings, variableSideCapacity);
    uint256 ethAmountOwed = 0;
    if (previousWithdrawnAmount < totalOwed) {
      ethAmountOwed = totalOwed - previousWithdrawnAmount;
    }

    uint256 currentState = ethAmountOwed + previousWithdrawnAmount;
    uint256 feeEarningsShare = ethAmountOwed;

    return (currentState, feeEarningsShare);
  }

  /// @notice Helper function to calculate a variable user's proportional amount of fees that they are owed
  /// @return amount The amount to give the user
  function calculateVariableFeeEarningsShare() internal returns (uint256) {
    (uint256 currentState, uint256 feeEarningsShare) = calculateVariableWithdrawState(
      feeEarnings + withdrawnFeeEarnings,
      variableToWithdrawnFees[msg.sender]
    );

    variableToWithdrawnFees[msg.sender] = currentState;
    withdrawnFeeEarnings += feeEarningsShare;
    feeEarnings -= feeEarningsShare;

    return feeEarningsShare;
  }

  /// @notice A helper function to calculate the proportional amount of the reward due to the user that is passed in the variable
  /// @param user to whom the reward amount is due
  /// @return amount The amount to give the user
  function calculateVariableFeeEarningsShareWithUser(address user) internal returns (uint256) {
    (uint256 currentState, uint256 feeEarningsShare) = calculateVariableWithdrawStateWithUser(
      feeEarnings + withdrawnFeeEarnings,
      variableToWithdrawnFees[user],
      user
    );

    variableToWithdrawnFees[user] = currentState;
    withdrawnFeeEarnings += feeEarningsShare;
    feeEarnings -= feeEarningsShare;

    return feeEarningsShare;
  }

  /// @notice Helper function to calculate the total fees to apply to a fixed user's principal that they are requesting to withdraw
  /// The earlier in the vault duration the withdraw request is made the higher the fee
  /// The user must also pay back their upfront premium
  /// @param upfrontPremium Upfront premium the fixed user received
  /// @param timestampRequested When the withdraw was requested
  /// @return earlyExitFee The total amount of early exit fees to apply
  function calculateFixedEarlyExitFees(
    uint256 upfrontPremium,
    uint256 timestampRequested
  ) internal view returns (uint256) {
    uint256 remainingProportion = (endTime > timestampRequested ? endTime - timestampRequested : 0).mulDiv(
      1e18,
      duration
    );

    // Calculate the scaling fee based on the linear factor and earlyExitFeeBps
    uint256 earlyExitFees = upfrontPremium.mulDiv( (1 + earlyExitFeeBps).mulDiv(remainingProportion, 1e18), 10000);

    // Calculate the amount to be paid back of their original upfront claimed premium, not influenced by quadratic scaling
    earlyExitFees += upfrontPremium - upfrontPremium.mulDiv(timestampRequested - startTime, duration);

    return earlyExitFees;
  }

  // Begin Adapter Code Refactor

  /// @notice Lido contract
  ILido public constant lido = ILido(0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84);

  /// @notice Lido withdrawal queue contract
  ILidoWithdrawalQueueERC721 public constant lidoWithdrawalQueue =
    ILidoWithdrawalQueueERC721(0x889edC2eDab5f40e902b864aD4d7AdE8E412F9B1);

  /// @notice minimum amount of stETH that is possible to withdraw from Lido
  /// @dev this is pulled from the Lido contract and shouldn't be changed, otherwise funds maybe stuck unnecessarily
  uint256 private constant MIN_STETH_WITHDRAWAL_AMOUNT = 100;

  /// @notice maximum amount of stETH that is possible to withdraw in a single withdrawal request to Lido
  /// @dev this is pulled from the Lido contract and shouldn't be changed
  uint256 private constant MAX_STETH_WITHDRAWAL_AMOUNT = 1000 * 1e18;

  /// @notice ETH diff tolerance either for the expected deposit or withdraw from Lido - some rounding errors of a few wei seem to occur
  uint256 public constant LIDO_ERROR_TOLERANCE_ETH = 10 wei;

  /// @notice Emitted when the vault contract address is updated
  /// @param vault New vault address
  /// @param setter Address of setter
  event VaultSet(address vault, address indexed setter);

  /// @notice Emitted when ETH is submitted to the Lido platform for staking
  /// @param amount ETH amount staked
  /// @param shares Amount of shares received for the ETH staked
  /// @param user User address
  event FundsStaked(uint256 amount, uint256 shares, address user);

  /// @notice Emitted when withdrawn funds are transferred out of this contract
  /// @param amount ETH amount transferred
  /// @param recipient Address of receiver of the funds
  event FundsTransferred(uint256 amount, address recipient);

  /// @notice Emitted when stETH funds are transferred out of this contract
  /// @param amount ETH amount transferred
  /// @param recipient Address of receiver of the stETH
  event TransferredStETH(uint256 amount, address recipient);

  /// @notice Emitted when a Lido withdrawal request ERC721 is transferred out of this contract
  /// @param requestId Request id of the Lido withdrawal request
  /// @param recipient Address of receiver of the ERC721
  event TransferredWithdrawalERC721(uint256 requestId, address recipient);

  /// @notice Emitted when stakings funds are requested to be withdrawn from Lido
  /// @param amount ETH amount requested to be withdrawn
  /// @param requestIds Request ids of the Lido withdrawal requests
  /// @param user User address
  event WithdrawalRequested(uint256 amount, uint256[] requestIds, address user);

  /// @notice Emitted when a Lido fixed withdrawal is claimed and the ETH is transferred to this contract
  /// @param amount ETH amount withdrawn
  /// @param requestIds Request ids of the Lido withdrawal requests
  /// @param user User address
  event WithdrawalClaimed(uint256 amount, uint256[] requestIds, address user);

  // Getters

  /// @notice The minimum amount of stETH that can be withdrawn from Lido
  function minStETHWithdrawalAmount() public pure returns (uint256) {
    return MIN_STETH_WITHDRAWAL_AMOUNT;
  }

  /// @notice The maximum amount of stETH that can be withdrawn from Lido in a single request
  /// If desired amount to withdraw is greater than the value it has to be broken up into multiple requests
  function maxStETHWithdrawalAmount() external pure returns (uint256) {
    return MAX_STETH_WITHDRAWAL_AMOUNT;
  }

  // Setters

  // Staking

  /// @notice Balance of stETH tokens, which is dynamic based off of staking rewards https://docs.lido.fi/contracts/lido#rebase
  function stakingBalance() public view returns (uint256) {
    return lido.balanceOf(address(this));
  }

  function thisBalance() public view returns (uint256) {
    return address(this).balance;
  }

  /// @notice Balance of Lido stakes
  function stakingShares() public view returns (uint256) {
    return lido.sharesOf(address(this));
  }

  /// @notice Transfers ETH unstaked from Lido
  /// @param recipient Address to transfer to
  /// @param amount Amount of ETH
  /// @dev Make sure this is only callable by the vault
  function transferWithdrawnFunds(address recipient, uint256 amount) internal {
    require(recipient != address(0), "NZA");

    (bool sent, ) = recipient.call{value: amount}("");
    require(sent, "ETF");

    emit FundsTransferred(amount, recipient);
  }

  // Withdrawals

  /// @notice Request a withdrawal on Lido to exchange stETH for ETH via shares amount
  /// @param user Address of the user that is requesting the withdraw
  /// @param shares Amount of staking shares to withdraw. Will be used to calculate the actual stETH amount to withdraw
  /// @return requestIds Ids of the withdrawal requests
  function requestWithdrawViaShares(address user, uint256 shares) internal returns (uint256[] memory) {
    require(shares != 0, "NEI");

    // get stETH amount corresponding to shares
    uint256 stETHAmount = lido.getPooledEthByShares(shares);

    return requestWithdrawViaETH(user, stETHAmount);
  }

  /// @notice Request a withdrawal of the entire stETH balance of the contract
  /// @param user Address of the user that is requesting the withdraw
  /// @return requestIds Ids of the withdrawal requests
  function requestEntireBalanceWithdraw(address user) internal returns (uint256[] memory) {
    uint256 stETHAmount = stakingBalance();

    return requestWithdrawViaETH(user, stETHAmount);
  }

  /// @notice Request a withdrawal on Lido to exchange stETH for ETH via stETH amount
  /// @param user Address of the user that is requesting the withdraw
  /// @param stETHAmount Amount of stETH to withdraw
  /// @return requestIds Ids of the withdrawal requests
  function requestWithdrawViaETH(address user, uint256 stETHAmount) internal returns (uint256[] memory) {
    return _requestWithdraw(user, stETHAmount);
  }

  /// @notice Request a withdrawal on Lido to exchange stETH for ETH
  /// @param stETHAmount amount of stETH to withdraw
  /// @return requestIds Ids of the withdrawal requests
  function _requestWithdraw(address user, uint256 stETHAmount) internal returns (uint256[] memory) {
    unlockReceive = true;
    require(stETHAmount >= MIN_STETH_WITHDRAWAL_AMOUNT, "WM");

    // Approve the withdrawal queue contract to pull the stETH tokens
    bool approved = lido.approve(address(lidoWithdrawalQueue), stETHAmount);
    require(approved, "AF");

    uint256[] memory amounts = calculateWithdrawals(stETHAmount);

    // Submit the stETH to the withdrawal queue
    uint256[] memory requestIds = lidoWithdrawalQueue.requestWithdrawals(amounts, address(this));
    require(requestIds.length > 0, "IWR");

    emit WithdrawalRequested(stETHAmount, requestIds, user);

    unlockReceive = false;
    return requestIds;
  }

  /// @notice Claim a withdrawal request on Lido once it has been finalized. This will transfer ETH to the contract if successful
  /// @param user Address of the user that is claiming the withdraw
  /// @param requestIds Withdrawal request ids to claim
  /// @return amount of ETH withdrawn
  function claimWithdrawals(address user, uint256[] memory requestIds) internal returns (uint256) {
    return _claimWithdrawals(user, requestIds);
  }

  /// @notice Claim a withdrawal request on Lido once it has been finalized
  /// @param requestIds Ids of the withdrawal requests
  /// @return amount of ETH withdrawn
  function _claimWithdrawals(address user, uint256[] memory requestIds) internal returns (uint256) {
    unlockReceive = true;
    uint256 beforeBalance = address(this).balance;

    // Claim Ether for the burned stETH positions
    // this will fail if the request is not finalized
    for (uint i = 0; i < requestIds.length; i++) {
      lidoWithdrawalQueue.claimWithdrawal(requestIds[i]);
    }

    uint256 withdrawnAmount = address(this).balance - beforeBalance;
    require(withdrawnAmount > 0, "IWA");

    emit WithdrawalClaimed(withdrawnAmount, requestIds, user);

    unlockReceive = false;
    return withdrawnAmount;
  }

  /// @notice Helper function to calculate how many requests will be needed to unstake from Lido due to the maximum withdrawal amount per request
  /// @param stETHAmount stETH amount requested to be withdrawn
  /// @return withdrawalAmounts an array of the amounts to withdraw in each withdrawal request to Lido
  function calculateWithdrawals(uint256 stETHAmount) internal pure returns (uint256[] memory) {
    uint256[] memory withdrawalAmounts;

    // If stETHAmount is less than the max withdrawal amount, only 1 withdrawal is needed.
    if (stETHAmount <= MAX_STETH_WITHDRAWAL_AMOUNT) {
      withdrawalAmounts = new uint256[](1);
      withdrawalAmounts[0] = stETHAmount;
      return withdrawalAmounts;
    }

    // Calculate the base number of maximum withdrawals
    uint256 numberOfWithdrawals = stETHAmount / MAX_STETH_WITHDRAWAL_AMOUNT;
    uint256 remainingAmount = stETHAmount % MAX_STETH_WITHDRAWAL_AMOUNT;

    // Allocate array size based on the number of withdrawals
    // Adding an extra slot in case we need to adjust the last withdrawal
    withdrawalAmounts = new uint256[](numberOfWithdrawals + (remainingAmount > 0 ? 1 : 0));

    // Fill the array with the amounts for the full withdrawals
    for (uint256 i = 0; i < numberOfWithdrawals; i++) {
      withdrawalAmounts[i] = MAX_STETH_WITHDRAWAL_AMOUNT;
    }

    // Handling the last withdrawal
    if (remainingAmount > 0) {
      if (remainingAmount < MIN_STETH_WITHDRAWAL_AMOUNT && numberOfWithdrawals > 0) {
        // Calculate the amount to be added to the remaining amount from the previous withdrawal
        uint256 amountToShift = (MIN_STETH_WITHDRAWAL_AMOUNT - remainingAmount);

        // Adjust the last two withdrawals
        withdrawalAmounts[numberOfWithdrawals - 1] -= amountToShift;
        withdrawalAmounts[numberOfWithdrawals] = remainingAmount + amountToShift;
      } else {
        // If remaining amount is above the minimum or if there are no full withdrawals, set it as the last withdrawal
        withdrawalAmounts[numberOfWithdrawals] = remainingAmount;
      }
    }

    return withdrawalAmounts;
  }

}
