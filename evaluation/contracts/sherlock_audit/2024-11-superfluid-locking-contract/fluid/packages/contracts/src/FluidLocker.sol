// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

/* Openzeppelin Contracts & Interfaces */
import { Math } from "@openzeppelin-v5/contracts/utils/math/Math.sol";
import { SafeCast } from "@openzeppelin-v5/contracts/utils/math/SafeCast.sol";
import { Initializable } from "@openzeppelin-v5/contracts/proxy/utils/Initializable.sol";
import { BeaconProxy } from "@openzeppelin/contracts/proxy/beacon/BeaconProxy.sol";
import { UpgradeableBeacon } from "@openzeppelin/contracts/proxy/beacon/UpgradeableBeacon.sol";

/* Superfluid Protocol Contracts & Interfaces */
import {
    ISuperfluidPool,
    ISuperToken
} from "@superfluid-finance/ethereum-contracts/contracts/interfaces/superfluid/ISuperfluid.sol";
import { SuperTokenV1Library } from "@superfluid-finance/ethereum-contracts/contracts/apps/SuperTokenV1Library.sol";

/* FLUID Interfaces */
import { IEPProgramManager } from "./interfaces/IEPProgramManager.sol";
import { IFluidLocker } from "./interfaces/IFluidLocker.sol";
import { IStakingRewardController } from "./interfaces/IStakingRewardController.sol";
import { IFontaine } from "./interfaces/IFontaine.sol";

import { ReentrancyGuard } from "solady/utils/ReentrancyGuard.sol";

using SuperTokenV1Library for ISuperToken;
using SafeCast for int256;

/**
 * @title Locker Contract
 * @author Superfluid
 * @notice Contract responsible for locking and holding FLUID token on behalf of users
 *
 */
contract FluidLocker is Initializable, ReentrancyGuard, IFluidLocker {
    //      ____                          __        __    __        _____ __        __
    //     /  _/___ ___  ____ ___  __  __/ /_____ _/ /_  / /__     / ___// /_____ _/ /____  _____
    //     / // __ `__ \/ __ `__ \/ / / / __/ __ `/ __ \/ / _ \    \__ \/ __/ __ `/ __/ _ \/ ___/
    //   _/ // / / / / / / / / / / /_/ / /_/ /_/ / /_/ / /  __/   ___/ / /_/ /_/ / /_/  __(__  )
    //  /___/_/ /_/ /_/_/ /_/ /_/\__,_/\__/\__,_/_.___/_/\___/   /____/\__/\__,_/\__/\___/____/

    /// @notice FLUID SuperToken interface
    ISuperToken public immutable FLUID;

    /// @notice Superfluid GDA Tax Distribution Pool interface
    ISuperfluidPool public immutable TAX_DISTRIBUTION_POOL;

    /// @notice Distribution Program Manager interface
    IEPProgramManager public immutable EP_PROGRAM_MANAGER;

    /// @notice Staking Reward Controller interface
    IStakingRewardController public immutable STAKING_REWARD_CONTROLLER;

    /// @notice Fontaine Beacon contract address
    UpgradeableBeacon public immutable FONTAINE_BEACON;

    /// @notice Boolean sets to True if unlock is available
    bool public immutable UNLOCK_AVAILABLE;

    /// @notice Staking cooldown period
    uint80 private constant _STAKING_COOLDOWN_PERIOD = 3 days;

    /// @notice Minimum unlock period allowed (1 week)
    uint128 private constant _MIN_UNLOCK_PERIOD = 7 days;

    /// @notice Maximum unlock period allowed (18 months)
    uint128 private constant _MAX_UNLOCK_PERIOD = 540 days;

    /// @notice Instant unlock penalty percentage (expressed in basis points)
    uint256 private constant _INSTANT_UNLOCK_PENALTY_BP = 8_000;

    /// @notice Basis points denominator (for percentage calculation)
    uint256 private constant _BP_DENOMINATOR = 10_000;

    /// @notice Scaler used for unlock percentage calculation
    uint256 private constant _SCALER = 1e18;

    /// @notice Scaler used for unlock percentage calculation
    uint256 private constant _PERCENT_TO_BP = 100;

    //     _____ __        __
    //    / ___// /_____ _/ /____  _____
    //    \__ \/ __/ __ `/ __/ _ \/ ___/
    //   ___/ / /_/ /_/ / /_/  __(__  )
    //  /____/\__/\__,_/\__/\___/____/

    /// @notice This locker owner address
    address public lockerOwner;

    /// @notice Timestamp at which the staking cooldown period is elapsed
    uint80 public stakingUnlocksAt;

    /// @notice Total unlock count
    uint16 public fontaineCount;

    /// @notice Balance of $FLUID token staked in this locker
    uint256 private _stakedBalance;

    /// @notice Stores the Fontaine contract associated to the given unlock identifier
    mapping(uint256 unlockId => IFontaine fontaine) public fontaines;

    //     ______                 __                  __
    //    / ____/___  ____  _____/ /________  _______/ /_____  _____
    //   / /   / __ \/ __ \/ ___/ __/ ___/ / / / ___/ __/ __ \/ ___/
    //  / /___/ /_/ / / / (__  ) /_/ /  / /_/ / /__/ /_/ /_/ / /
    //  \____/\____/_/ /_/____/\__/_/   \__,_/\___/\__/\____/_/

    /**
     * @notice Locker contract constructor
     * @param fluid FLUID SuperToken contract interface
     * @param taxDistributionPool Tax Distribution Pool GDA contract interface
     * @param programManager Ecosystem Partner Program Manager contract interface
     * @param stakingRewardController Staking Reward Controller contract interface
     * @param fontaineBeacon Fontaine Beacon contract address
     * @param isUnlockAvailable True if the unlock is available, false otherwise
     */
    constructor(
        ISuperToken fluid,
        ISuperfluidPool taxDistributionPool,
        IEPProgramManager programManager,
        IStakingRewardController stakingRewardController,
        address fontaineBeacon,
        bool isUnlockAvailable
    ) {
        // Disable initializers to prevent implementation contract initalization
        _disableInitializers();

        // Sets immutable states
        UNLOCK_AVAILABLE = isUnlockAvailable;
        FLUID = fluid;
        TAX_DISTRIBUTION_POOL = taxDistributionPool;
        EP_PROGRAM_MANAGER = programManager;
        STAKING_REWARD_CONTROLLER = stakingRewardController;

        // Sets the Fontaine beacon address
        FONTAINE_BEACON = UpgradeableBeacon(fontaineBeacon);
    }

    /**
     * @notice Locker contract initializer
     * @param owner this Locker contract owner account
     */
    function initialize(address owner) external initializer {
        // Sets the owner of this locker
        lockerOwner = owner;
    }

    //      ______     __                        __   ______                 __  _
    //     / ____/  __/ /____  _________  ____ _/ /  / ____/_  ______  _____/ /_(_)___  ____  _____
    //    / __/ | |/_/ __/ _ \/ ___/ __ \/ __ `/ /  / /_  / / / / __ \/ ___/ __/ / __ \/ __ \/ ___/
    //   / /____>  </ /_/  __/ /  / / / / /_/ / /  / __/ / /_/ / / / / /__/ /_/ / /_/ / / / (__  )
    //  /_____/_/|_|\__/\___/_/  /_/ /_/\__,_/_/  /_/    \__,_/_/ /_/\___/\__/_/\____/_/ /_/____/

    /// @inheritdoc IFluidLocker
    function claim(uint256 programId, uint256 totalProgramUnits, uint256 nonce, bytes memory stackSignature)
        external
        nonReentrant
    {
        // Get the corresponding program pool
        ISuperfluidPool programPool = EP_PROGRAM_MANAGER.getProgramPool(programId);

        if (!FLUID.isMemberConnected(address(programPool), address(this))) {
            // Connect this locker to the Program Pool
            FLUID.connectPool(programPool);
        }

        // Request program manager to update this locker's units
        EP_PROGRAM_MANAGER.updateUserUnits(lockerOwner, programId, totalProgramUnits, nonce, stackSignature);

        emit IFluidLocker.FluidStreamClaimed(programId, totalProgramUnits);
    }

    /// @inheritdoc IFluidLocker
    function claim(
        uint256[] memory programIds,
        uint256[] memory totalProgramUnits,
        uint256[] memory nonces,
        bytes[] memory stackSignatures
    ) external nonReentrant {
        for (uint256 i = 0; i < programIds.length; ++i) {
            // Get the corresponding program pool
            ISuperfluidPool programPool = EP_PROGRAM_MANAGER.getProgramPool(programIds[i]);

            if (!FLUID.isMemberConnected(address(programPool), address(this))) {
                // Connect this locker to the Program Pool
                FLUID.connectPool(programPool);
            }
        }

        // Request program manager to update this locker's units
        EP_PROGRAM_MANAGER.batchUpdateUserUnits(lockerOwner, programIds, totalProgramUnits, nonces, stackSignatures);

        emit IFluidLocker.FluidStreamsClaimed(programIds, totalProgramUnits);
    }

    /// @inheritdoc IFluidLocker
    function lock(uint256 amount) external nonReentrant {
        // Fetch the amount of FLUID Token to be locked from the caller
        FLUID.transferFrom(msg.sender, address(this), amount);

        emit FluidLocked(amount);
    }

    /// @inheritdoc IFluidLocker
    function unlock(uint128 unlockPeriod, address recipient) external nonReentrant onlyLockerOwner unlockAvailable {
        // Enforce unlock period validity
        if (unlockPeriod != 0 && (unlockPeriod < _MIN_UNLOCK_PERIOD || unlockPeriod > _MAX_UNLOCK_PERIOD)) {
            revert INVALID_UNLOCK_PERIOD();
        }

        // Ensure recipient is not the zero-address
        if (recipient == address(0)) {
            revert FORBIDDEN();
        }

        // Get balance available for unlocking
        uint256 availableBalance = getAvailableBalance();

        // Revert if there is no FLUID to unlock
        if (availableBalance == 0) revert NO_FLUID_TO_UNLOCK();

        if (unlockPeriod == 0) {
            _instantUnlock(availableBalance, recipient);
        } else {
            _vestUnlock(availableBalance, unlockPeriod, recipient);
        }
    }

    /// @inheritdoc IFluidLocker
    function stake() external nonReentrant onlyLockerOwner unlockAvailable {
        uint256 amountToStake = getAvailableBalance();

        if (amountToStake == 0) revert NO_FLUID_TO_STAKE();

        if (!FLUID.isMemberConnected(address(TAX_DISTRIBUTION_POOL), address(this))) {
            // Connect this locker to the Tax Distribution Pool
            FLUID.connectPool(TAX_DISTRIBUTION_POOL);
        }

        // Update staked balance
        _stakedBalance += amountToStake;

        // Update unlock timestamp
        stakingUnlocksAt = uint80(block.timestamp) + _STAKING_COOLDOWN_PERIOD;

        // Call Staking Reward Controller to update staker's units
        STAKING_REWARD_CONTROLLER.updateStakerUnits(_stakedBalance);

        emit FluidStaked(_stakedBalance, amountToStake);
    }

    /// @inheritdoc IFluidLocker
    function unstake() external nonReentrant onlyLockerOwner unlockAvailable {
        if (block.timestamp < stakingUnlocksAt) {
            revert STAKING_COOLDOWN_NOT_ELAPSED();
        }

        // Enfore staked balance is not zero
        if (_stakedBalance == 0) revert NO_FLUID_TO_UNSTAKE();

        // Set staked balance to 0
        _stakedBalance = 0;

        // Call Staking Reward Controller to update staker's units
        STAKING_REWARD_CONTROLLER.updateStakerUnits(0);

        // Disconnect this locker from the Tax Distribution Pool
        FLUID.disconnectPool(TAX_DISTRIBUTION_POOL);

        emit FluidUnstaked();
    }

    //   _    ___                 ______                 __  _
    //  | |  / (_)__ _      __   / ____/_  ______  _____/ /_(_)___  ____  _____
    //  | | / / / _ \ | /| / /  / /_  / / / / __ \/ ___/ __/ / __ \/ __ \/ ___/
    //  | |/ / /  __/ |/ |/ /  / __/ / /_/ / / / / /__/ /_/ / /_/ / / / (__  )
    //  |___/_/\___/|__/|__/  /_/    \__,_/_/ /_/\___/\__/_/\____/_/ /_/____/

    /// @inheritdoc IFluidLocker
    function getFlowRatePerProgram(uint256 programId) public view returns (int96 flowRate) {
        // Get the corresponding program pool
        ISuperfluidPool programPool = EP_PROGRAM_MANAGER.getProgramPool(programId);

        // Get the flow rate
        flowRate = programPool.getMemberFlowRate(address(this));
    }

    /// @inheritdoc IFluidLocker
    function getFlowRatePerProgram(uint256[] memory programIds) external view returns (int96[] memory flowRates) {
        flowRates = new int96[](programIds.length);

        for (uint256 i = 0; i < programIds.length; ++i) {
            flowRates[i] = getFlowRatePerProgram(programIds[i]);
        }
    }

    /// @inheritdoc IFluidLocker
    function getUnitsPerProgram(uint256 programId) public view returns (uint128 units) {
        // Get the corresponding program pool
        ISuperfluidPool programPool = EP_PROGRAM_MANAGER.getProgramPool(programId);

        // Get this locker's unit within the given program identifier
        units = programPool.getUnits(address(this));
    }

    /// @inheritdoc IFluidLocker
    function getUnitsPerProgram(uint256[] memory programIds) external view returns (uint128[] memory units) {
        units = new uint128[](programIds.length);

        for (uint256 i = 0; i < programIds.length; ++i) {
            units[i] = getUnitsPerProgram(programIds[i]);
        }
    }

    /// @inheritdoc IFluidLocker
    function getStakedBalance() external view returns (uint256 sBalance) {
        sBalance = _stakedBalance;
    }

    /// @inheritdoc IFluidLocker
    function getAvailableBalance() public view returns (uint256 aBalance) {
        aBalance = FLUID.balanceOf(address(this)) - _stakedBalance;
    }

    /// @inheritdoc IFluidLocker
    function getFontaineBeaconImplementation() public view returns (address fontaineBeaconImpl) {
        fontaineBeaconImpl = FONTAINE_BEACON.implementation();
    }

    //      ____      __                        __   ______                 __  _
    //     /  _/___  / /____  _________  ____ _/ /  / ____/_  ______  _____/ /_(_)___  ____  _____
    //     / // __ \/ __/ _ \/ ___/ __ \/ __ `/ /  / /_  / / / / __ \/ ___/ __/ / __ \/ __ \/ ___/
    //   _/ // / / / /_/  __/ /  / / / / /_/ / /  / __/ / /_/ / / / / /__/ /_/ / /_/ / / / (__  )
    //  /___/_/ /_/\__/\___/_/  /_/ /_/\__,_/_/  /_/    \__,_/_/ /_/\___/\__/_/\____/_/ /_/____/

    function _instantUnlock(uint256 amountToUnlock, address recipient) internal {
        // Calculate instant unlock penalty amount
        uint256 penaltyAmount = (amountToUnlock * _INSTANT_UNLOCK_PENALTY_BP) / _BP_DENOMINATOR;

        // Distribute penalty to staker (connected to the TAX_DISTRIBUTION_POOL)
        FLUID.distributeToPool(address(this), TAX_DISTRIBUTION_POOL, penaltyAmount);

        // Transfer the leftover $FLUID to the locker owner
        FLUID.transfer(recipient, amountToUnlock - penaltyAmount);

        emit FluidUnlocked(0, amountToUnlock, recipient, address(0));
    }

    function _vestUnlock(uint256 amountToUnlock, uint128 unlockPeriod, address recipient) internal {
        // Calculate the unlock and penalty flow rates based on requested amount and unlock period
        (int96 unlockFlowRate, int96 taxFlowRate) = _calculateVestUnlockFlowRates(amountToUnlock, unlockPeriod);

        // Use create2 to deploy a Fontaine Beacon Proxy
        // The salt used for deployment is the hashed encoded Locker address and unlock identifier
        address newFontaine = address(
            new BeaconProxy{ salt: keccak256(abi.encode(address(this), fontaineCount)) }(address(FONTAINE_BEACON), "")
        );

        // Transfer the total amount to unlock to the newly created Fontaine
        FLUID.transfer(newFontaine, amountToUnlock);

        // Persist the fontaine address and increment fontaine counter
        fontaines[fontaineCount] = IFontaine(newFontaine);
        fontaineCount++;

        // Initialize the new Fontaine instance (this initiate the unlock process)
        IFontaine(newFontaine).initialize(recipient, unlockFlowRate, taxFlowRate);

        emit FluidUnlocked(unlockPeriod, amountToUnlock, recipient, newFontaine);
    }

    function _calculateVestUnlockFlowRates(uint256 amountToUnlock, uint128 unlockPeriod)
        internal
        pure
        returns (int96 unlockFlowRate, int96 taxFlowRate)
    {
        int96 globalFlowRate = int256(amountToUnlock / unlockPeriod).toInt96();

        unlockFlowRate = (globalFlowRate * int256(_getUnlockingPercentage(unlockPeriod))).toInt96()
            / int256(_BP_DENOMINATOR).toInt96();
        taxFlowRate = globalFlowRate - unlockFlowRate;
    }

    function _getUnlockingPercentage(uint128 unlockPeriod) internal pure returns (uint256 unlockingPercentageBP) {
        unlockingPercentageBP = (
            _PERCENT_TO_BP
                * (
                    ((80 * _SCALER) / Math.sqrt(540 * _SCALER)) * (Math.sqrt(unlockPeriod * _SCALER) / _SCALER)
                        + 20 * _SCALER
                )
        ) / _SCALER;
    }

    //      __  ___          ___ _____
    //     /  |/  /___  ____/ (_) __(_)__  __________
    //    / /|_/ / __ \/ __  / / /_/ / _ \/ ___/ ___/
    //   / /  / / /_/ / /_/ / / __/ /  __/ /  (__  )
    //  /_/  /_/\____/\__,_/_/_/ /_/\___/_/  /____/

    /**
     * @dev Throws if called by any account other than the owner
     */
    modifier onlyLockerOwner() {
        if (msg.sender != lockerOwner) revert NOT_LOCKER_OWNER();
        _;
    }

    /**
     * @dev Throws if called operation is not available
     */
    modifier unlockAvailable() {
        if (!UNLOCK_AVAILABLE) revert TTE_NOT_ACTIVATED();
        _;
    }
}
