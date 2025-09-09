// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

/* Openzeppelin Contracts & Interfaces */
import { BeaconProxy } from "@openzeppelin/contracts/proxy/beacon/BeaconProxy.sol";
import { UpgradeableBeacon } from "@openzeppelin/contracts/proxy/beacon/UpgradeableBeacon.sol";
import { ERC1967Utils } from "@openzeppelin-v5/contracts/proxy/ERC1967/ERC1967Utils.sol";
import { Initializable } from "@openzeppelin-v5/contracts/proxy/utils/Initializable.sol";

/* FLUID Contracts & Interfaces */
import { FluidLocker } from "./FluidLocker.sol";
import { Fontaine } from "./Fontaine.sol";
import { IFluidLockerFactory } from "./interfaces/IFluidLockerFactory.sol";
import { IStakingRewardController } from "./interfaces/IStakingRewardController.sol";

/**
 * @title Fluid Locker Factory Contract
 * @author Superfluid
 * @notice Deploys new Fluid Locker contracts and their associated Fontaine
 *
 */
contract FluidLockerFactory is Initializable, IFluidLockerFactory {
    //      ____                          __        __    __        _____ __        __
    //     /  _/___ ___  ____ ___  __  __/ /_____ _/ /_  / /__     / ___// /_____ _/ /____  _____
    //     / // __ `__ \/ __ `__ \/ / / / __/ __ `/ __ \/ / _ \    \__ \/ __/ __ `/ __/ _ \/ ___/
    //   _/ // / / / / / / / / / / /_/ / /_/ /_/ / /_/ / /  __/   ___/ / /_/ /_/ / /_/  __(__  )
    //  /___/_/ /_/ /_/_/ /_/ /_/\__,_/\__/\__,_/_.___/_/\___/   /____/\__/\__,_/\__/\___/____/

    /// @notice Locker Beacon contract address
    UpgradeableBeacon public immutable LOCKER_BEACON;

    /// @notice Staking Reward Controller interface
    IStakingRewardController public immutable STAKING_REWARD_CONTROLLER;

    /// @notice Pause Status of this contract
    bool public immutable IS_PAUSED;

    //     _____ __        __
    //    / ___// /_____ _/ /____  _____
    //    \__ \/ __/ __ `/ __/ _ \/ ___/
    //   ___/ / /_/ /_/ / /_/  __(__  )
    //  /____/\__/\__,_/\__/\___/____/

    /// @notice Governance Multisig address
    address public governor;

    /// @notice Stores the locker address of a given user address
    mapping(address user => address locker) private _lockers;

    //     ______                 __                  __
    //    / ____/___  ____  _____/ /________  _______/ /_____  _____
    //   / /   / __ \/ __ \/ ___/ __/ ___/ / / / ___/ __/ __ \/ ___/
    //  / /___/ /_/ / / / (__  ) /_/ /  / /_/ / /__/ /_/ /_/ / /
    //  \____/\____/_/ /_/____/\__/_/   \__,_/\___/\__/\____/_/

    /**
     * @notice FLUID Locker Factory contract constructor
     * @param lockerBeacon Locker Beacon contract address
     * @param stakingRewardController Staking Reward Controller interface contract address
     */
    constructor(address lockerBeacon, IStakingRewardController stakingRewardController, bool pauseStatus) {
        // Disable initializers to prevent implementation contract initalization
        _disableInitializers();

        // Sets the Staking Reward Controller interface
        STAKING_REWARD_CONTROLLER = stakingRewardController;

        // Sets the pause status
        IS_PAUSED = pauseStatus;

        // Sets the Locker Beacon address
        LOCKER_BEACON = UpgradeableBeacon(lockerBeacon);
    }

    /**
     * @notice FLUID Locker Factory contract initializer
     * @param _governor the governor address
     */
    function initialize(address _governor) external initializer {
        // Sets the governor address
        governor = _governor;
    }

    //      ______     __                        __   ______                 __  _
    //     / ____/  __/ /____  _________  ____ _/ /  / ____/_  ______  _____/ /_(_)___  ____  _____
    //    / __/ | |/_/ __/ _ \/ ___/ __ \/ __ `/ /  / /_  / / / / __ \/ ___/ __/ / __ \/ __ \/ ___/
    //   / /____>  </ /_/  __/ /  / / / / /_/ / /  / __/ / /_/ / / / / /__/ /_/ / /_/ / / / (__  )
    //  /_____/_/|_|\__/\___/_/  /_/ /_/\__,_/_/  /_/    \__,_/_/ /_/\___/\__/_/\____/_/ /_/____/

    /// @inheritdoc IFluidLockerFactory
    function createLockerContract() external notPaused returns (address lockerInstance) {
        lockerInstance = _createLockerContract(msg.sender);
    }

    /// @inheritdoc IFluidLockerFactory
    function upgradeTo(address newImplementation, bytes calldata data) external onlyGovernor {
        ERC1967Utils.upgradeToAndCall(newImplementation, data);
    }

    /// @inheritdoc IFluidLockerFactory
    function setGovernor(address newGovernor) external onlyGovernor {
        governor = newGovernor;
        emit GovernorUpdated(newGovernor);
    }

    //   _    ___                 ______                 __  _
    //  | |  / (_)__ _      __   / ____/_  ______  _____/ /_(_)___  ____  _____
    //  | | / / / _ \ | /| / /  / /_  / / / / __ \/ ___/ __/ / __ \/ __ \/ ___/
    //  | |/ / /  __/ |/ |/ /  / __/ / /_/ / / / / /__/ /_/ / /_/ / / / (__  )
    //  |___/_/\___/|__/|__/  /_/    \__,_/_/ /_/\___/\__/_/\____/_/ /_/____/

    /// @inheritdoc IFluidLockerFactory
    function getUserLocker(address user) external view returns (bool isCreated, address lockerAddress) {
        lockerAddress = getLockerAddress(user);
        isCreated = lockerAddress != address(0);
    }

    /// @inheritdoc IFluidLockerFactory
    function getLockerAddress(address user) public view returns (address lockerAddress) {
        lockerAddress = _lockers[user];
    }

    /// @inheritdoc IFluidLockerFactory
    function getLockerBeaconImplementation() public view returns (address lockerBeaconImpl) {
        lockerBeaconImpl = LOCKER_BEACON.implementation();
    }

    //      ____      __                        __   ______                 __  _
    //     /  _/___  / /____  _________  ____ _/ /  / ____/_  ______  _____/ /_(_)___  ____  _____
    //     / // __ \/ __/ _ \/ ___/ __ \/ __ `/ /  / /_  / / / / __ \/ ___/ __/ / __ \/ __ \/ ___/
    //   _/ // / / / /_/  __/ /  / / / / /_/ / /  / __/ / /_/ / / / / /__/ /_/ / /_/ / / / (__  )
    //  /___/_/ /_/\__/\___/_/  /_/ /_/\__,_/_/  /_/    \__,_/_/ /_/\___/\__/_/\____/_/ /_/____/

    /**
     * @notice Deploy a Locker Beacon Proxy with the hashed encoded LockerOwner as the salt
     * @param lockerOwner the owner of the Locker to be deployed
     */
    function _createLockerContract(address lockerOwner) internal returns (address lockerInstance) {
        lockerInstance =
            address(new BeaconProxy{ salt: keccak256(abi.encode(lockerOwner)) }(address(LOCKER_BEACON), ""));

        _lockers[lockerOwner] = lockerInstance;

        // Initialize the new Locker instance
        FluidLocker(lockerInstance).initialize(lockerOwner);

        // Approve the newly created locker to interact with the Staking Reward Controller
        STAKING_REWARD_CONTROLLER.approveLocker(lockerInstance);

        emit LockerCreated(lockerOwner, lockerInstance);
    }

    //      __  ___          ___ _____
    //     /  |/  /___  ____/ (_) __(_)__  __________
    //    / /|_/ / __ \/ __  / / /_/ / _ \/ ___/ ___/
    //   / /  / / /_/ / /_/ / / __/ /  __/ /  (__  )
    //  /_/  /_/\____/\__,_/_/_/ /_/\___/_/  /____/

    /**
     * @dev Throws if called by any account other than the Governor account
     */
    modifier onlyGovernor() {
        if (msg.sender != governor) revert NOT_GOVERNOR();
        _;
    }

    /**
     * @dev Throws if attempting to create a locker while this contract is paused
     */
    modifier notPaused() {
        if (IS_PAUSED) revert LOCKER_CREATION_PAUSED();
        _;
    }
}
