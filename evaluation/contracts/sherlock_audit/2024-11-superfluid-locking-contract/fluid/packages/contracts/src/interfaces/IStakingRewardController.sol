// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import { ISuperfluidPool } from "@superfluid-finance/ethereum-contracts/contracts/interfaces/superfluid/ISuperfluid.sol";

/**
 * @title Staking Reward Controller Contract Interface
 * @author Superfluid
 * @notice Contract responsible for administrating the GDA pool that distribute the unlocking tax to stakers
 *
 */
interface IStakingRewardController {
    //      ______                 __
    //     / ____/   _____  ____  / /______
    //    / __/ | | / / _ \/ __ \/ __/ ___/
    //   / /___ | |/ /  __/ / / / /_(__  )
    //  /_____/ |___/\___/_/ /_/\__/____/

    /// @notice Event emitted when locker updates their units from staking or unstaking
    event UpdatedStakersUnits(address indexed staker, uint128 indexed totalStakerUnits);

    /// @notice Event emitted when the subsidy flowrate is updated
    event SubsidyFlowRateUpdated(int96 indexed newSubsidyFlowRate);

    /// @notice Event emitted when the Locker Factory address is updated
    event LockerFactoryAddressUpdated(address indexed newLockerFactoryAddress);

    /// @notice Event emitted when the Locker Factory address is updated
    event ProgramManagerAddressUpdated(address indexed newProgramManagerAddress);

    /// @notice Event emitted when a Locker is approved
    event LockerApproved(address indexed approvedLocker);

    //     ______           __                     ______
    //    / ____/_  _______/ /_____  ____ ___     / ____/_____________  __________
    //   / /   / / / / ___/ __/ __ \/ __ `__ \   / __/ / ___/ ___/ __ \/ ___/ ___/
    //  / /___/ /_/ (__  ) /_/ /_/ / / / / / /  / /___/ /  / /  / /_/ / /  (__  )
    //  \____/\__,_/____/\__/\____/_/ /_/ /_/  /_____/_/  /_/   \____/_/  /____/

    /// @notice Error thrown when the caller is not an approved Locker
    error NOT_APPROVED_LOCKER();

    /// @notice Error thrown when the caller is not the Locker Factory contract
    error NOT_LOCKER_FACTORY();

    /// @notice Error thrown when the caller is not the Program Manager contract
    error NOT_PROGRAM_MANAGER();

    //      ______     __                        __   ______                 __  _
    //     / ____/  __/ /____  _________  ____ _/ /  / ____/_  ______  _____/ /_(_)___  ____  _____
    //    / __/ | |/_/ __/ _ \/ ___/ __ \/ __ `/ /  / /_  / / / / __ \/ ___/ __/ / __ \/ __ \/ ___/
    //   / /____>  </ /_/  __/ /  / / / / /_/ / /  / __/ / /_/ / / / / /__/ /_/ / /_/ / / / (__  )
    //  /_____/_/|_|\__/\___/_/  /_/ /_/\__,_/_/  /_/    \__,_/_/ /_/\___/\__/_/\____/_/ /_/____/

    /**
     * @notice Update the caller's (staker) units within the GDA Penalty Pool
     * @dev Only approved lockers can perform this operation
     * @param lockerStakedBalance locker's new staked balance amount
     */
    function updateStakerUnits(uint256 lockerStakedBalance) external;

    /**
     * @notice Update the Locker Factory contract address
     * @dev Only the contract owner can perform this operation
     * @param lockerFactoryAddress Locker Factory contract address to be set
     */
    function setLockerFactory(address lockerFactoryAddress) external;

    /**
     * @notice Approve a Locker to interact with the Staking Reward Controller contract
     * @dev Only the Locker Factory contract can perform this operation
     * @param lockerAddress Locker contract address to be approved
     */
    function approveLocker(address lockerAddress) external;

    /**
     * @notice Upgrade this proxy logic
     * @dev Only the owner address can perform this operation
     * @param newImplementation new logic contract address
     * @param data calldata for potential initializer
     */
    function upgradeTo(address newImplementation, bytes calldata data) external;
}
