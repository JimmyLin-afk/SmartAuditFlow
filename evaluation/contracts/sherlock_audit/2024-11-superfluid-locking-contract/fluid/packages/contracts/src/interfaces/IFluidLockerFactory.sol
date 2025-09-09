// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

/**
 * @title Fluid Locker Factory Contract Interface
 * @author Superfluid
 * @notice Deploys new Fluid Locker contracts and their associated Fontaine
 *
 */
interface IFluidLockerFactory {
    //      ______                 __
    //     / ____/   _____  ____  / /______
    //    / __/ | | / / _ \/ __ \/ __/ ___/
    //   / /___ | |/ /  __/ / / / /_(__  )
    //  /_____/ |___/\___/_/ /_/\__/____/

    /// @notice Event emitted upon creation of a new locker
    event LockerCreated(address lockerOwner, address lockerAddress);

    /// @notice Event emitted upon governor address update
    event GovernorUpdated(address newGovernor);

    //     ______           __                     ______
    //    / ____/_  _______/ /_____  ____ ___     / ____/_____________  __________
    //   / /   / / / / ___/ __/ __ \/ __ `__ \   / __/ / ___/ ___/ __ \/ ___/ ___/
    //  / /___/ /_/ (__  ) /_/ /_/ / / / / / /  / /___/ /  / /  / /_/ / /  (__  )
    //  \____/\__,_/____/\__/\____/_/ /_/ /_/  /_____/_/  /_/   \____/_/  /____/

    /// @notice Error thrown when the attempting to create a locker while this operation is paused
    error LOCKER_CREATION_PAUSED();

    /// @notice Error thrown when the attempting to perform a governance protected operation
    error NOT_GOVERNOR();

    //      ______     __                        __   ______                 __  _
    //     / ____/  __/ /____  _________  ____ _/ /  / ____/_  ______  _____/ /_(_)___  ____  _____
    //    / __/ | |/_/ __/ _ \/ ___/ __ \/ __ `/ /  / /_  / / / / __ \/ ___/ __/ / __ \/ __ \/ ___/
    //   / /____>  </ /_/  __/ /  / / / / /_/ / /  / __/ / /_/ / / / / /__/ /_/ / /_/ / / / (__  )
    //  /_____/_/|_|\__/\___/_/  /_/ /_/\__,_/_/  /_/    \__,_/_/ /_/\___/\__/_/\____/_/ /_/____/

    /**
     * @notice Deploy a Locker and connected Fontaine for the caller
     * @return lockerInstance Deployed Locker contract address
     */
    function createLockerContract() external returns (address lockerInstance);

    /**
     * @notice Upgrade this proxy logic
     * @dev Only the governor address can perform this operation
     * @param newImplementation new logic contract address
     * @param data calldata for potential initializer
     */
    function upgradeTo(address newImplementation, bytes calldata data) external;

    /**
     * @notice Sets the governor address
     * @dev Only the governor address can perform this operation
     * @param newGovernor new governor address
     */
    function setGovernor(address newGovernor) external;

    //   _    ___                 ______                 __  _
    //  | |  / (_)__ _      __   / ____/_  ______  _____/ /_(_)___  ____  _____
    //  | | / / / _ \ | /| / /  / /_  / / / / __ \/ ___/ __/ / __ \/ __ \/ ___/
    //  | |/ / /  __/ |/ |/ /  / __/ / /_/ / / / / /__/ /_/ / /_/ / / / (__  )
    //  |___/_/\___/|__/|__/  /_/    \__,_/_/ /_/\___/\__/_/\____/_/ /_/____/

    /**
     * @notice Returns the users' locker address
     * @param user User address to be queried
     * @return isCreated True if the locker is created, false otherwise
     * @return lockerAddress The user's locker contract address
     */
    function getUserLocker(address user) external view returns (bool isCreated, address lockerAddress);

    /**
     * @notice Returns the locker contract address of the given user
     * @param user User address to be queried
     * @return lockerAddress The user's locker contract address
     */
    function getLockerAddress(address user) external view returns (address lockerAddress);

    /**
     * @notice Returns the locker beacon implementation contract address
     * @return lockerBeaconImpl The locker beacon implementation contract address
     */
    function getLockerBeaconImplementation() external view returns (address lockerBeaconImpl);
}
