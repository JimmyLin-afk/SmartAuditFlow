// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

/**
 * @title Locker Contract Interface
 * @author Superfluid
 * @notice Contract responsible for locking and holding FLUID token on behalf of users
 *
 */
interface IFluidLocker {
    //      ______                 __
    //     / ____/   _____  ____  / /______
    //    / __/ | | / / _ \/ __ \/ __/ ___/
    //   / /___ | |/ /  __/ / / / /_(__  )
    //  /_____/ |___/\___/_/ /_/\__/____/

    /// @notice Event emitted when locker owner claims units
    event FluidStreamClaimed(uint256 indexed programId, uint256 indexed totalProgramUnits);

    /// @notice Event emitted when locker owner batch claims units
    event FluidStreamsClaimed(uint256[] indexed programId, uint256[] indexed totalProgramUnits);

    /// @notice Event emitted when new $FLUID are locked into the locker
    event FluidLocked(uint256 indexed amount);

    /// @notice Event emitted when $FLUID are unlocked from the locker
    event FluidUnlocked(
        uint128 indexed unlockPeriod, uint256 indexed availableBalance, address recipient, address indexed fontaine
    );

    /// @notice Event emitted when $FLUID are staked
    event FluidStaked(uint256 indexed newTotalStakedBalance, uint256 indexed addedAmount);

    /// @notice Event emitted when $FLUID are unstaked
    event FluidUnstaked();

    //     ______           __                     ______
    //    / ____/_  _______/ /_____  ____ ___     / ____/_____________  __________
    //   / /   / / / / ___/ __/ __ \/ __ `__ \   / __/ / ___/ ___/ __ \/ ___/ ___/
    //  / /___/ /_/ (__  ) /_/ /_/ / / / / / /  / /___/ /  / /  / /_/ / /  (__  )
    //  \____/\__,_/____/\__/\____/_/ /_/ /_/  /_____/_/  /_/   \____/_/  /____/

    /// @notice Error thrown when the caller is not the owner
    error NOT_LOCKER_OWNER();

    /// @notice Error thrown when attempting to perform a forbidden operation
    error FORBIDDEN();

    /// @notice Error thrown when attempting to unlock FLUID from a locker with an invalid unlock period
    error INVALID_UNLOCK_PERIOD();

    /// @notice Error thrown when attempting to unlock FLUID from a locker that does not have available $FLUID
    error NO_FLUID_TO_UNLOCK();

    /// @notice Error thrown when attempting to unstake from locker that does not have staked $FLUID
    error NO_FLUID_TO_UNSTAKE();

    /// @notice Error thrown when attempting to stake from locker that does not have available $FLUID
    error NO_FLUID_TO_STAKE();

    /// @notice Error thrown when attempting to unstake while the staking cooldown is not yet elapsed
    error STAKING_COOLDOWN_NOT_ELAPSED();

    /// @notice Error thrown when attempting to unlock or stake while this operation is not yet available
    error TTE_NOT_ACTIVATED();

    //      ______     __                        __   ______                 __  _
    //     / ____/  __/ /____  _________  ____ _/ /  / ____/_  ______  _____/ /_(_)___  ____  _____
    //    / __/ | |/_/ __/ _ \/ ___/ __ \/ __ `/ /  / /_  / / / / __ \/ ___/ __/ / __ \/ __ \/ ___/
    //   / /____>  </ /_/  __/ /  / / / / /_/ / /  / __/ / /_/ / / / / /__/ /_/ / /_/ / / / (__  )
    //  /_____/_/|_|\__/\___/_/  /_/ /_/\__,_/_/  /_/    \__,_/_/ /_/\___/\__/_/\____/_/ /_/____/

    /**
     * @notice Update this locker units within the given program identifier's GDA pool
     * @param programId program identifier corresponding to the unit update
     * @param totalProgramUnits new total amount of units
     * @param nonce nonce associated to the signature provided by Stack
     * @param stackSignature stack signature containing necessary info to update units
     */
    function claim(uint256 programId, uint256 totalProgramUnits, uint256 nonce, bytes memory stackSignature) external;

    /**
     * @notice Batch update this locker units within the given programs identifier's GDA pools
     * @param programIds array of program identifiers corresponding to the unit update
     * @param totalProgramUnits array new total amount of units
     * @param nonces array of nonce associated to the signatures provided by Stack
     * @param stackSignatures array of stack signatures containing necessary info to update units
     */
    function claim(
        uint256[] memory programIds,
        uint256[] memory totalProgramUnits,
        uint256[] memory nonces,
        bytes[] memory stackSignatures
    ) external;

    /**
     * @notice Lock a given `amount` of FLUID Token in this Locker
     * @dev Requires preliminary token approval
     * @param amount amount of FLUID Token to lock
     */
    function lock(uint256 amount) external;

    /**
     * @notice Unlock the available FLUID Token from this locker
     * @dev Only this Locker owner can call this function
     * @param unlockPeriod the desired unlocking period (instant unlock if sets to 0)
     * @param recipient account to receive the unlocked FLUID tokens
     */
    function unlock(uint128 unlockPeriod, address recipient) external;

    /**
     * @notice Stake all the available FLUID Token of this locker
     * @dev Only this Locker owner can call this function
     */
    function stake() external;

    /**
     * @notice Unstake all the staked FLUID Token of this locker
     * @dev Only this Locker owner can call this function
     */
    function unstake() external;

    //   _    ___                 ______                 __  _
    //  | |  / (_)__ _      __   / ____/_  ______  _____/ /_(_)___  ____  _____
    //  | | / / / _ \ | /| / /  / /_  / / / / __ \/ ___/ __/ / __ \/ __ \/ ___/
    //  | |/ / /  __/ |/ |/ /  / __/ / /_/ / / / / /__/ /_/ / /_/ / / / (__  )
    //  |___/_/\___/|__/|__/  /_/    \__,_/_/ /_/\___/\__/_/\____/_/ /_/____/

    /**
     * @notice Returns the flowrate received by this Locker for the given program identifier
     * @param programId program identifier to query
     * @return flowRate the flowrate received by this Locker for the given program identifier
     */
    function getFlowRatePerProgram(uint256 programId) external view returns (int96 flowRate);

    /**
     * @notice Returns the flowrates received by this Locker for the given program identifiers
     * @param programIds array of program identifiers to query
     * @return flowRates array of flowrate received by this Locker for the given program identifiers
     */
    function getFlowRatePerProgram(uint256[] memory programIds) external view returns (int96[] memory flowRates);

    /**
     * @notice Returns the amount of GDA units owned by this Locker for the given program identifier
     * @param programId program identifier to query
     * @return units the amount of GDA units owned by this Locker for the given program identifier
     */
    function getUnitsPerProgram(uint256 programId) external view returns (uint128 units);

    /**
     * @notice Returns the amounts of GDA units owned by this Locker for the given program identifiers
     * @param programIds array of program identifiers to query
     * @return units array of GDA units amount owned by this Locker for the given program identifiers
     */
    function getUnitsPerProgram(uint256[] memory programIds) external view returns (uint128[] memory units);

    /**
     * @notice Returns this Lockers' staked FLUID Token balance
     * @return sBalance amount of FLUID Token staked in this Locker
     */
    function getStakedBalance() external view returns (uint256 sBalance);

    /**
     * @notice Returns this Lockers' available FLUID Token balance
     * @dev Available balance is the total balance minus the staked balance
     * @return aBalance amount of FLUID Token available in this Locker
     */
    function getAvailableBalance() external view returns (uint256 aBalance);

    /**
     * @notice Returns the fontaine beacon implementation contract address
     * @return fontaineBeaconImpl The fontaine beacon implementation contract address
     */
    function getFontaineBeaconImplementation() external view returns (address fontaineBeaconImpl);
}
