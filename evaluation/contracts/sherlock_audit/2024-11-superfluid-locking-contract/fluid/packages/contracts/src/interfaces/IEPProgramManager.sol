// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

/* Superfluid Protocol Contracts & Interfaces */
import {
    ISuperfluidPool,
    ISuperToken
} from "@superfluid-finance/ethereum-contracts/contracts/interfaces/superfluid/ISuperfluid.sol";

/**
 * @title Ecosystem Partner Program Manager Contract Interface
 * @author Superfluid
 * @notice Contract responsible for administrating the GDA pool that distribute FLUID to lockers
 *
 */
interface IEPProgramManager {
    //      ______                 __
    //     / ____/   _____  ____  / /______
    //    / __/ | | / / _ \/ __ \/ __/ ___/
    //   / /___ | |/ /  __/ / / / /_(__  )
    //  /_____/ |___/\___/_/ /_/\__/____/

    /// @notice Event emitted when a new reward program is created
    event ProgramCreated(
        uint256 programId, address programAdmin, address signer, address token, address distributionPool
    );

    /// @notice Event emitted when a reward program signer is updated
    event ProgramSignerUpdated(uint256 programId, address newSigner);

    /// @notice Event emitted when user's units are updated
    event UserUnitsUpdated(address user, uint256 programId, uint256 newUnits);

    //      ____        __        __
    //     / __ \____ _/ /_____ _/ /___  ______  ___  _____
    //    / / / / __ `/ __/ __ `/ __/ / / / __ \/ _ \/ ___/
    //   / /_/ / /_/ / /_/ /_/ / /_/ /_/ / /_/ /  __(__  )
    //  /_____/\__,_/\__/\__,_/\__/\__, / .___/\___/____/
    //                            /____/_/

    /**
     * @notice Ecosystem Partner Program Data Type
     * @param programAdmin program admin address
     * @param stackSigner signer address
     * @param token SuperToken to be distributed
     * @return distributionPool deployed Superfluid Pool contract address
     */
    struct EPProgram {
        address programAdmin;
        address stackSigner;
        ISuperToken token;
        ISuperfluidPool distributionPool;
    }

    //     ______           __                     ______
    //    / ____/_  _______/ /_____  ____ ___     / ____/_____________  __________
    //   / /   / / / / ___/ __/ __ \/ __ `__ \   / __/ / ___/ ___/ __ \/ ___/ ___/
    //  / /___/ /_/ (__  ) /_/ /_/ / / / / / /  / /___/ /  / /  / /_/ / /  (__  )
    //  \____/\__,_/____/\__/\____/_/ /_/ /_/  /_____/_/  /_/   \____/_/  /____/

    /// @notice Error thrown when attempting to create a program with an alredy existsing program identifier
    /// @dev Error Selector : 0x3a33ee23
    error PROGRAM_ALREADY_CREATED();

    /// @notice Error thrown when caller is not the program admin
    /// @dev Error Selector : 0x4c7f89d7
    error NOT_PROGRAM_ADMIN();

    /// @notice Error thrown when attempting an operation on a non-existent program
    error PROGRAM_NOT_FOUND();

    /// @notice Error thrown when passing an invalid parameter
    /// @dev Error Selector : 0x4c4f685a
    error INVALID_PARAMETER();

    /// @notice Error thrown when a signature is of invalid
    /// @dev Error Selector : 0x30f01ccf
    /// @param reason Description of what part of the signature was invalid
    error INVALID_SIGNATURE(string reason);

    //      ______     __                        __   ______                 __  _
    //     / ____/  __/ /____  _________  ____ _/ /  / ____/_  ______  _____/ /_(_)___  ____  _____
    //    / __/ | |/_/ __/ _ \/ ___/ __ \/ __ `/ /  / /_  / / / / __ \/ ___/ __/ / __ \/ __ \/ ___/
    //   / /____>  </ /_/  __/ /  / / / / /_/ / /  / __/ / /_/ / / / / /__/ /_/ / /_/ / / / (__  )
    //  /_____/_/|_|\__/\___/_/  /_/ /_/\__,_/_/  /_/    \__,_/_/ /_/\___/\__/_/\____/_/ /_/____/

    /**
     * @notice Creates a new distribution program
     * @param programId program identifier to be created
     * @param programAdmin program admin address
     * @param signer signer address
     * @param token SuperToken to be distributed
     * @return distributionPool deployed Superfluid Pool contract address
     */
    function createProgram(uint256 programId, address programAdmin, address signer, ISuperToken token)
        external
        returns (ISuperfluidPool distributionPool);

    /**
     * @notice Update program signer
     * @dev Only the program admin can perform this operation
     * @param programId program identifier to be updated
     * @param newSigner new signer address
     */
    function updateProgramSigner(uint256 programId, address newSigner) external;

    /**
     * @notice Update units within the distribution pool associated to the given program
     * @param programId program identifier associated to the distribution pool
     * @param newUnits unit amount to be granted
     * @param nonce nonce corresponding to the stack signature
     * @param stackSignature stack signature containing necessary info to update units
     */
    function updateUnits(uint256 programId, uint256 newUnits, uint256 nonce, bytes memory stackSignature) external;

    /**
     * @notice Batch update units within the distribution pools associated to the given programs
     * @param programIds array of program identifiers associated to the distribution pool
     * @param newUnits array of unit amounts to be granted
     * @param nonces array nonces corresponding to the stack signatures
     * @param stackSignatures array of stack signatures containing necessary info to update units
     */
    function batchUpdateUnits(
        uint256[] memory programIds,
        uint256[] memory newUnits,
        uint256[] memory nonces,
        bytes[] memory stackSignatures
    ) external;

    /**
     * @notice Update units within the distribution pool associated to the given program
     * @param user address to grants the units to
     * @param programId program identifier associated to the distribution pool
     * @param newUnits unit amount to be granted
     * @param nonce nonce corresponding to the stack signature
     * @param stackSignature stack signature containing necessary info to update units
     */
    function updateUserUnits(
        address user,
        uint256 programId,
        uint256 newUnits,
        uint256 nonce,
        bytes memory stackSignature
    ) external;

    /**
     * @notice Batch update units within the distribution pools associated to the given programs
     * @param user address to grants the units to
     * @param programIds array of program identifiers associated to the distribution pool
     * @param newUnits array of unit amounts to be granted
     * @param nonces array nonces corresponding to the stack signatures
     * @param stackSignatures array of stack signatures containing necessary info to update units
     */
    function batchUpdateUserUnits(
        address user,
        uint256[] memory programIds,
        uint256[] memory newUnits,
        uint256[] memory nonces,
        bytes[] memory stackSignatures
    ) external;

    //   _    ___                 ______                 __  _
    //  | |  / (_)__ _      __   / ____/_  ______  _____/ /_(_)___  ____  _____
    //  | | / / / _ \ | /| / /  / /_  / / / / __ \/ ___/ __/ / __ \/ __ \/ ___/
    //  | |/ / /  __/ |/ |/ /  / __/ / /_/ / / / / /__/ /_/ / /_/ / / / (__  )
    //  |___/_/\___/|__/|__/  /_/    \__,_/_/ /_/\___/\__/_/\____/_/ /_/____/

    /**
     * @notice Returns the distribution pool associated to the given program identifier
     * @param programId program identifier to be queried
     * @return programPool the GDA pool interface associated to the program identifier
     */
    function getProgramPool(uint256 programId) external view returns (ISuperfluidPool programPool);

    /**
     * @notice Returns the next valid nonce for the given user and the given program identifier
     * @param programId program identifier to be queried
     * @param user user to be queried
     * @return validNonce the next valid nonce for the given user and the given program identifier
     */
    function getNextValidNonce(uint256 programId, address user) external view returns (uint256 validNonce);
}
