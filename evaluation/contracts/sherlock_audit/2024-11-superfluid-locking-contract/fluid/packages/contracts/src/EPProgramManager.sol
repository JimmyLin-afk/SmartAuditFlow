// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

/* Superfluid Protocol Contracts & Interfaces */
import {
    ISuperfluidPool,
    ISuperToken,
    PoolConfig
} from "@superfluid-finance/ethereum-contracts/contracts/interfaces/superfluid/ISuperfluid.sol";
import { SuperTokenV1Library } from "@superfluid-finance/ethereum-contracts/contracts/apps/SuperTokenV1Library.sol";

/* Solady ECDSA Library */
import { ECDSA } from "solady/utils/ECDSA.sol";

/* FLUID Interfaces */
import { IEPProgramManager } from "./interfaces/IEPProgramManager.sol";

using SuperTokenV1Library for ISuperToken;

/**
 * @title Ecosystem Partner Program Manager Contract
 * @author Superfluid x Stack
 * @notice Contract responsible for administrating the GDA pool that distribute tokens to Stack Points holders
 *
 */
contract EPProgramManager is IEPProgramManager {
    //      ____                          __        __    __        _____ __        __
    //     /  _/___ ___  ____ ___  __  __/ /_____ _/ /_  / /__     / ___// /_____ _/ /____  _____
    //     / // __ `__ \/ __ `__ \/ / / / __/ __ `/ __ \/ / _ \    \__ \/ __/ __ `/ __/ _ \/ ___/
    //   _/ // / / / / / / / / / / /_/ / /_/ /_/ / /_/ / /  __/   ___/ / /_/ /_/ / /_/  __(__  )
    //  /___/_/ /_/ /_/_/ /_/ /_/\__,_/\__/\__,_/_.___/_/\___/   /____/\__/\__,_/\__/\___/____/

    /// @notice Signature length requirement (r: 32 bytes, s: 32 bytes, v: 1 byte)
    uint256 private constant _SIGNATURE_LENGTH = 65;

    //     _____ __        __
    //    / ___// /_____ _/ /____  _____
    //    \__ \/ __/ __ `/ __/ _ \/ ___/
    //   ___/ / /_/ /_/ / /_/  __(__  )
    //  /____/\__/\__,_/\__/\___/____/

    /// @notice Stores the program details for a given program identifier
    mapping(uint256 programId => EPProgram program) public programs;

    /// @notice Stores the last valid nonce used for a given program identifier and a given user
    mapping(uint256 programId => mapping(address user => uint256 lastValidNonce)) private _lastValidNonces;

    //      ______     __                        __   ______                 __  _
    //     / ____/  __/ /____  _________  ____ _/ /  / ____/_  ______  _____/ /_(_)___  ____  _____
    //    / __/ | |/_/ __/ _ \/ ___/ __ \/ __ `/ /  / /_  / / / / __ \/ ___/ __/ / __ \/ __ \/ ___/
    //   / /____>  </ /_/  __/ /  / / / / /_/ / /  / __/ / /_/ / / / / /__/ /_/ / /_/ / / / (__  )
    //  /_____/_/|_|\__/\___/_/  /_/ /_/\__,_/_/  /_/    \__,_/_/ /_/\___/\__/_/\____/_/ /_/____/

    /// @inheritdoc IEPProgramManager
    function createProgram(uint256 programId, address programAdmin, address signer, ISuperToken token)
        external
        virtual
        returns (ISuperfluidPool distributionPool)
    {
        // Input validation
        if (programId == 0) revert INVALID_PARAMETER();
        if (programAdmin == address(0)) revert INVALID_PARAMETER();
        if (signer == address(0)) revert INVALID_PARAMETER();
        if (address(token) == address(0)) revert INVALID_PARAMETER();
        if (address(programs[programId].distributionPool) != address(0)) {
            revert PROGRAM_ALREADY_CREATED();
        }

        // Configure Superfluid GDA Pool
        PoolConfig memory poolConfig =
            PoolConfig({ transferabilityForUnitsOwner: false, distributionFromAnyAddress: true });

        // Create Superfluid GDA Pool
        distributionPool = token.createPool(address(this), poolConfig);

        // Persist program details
        programs[programId] = EPProgram({
            programAdmin: programAdmin,
            stackSigner: signer,
            token: token,
            distributionPool: distributionPool
        });

        emit IEPProgramManager.ProgramCreated(
            programId, programAdmin, signer, address(token), address(distributionPool)
        );
    }

    /// @inheritdoc IEPProgramManager
    function updateProgramSigner(uint256 programId, address newSigner)
        external
        programExists(programId)
        onlyProgramAdmin(programId)
    {
        if (newSigner == address(0)) revert INVALID_PARAMETER();

        // Update the program signer
        programs[programId].stackSigner = newSigner;

        emit IEPProgramManager.ProgramSignerUpdated(programId, newSigner);
    }

    /// @inheritdoc IEPProgramManager
    function updateUnits(uint256 programId, uint256 newUnits, uint256 nonce, bytes memory stackSignature) external {
        updateUserUnits(msg.sender, programId, newUnits, nonce, stackSignature);
    }

    /// @inheritdoc IEPProgramManager
    function batchUpdateUnits(
        uint256[] memory programIds,
        uint256[] memory newUnits,
        uint256[] memory nonces,
        bytes[] memory stackSignatures
    ) external {
        batchUpdateUserUnits(msg.sender, programIds, newUnits, nonces, stackSignatures);
    }

    /// @inheritdoc IEPProgramManager
    function updateUserUnits(
        address user,
        uint256 programId,
        uint256 newUnits,
        uint256 nonce,
        bytes memory stackSignature
    ) public programExists(programId) {
        // Input validation
        if (user == address(0)) revert INVALID_PARAMETER();
        if (stackSignature.length != _SIGNATURE_LENGTH) {
            revert INVALID_SIGNATURE("signature length");
        }

        // Verify and update nonce
        if (!_isNonceValid(programId, user, nonce)) {
            revert INVALID_SIGNATURE("nonce");
        }
        _lastValidNonces[programId][user] = nonce;

        EPProgram memory program = programs[programId];

        // Verify signature
        if (!_verifySignature(program.stackSigner, user, newUnits, programId, nonce, stackSignature)) {
            revert INVALID_SIGNATURE("signer");
        }

        // Update units in pool
        _poolUpdate(program, newUnits, user);

        emit UserUnitsUpdated(user, programId, newUnits);
    }

    /// @inheritdoc IEPProgramManager
    function batchUpdateUserUnits(
        address user,
        uint256[] memory programIds,
        uint256[] memory newUnits,
        uint256[] memory nonces,
        bytes[] memory stackSignatures
    ) public {
        uint256 length = programIds.length;

        // Validate array sizes
        if (length == 0) revert INVALID_PARAMETER();
        if (length != newUnits.length || length != nonces.length || length != stackSignatures.length) {
            revert INVALID_PARAMETER();
        }

        for (uint256 i; i < length; ++i) {
            updateUserUnits(user, programIds[i], newUnits[i], nonces[i], stackSignatures[i]);
        }
    }

    //   _    ___                 ______                 __  _
    //  | |  / (_)__ _      __   / ____/_  ______  _____/ /_(_)___  ____  _____
    //  | | / / / _ \ | /| / /  / /_  / / / / __ \/ ___/ __/ / __ \/ __ \/ ___/
    //  | |/ / /  __/ |/ |/ /  / __/ / /_/ / / / / /__/ /_/ / /_/ / / / (__  )
    //  |___/_/\___/|__/|__/  /_/    \__,_/_/ /_/\___/\__/_/\____/_/ /_/____/

    /// @inheritdoc IEPProgramManager
    function getProgramPool(uint256 programId) external view returns (ISuperfluidPool programPool) {
        programPool = programs[programId].distributionPool;
    }

    /// @inheritdoc IEPProgramManager
    function getNextValidNonce(uint256 programId, address user) external view returns (uint256 validNonce) {
        validNonce = _lastValidNonces[programId][user] + 1;
    }

    //      ____      __                        __   ______                 __  _
    //     /  _/___  / /____  _________  ____ _/ /  / ____/_  ______  _____/ /_(_)___  ____  _____
    //     / // __ \/ __/ _ \/ ___/ __ \/ __ `/ /  / /_  / / / / __ \/ ___/ __/ / __ \/ __ \/ ___/
    //   _/ // / / / /_/  __/ /  / / / / /_/ / /  / __/ / /_/ / / / / /__/ /_/ / /_/ / / / (__  )
    //  /___/_/ /_/\__/\___/_/  /_/ /_/\__,_/_/  /_/    \__,_/_/ /_/\___/\__/_/\____/_/ /_/____/

    /**
     * @notice Update GDA Pool units
     * @dev This function can be overriden if there is a need to convert the stackPoints into GDA pool units
     * @param program The program associated with the update
     * @param stackPoints Amount of stack points
     * @param user The user address to receive the units
     */
    function _poolUpdate(EPProgram memory program, uint256 stackPoints, address user) internal virtual {
        program.distributionPool.updateMemberUnits(user, uint128(stackPoints));
    }

    /**
     * @notice Checks if a nonce is valid for a given program and user
     * @dev A nonce is valid if it's greater than the last used nonce
     * @param programId The program identifier
     * @param user The user address
     * @param nonce The nonce to validate
     * @return isValid True if the nonce is valid
     */
    function _isNonceValid(uint256 programId, address user, uint256 nonce) internal view returns (bool isValid) {
        isValid = nonce > _lastValidNonces[programId][user];
    }

    /**
     * @notice Verifies a signature for updating units
     *  @param signer The expected signer address
     *  @param user The user whose units are being updated
     *  @param newUnits The new units value
     *  @param programId The program identifier
     *  @param nonce The nonce used in the signature
     *  @param signature The signature to verify
     *  @return isValid True if the signature is valid
     */
    function _verifySignature(
        address signer,
        address user,
        uint256 newUnits,
        uint256 programId,
        uint256 nonce,
        bytes memory signature
    ) internal view returns (bool isValid) {
        bytes32 hash = ECDSA.toEthSignedMessageHash(keccak256(abi.encodePacked(user, newUnits, programId, nonce)));

        isValid = ECDSA.recover(hash, signature) == signer;
    }

    //      __  ___          ___ _____
    //     /  |/  /___  ____/ (_) __(_)__  __________
    //    / /|_/ / __ \/ __  / / /_/ / _ \/ ___/ ___/
    //   / /  / / /_/ / /_/ / / __/ /  __/ /  (__  )
    //  /_/  /_/\____/\__,_/_/_/ /_/\___/_/  /____/

    /**
     * @notice Ensures the program exists
     * @param programId identifier of the program to check
     */
    modifier programExists(uint256 programId) {
        if (address(programs[programId].distributionPool) == address(0)) {
            revert PROGRAM_NOT_FOUND();
        }
        _;
    }

    /**
     * @notice Ensures the caller is the program admin
     * @param programId identifier of the program to check
     */
    modifier onlyProgramAdmin(uint256 programId) {
        if (msg.sender != programs[programId].programAdmin) {
            revert NOT_PROGRAM_ADMIN();
        }
        _;
    }
}
