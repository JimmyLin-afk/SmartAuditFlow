// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

/* Openzeppelin Contracts & Interfaces */
import { OwnableUpgradeable } from "@openzeppelin/contracts-upgradeable/contracts/access/OwnableUpgradeable.sol";
import { Initializable } from "@openzeppelin/contracts-upgradeable/contracts/proxy/utils/Initializable.sol";
import { ERC1967Utils } from "@openzeppelin-v5/contracts/proxy/ERC1967/ERC1967Utils.sol";
import { SafeCast } from "@openzeppelin-v5/contracts/utils/math/SafeCast.sol";

/* Superfluid Protocol Contracts & Interfaces */
import {
    ISuperToken,
    ISuperfluidPool,
    PoolConfig
} from "@superfluid-finance/ethereum-contracts/contracts/interfaces/superfluid/ISuperfluid.sol";
import { SuperTokenV1Library } from "@superfluid-finance/ethereum-contracts/contracts/apps/SuperTokenV1Library.sol";

/* FLUID Contracts & Interfaces */
import { EPProgramManager, IEPProgramManager } from "./EPProgramManager.sol";
import { IFluidLockerFactory } from "./interfaces/IFluidLockerFactory.sol";

using SuperTokenV1Library for ISuperToken;
using SafeCast for int256;

/**
 * @title Superfluid Ecosystem Partner Program Manager Contract
 * @author Superfluid
 * @notice Contract responsible for administrating the GDA pool that distribute FLUID to lockers
 *
 */
contract FluidEPProgramManager is Initializable, OwnableUpgradeable, EPProgramManager {
    //      ______                 __
    //     / ____/   _____  ____  / /______
    //    / __/ | | / / _ \/ __ \/ __/ ___/
    //   / /___ | |/ /  __/ / / / /_(__  )
    //  /_____/ |___/\___/_/ /_/\__/____/

    /// @notice Event emitted when a reward program is cancelled
    event ProgramCancelled(
        uint256 indexed programId,
        uint256 indexed undistributedFundingAmount,
        uint256 indexed undistributedSubsidyAmount
    );

    /// @notice Event emitted when a reward program is funded
    event ProgramFunded(
        uint256 indexed programId,
        uint256 indexed fundingAmount,
        uint256 indexed subsidyAmount,
        uint256 earlyEndDate,
        uint256 endDate
    );

    /// @notice Event emitted when a reward program is stopped
    event ProgramStopped(
        uint256 indexed programId, uint256 indexed fundingCompensationAmount, uint256 indexed subsidyCompensationAmount
    );

    //      ____        __        __
    //     / __ \____ _/ /_____ _/ /___  ______  ___  _____
    //    / / / / __ `/ __/ __ `/ __/ / / / __ \/ _ \/ ___/
    //   / /_/ / /_/ / /_/ /_/ / /_/ /_/ / /_/ /  __(__  )
    //  /_____/\__,_/\__/\__,_/\__/\__, / .___/\___/____/
    //                            /____/_/

    /**
     * @notice Fluid Program related details Data Type
     * @param fundingFlowRate flow rate between this contract and the program pool
     * @param subsidyFlowRate flow rate between the Staking Reward Controller contract and the tax distribution pool
     * @param fundingStartDate timestamp at which the program is funded
     */
    struct FluidProgramDetails {
        int96 fundingFlowRate;
        int96 subsidyFlowRate;
        uint64 fundingStartDate;
    }

    //     ______           __                     ______
    //    / ____/_  _______/ /_____  ____ ___     / ____/_____________  __________
    //   / /   / / / / ___/ __/ __ \/ __ `__ \   / __/ / ___/ ___/ __ \/ ___/ ___/
    //  / /___/ /_/ (__  ) /_/ /_/ / / / / / /  / /___/ /  / /  / /_/ / /  (__  )
    //  \____/\__,_/____/\__/\____/_/ /_/ /_/  /_____/_/  /_/   \____/_/  /____/

    /// @notice Error thrown when attempting to add units to a non-existant locker
    /// @dev Error Selector : 0x2d2cd50c
    error LOCKER_NOT_FOUND();

    /// @notice Error thrown when attempting to stop a program's funding earlier than expected
    /// @dev Error Selector : 0xc582137f
    error TOO_EARLY_TO_END_PROGRAM();

    //      ____                          __        __    __        _____ __        __
    //     /  _/___ ___  ____ ___  __  __/ /_____ _/ /_  / /__     / ___// /_____ _/ /____  _____
    //     / // __ `__ \/ __ `__ \/ / / / __/ __ `/ __ \/ / _ \    \__ \/ __/ __ `/ __/ _ \/ ___/
    //   _/ // / / / / / / / / / / /_/ / /_/ /_/ / /_/ / /  __/   ___/ / /_/ /_/ / /_/  __(__  )
    //  /___/_/ /_/ /_/_/ /_/ /_/\__,_/\__/\__,_/_.___/_/\___/   /____/\__/\__,_/\__/\___/____/

    /// @notice Staking Reward Controller contract interface
    ISuperfluidPool public immutable TAX_DISTRIBUTION_POOL;

    /// @notice Program Duration used to calculate flow rates
    uint256 public constant PROGRAM_DURATION = 90 days;

    /// @notice Constant used to calculate the earliest date a program can be stopped
    uint256 public constant EARLY_PROGRAM_END = 7 days;

    /// @notice Basis points denominator (for percentage calculation)
    uint96 private constant _BP_DENOMINATOR = 10_000;

    //     _____ __        __
    //    / ___// /_____ _/ /____  _____
    //    \__ \/ __/ __ `/ __/ _ \/ ___/
    //   ___/ / /_/ /_/ / /_/  __(__  )
    //  /____/\__/\__,_/\__/\___/____/

    /// @notice Stores the program details of a given program
    mapping(uint256 programId => FluidProgramDetails programDetails) private _fluidProgramDetails;

    /// @notice Staking subsidy funding rate
    uint96 public subsidyFundingRate;

    /// @notice Fluid Locker Factory interface
    IFluidLockerFactory public fluidLockerFactory;

    /// @notice Fluid treasury account address
    address public fluidTreasury;

    //     ______                 __                  __
    //    / ____/___  ____  _____/ /________  _______/ /_____  _____
    //   / /   / __ \/ __ \/ ___/ __/ ___/ / / / ___/ __/ __ \/ ___/
    //  / /___/ /_/ / / / (__  ) /_/ /  / /_/ / /__/ /_/ /_/ / /
    //  \____/\____/_/ /_/____/\__/_/   \__,_/\___/\__/\____/_/

    /**
     * @notice Superfluid Ecosystem Partner Program Manager constructor
     * @param taxDistributionPool Tax Distribution Pool GDA contract interface
     */
    constructor(ISuperfluidPool taxDistributionPool) {
        // Disable initializers to prevent implementation contract initalization
        _disableInitializers();

        TAX_DISTRIBUTION_POOL = taxDistributionPool;
    }

    /**
     * @notice Superfluid Ecosystem Partner Program Manager initializer
     * @param owner contract owner address
     * @param treasury fluid treasury address
     */
    function initialize(address owner, address treasury) external initializer {
        // Initialize Ownable
        __Ownable_init(owner);

        // Sets the treasury address
        fluidTreasury = treasury;
    }

    //      ______     __                        __   ______                 __  _
    //     / ____/  __/ /____  _________  ____ _/ /  / ____/_  ______  _____/ /_(_)___  ____  _____
    //    / __/ | |/_/ __/ _ \/ ___/ __ \/ __ `/ /  / /_  / / / / __ \/ ___/ __/ / __ \/ __ \/ ___/
    //   / /____>  </ /_/  __/ /  / / / / /_/ / /  / __/ / /_/ / / / / /__/ /_/ / /_/ / / / (__  )
    //  /_____/_/|_|\__/\___/_/  /_/ /_/\__,_/_/  /_/    \__,_/_/ /_/\___/\__/_/\____/_/ /_/____/

    /// @inheritdoc IEPProgramManager
    /// @dev Only the contract owner can perform this operation
    function createProgram(uint256 programId, address programAdmin, address signer, ISuperToken token)
        external
        override
        onlyOwner
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

    /**
     * @notice Stop flows from this contract to the distribution pool and to the staking reserve.
     *         Return the undistributed funds to the treasury
     *  @dev Only the contract owner can perform this operation
     * @param programId program identifier to cancel
     */
    function cancelProgram(uint256 programId) external onlyOwner {
        EPProgram memory program = programs[programId];
        FluidProgramDetails memory programDetails = _fluidProgramDetails[programId];

        // Ensure program exists or has not already been terminated
        if (programDetails.fundingStartDate == 0) revert IEPProgramManager.INVALID_PARAMETER();

        // Calculate the end date
        uint256 endDate = programDetails.fundingStartDate + PROGRAM_DURATION;

        // Calculate the undistributed amounts (if the end date has not passed)
        uint256 undistributedFundingAmount;
        uint256 undistributedSubsidyAmount;

        if (endDate > block.timestamp) {
            undistributedFundingAmount = (endDate - block.timestamp) * uint96(programDetails.fundingFlowRate);
            undistributedSubsidyAmount = (endDate - block.timestamp) * uint96(programDetails.subsidyFlowRate);
        }

        // Stop the stream to the program pool
        program.token.distributeFlow(address(this), program.distributionPool, 0);

        if (programDetails.subsidyFlowRate > 0) {
            // Decrease the subsidy flow to the tax distribution pool
            _decreaseSubsidyFlow(program.token, programDetails.subsidyFlowRate);
        }

        if (undistributedFundingAmount + undistributedSubsidyAmount > 0) {
            // Transfer back the undistributed amounts to the treasury
            program.token.transfer(fluidTreasury, undistributedFundingAmount + undistributedSubsidyAmount);
        }

        // Delete the program details
        delete _fluidProgramDetails[programId];

        emit ProgramCancelled(programId, undistributedFundingAmount, undistributedSubsidyAmount);
    }

    /**
     * @notice Programatically calculate and initiate distribution to the GDA pools and staking reserve
     * @dev Only the contract owner can perform this operation
     * @param programId program identifier to start funding
     * @param totalAmount total amount to be distributed (including staking subsidy)
     */
    function startFunding(uint256 programId, uint256 totalAmount) external onlyOwner {
        EPProgram memory program = programs[programId];

        // Calculate the funding and subsidy amount
        uint256 subsidyAmount = (totalAmount * subsidyFundingRate) / _BP_DENOMINATOR;
        uint256 fundingAmount = totalAmount - subsidyAmount;

        // Calculate the funding and subsidy flow rates
        int96 subsidyFlowRate = int256(subsidyAmount / PROGRAM_DURATION).toInt96();
        int96 fundingFlowRate = int256(fundingAmount / PROGRAM_DURATION).toInt96();

        // Persist program details
        _fluidProgramDetails[programId] = FluidProgramDetails({
            fundingFlowRate: fundingFlowRate,
            subsidyFlowRate: subsidyFlowRate,
            fundingStartDate: uint64(block.timestamp)
        });

        // Fetch funds from FLUID Treasury (requires prior approval from the Treasury)
        program.token.transferFrom(fluidTreasury, address(this), totalAmount);

        // Distribute flow to Program GDA pool
        program.token.distributeFlow(address(this), program.distributionPool, fundingFlowRate);

        if (subsidyFlowRate > 0) {
            // Create or update the subsidy flow to the Staking Reward Controller
            _increaseSubsidyFlow(program.token, subsidyFlowRate);
        }

        emit ProgramFunded(
            programId,
            fundingAmount,
            subsidyAmount,
            block.timestamp + PROGRAM_DURATION - EARLY_PROGRAM_END,
            block.timestamp + PROGRAM_DURATION
        );
    }

    /**
     * @notice Stop flows from this contract to the distribution pool and to the staking reserve
     *         Send the undistributed funds to the program pool and tax distribution pool
     * @param programId program identifier to stop funding
     */
    function stopFunding(uint256 programId) external {
        EPProgram memory program = programs[programId];
        FluidProgramDetails memory programDetails = _fluidProgramDetails[programId];

        // Ensure program exists or has not already been terminated
        if (programDetails.fundingStartDate == 0) revert IEPProgramManager.INVALID_PARAMETER();

        uint256 endDate = programDetails.fundingStartDate + PROGRAM_DURATION;

        // Ensure time window is valid to stop the funding
        if (block.timestamp < endDate - EARLY_PROGRAM_END) {
            revert TOO_EARLY_TO_END_PROGRAM();
        }

        uint256 earlyEndCompensation;
        uint256 subsidyEarlyEndCompensation;

        // if the program is stopped during its early end period, calculate the flow compensations
        if (endDate > block.timestamp) {
            earlyEndCompensation = (endDate - block.timestamp) * uint96(programDetails.fundingFlowRate);
            subsidyEarlyEndCompensation = (endDate - block.timestamp) * uint96(programDetails.subsidyFlowRate);
        }

        // Stops the distribution flow to the program pool
        program.token.distributeFlow(address(this), program.distributionPool, 0);

        if (programDetails.subsidyFlowRate > 0) {
            // Delete or update the subsidy flow to the Staking Reward Controller
            _decreaseSubsidyFlow(program.token, programDetails.subsidyFlowRate);
        }

        if (earlyEndCompensation > 0) {
            // Distribute the early end compensation to the program pool
            program.token.distributeToPool(address(this), program.distributionPool, earlyEndCompensation);
        }

        if (subsidyEarlyEndCompensation > 0) {
            // Distribute the early end compensation to the stakers pool
            program.token.distributeToPool(address(this), TAX_DISTRIBUTION_POOL, subsidyEarlyEndCompensation);
        }

        // Delete the program details
        delete _fluidProgramDetails[programId];

        emit ProgramStopped(programId, earlyEndCompensation, subsidyEarlyEndCompensation);
    }

    /**
     * @notice Update the Locker Factory contract address
     * @dev Only the contract owner can perform this operation
     * @param lockerFactoryAddress Locker Factory contract address to be set
     */
    function setLockerFactory(address lockerFactoryAddress) external onlyOwner {
        // Input validation
        if (lockerFactoryAddress == address(0)) revert INVALID_PARAMETER();
        fluidLockerFactory = IFluidLockerFactory(lockerFactoryAddress);
    }

    /**
     * @notice Update the Treasury address
     * @dev Only the contract owner can perform this operation
     * @param treasuryAddress Treasury address to be set
     */
    function setTreasury(address treasuryAddress) external onlyOwner {
        // Input validation
        if (treasuryAddress == address(0)) revert INVALID_PARAMETER();
        fluidTreasury = treasuryAddress;
    }

    /**
     * @notice Update the Staking Subsidy Rate
     * @dev Only the contract owner can perform this operation
     * @param subsidyRate Subsidy rate to be set (expressed in basis points)
     */
    function setSubsidyRate(uint96 subsidyRate) external onlyOwner {
        // Input validation
        if (subsidyRate > _BP_DENOMINATOR) revert INVALID_PARAMETER();
        subsidyFundingRate = subsidyRate;
    }

    /**
     * @notice Withdraw all funds from this contract to the treasury
     * @dev Only the contract owner can perform this operation
     * @param token token contract address
     */
    function emergencyWithdraw(ISuperToken token) external onlyOwner {
        token.transfer(fluidTreasury, token.balanceOf(address(this)));
    }

    /**
     * @notice Upgrade this proxy logic
     * @dev Only the owner address can perform this operation
     * @param newImplementation new logic contract address
     * @param data calldata for potential initializer
     */
    function upgradeTo(address newImplementation, bytes calldata data) external onlyOwner {
        ERC1967Utils.upgradeToAndCall(newImplementation, data);
    }

    //      ____      __                        __   ______                 __  _
    //     /  _/___  / /____  _________  ____ _/ /  / ____/_  ______  _____/ /_(_)___  ____  _____
    //     / // __ \/ __/ _ \/ ___/ __ \/ __ `/ /  / /_  / / / / __ \/ ___/ __/ / __ \/ __ \/ ___/
    //   _/ // / / / /_/  __/ /  / / / / /_/ / /  / __/ / /_/ / / / / /__/ /_/ / /_/ / / / (__  )
    //  /___/_/ /_/\__/\___/_/  /_/ /_/\__,_/_/  /_/    \__,_/_/ /_/\___/\__/_/\____/_/ /_/____/

    /**
     * @notice Update GDA Pool units of the locker associated to the given user
     * @param program The program associated with the update
     * @param stackPoints Amount of stack points
     * @param user The user address to receive the units
     */
    function _poolUpdate(EPProgram memory program, uint256 stackPoints, address user) internal override {
        // Get the locker address belonging to the given user
        address locker = fluidLockerFactory.getLockerAddress(user);

        // Ensure the locker exists
        if (locker == address(0)) revert LOCKER_NOT_FOUND();

        // Update the locker's units in the program GDA pool
        program.distributionPool.updateMemberUnits(locker, uint128(stackPoints));
    }

    /**
     * @notice Create or update the subsidy flow from this contract to the tax distribution pool
     * @param token token contract address
     * @param subsidyFlowRateToIncrease flow rate to add to the current global subsidy flow rate
     * @return newSubsidyFlowRate the new current global subsidy flow rate
     */
    function _increaseSubsidyFlow(ISuperToken token, int96 subsidyFlowRateToIncrease)
        internal
        returns (int96 newSubsidyFlowRate)
    {
        // Fetch current flow between this contract and the tax distribution pool
        int96 currentSubsidyFlowRate = token.getFlowDistributionFlowRate(address(this), TAX_DISTRIBUTION_POOL);

        // Calculate the new subsidy flow rate
        newSubsidyFlowRate = currentSubsidyFlowRate + subsidyFlowRateToIncrease;

        // Update the distribution flow rate to the tax distribution pool
        token.distributeFlow(address(this), TAX_DISTRIBUTION_POOL, newSubsidyFlowRate);
    }

    /**
     * @notice Delete or update the subsidy flow from this contract to the tax distribution pool
     * @param token token contract address
     * @param subsidyFlowRateToDecrease flow rate to deduce from the current global subsidy flow rate
     * @return newSubsidyFlowRate the new current global subsidy flow rate
     */
    function _decreaseSubsidyFlow(ISuperToken token, int96 subsidyFlowRateToDecrease)
        internal
        returns (int96 newSubsidyFlowRate)
    {
        // Fetch current flow between this contract and the tax distribution pool
        int96 currentSubsidyFlowRate = token.getFlowDistributionFlowRate(address(this), TAX_DISTRIBUTION_POOL);

        // Calculate the new subsidy flow rate
        newSubsidyFlowRate = currentSubsidyFlowRate <= subsidyFlowRateToDecrease
            ? int96(0)
            : currentSubsidyFlowRate - subsidyFlowRateToDecrease;

        // Update the distribution flow rate to the tax distribution pool
        token.distributeFlow(address(this), TAX_DISTRIBUTION_POOL, newSubsidyFlowRate);
    }
}
