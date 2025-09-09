// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

/* Openzeppelin Contracts & Interfaces */
import { Initializable } from "@openzeppelin-v5/contracts/proxy/utils/Initializable.sol";

/* Superfluid Protocol Contracts & Interfaces */
import {
    ISuperfluid,
    ISuperfluidPool,
    ISuperToken,
    ISuperApp
} from "@superfluid-finance/ethereum-contracts/contracts/interfaces/superfluid/ISuperfluid.sol";
import { SuperTokenV1Library } from "@superfluid-finance/ethereum-contracts/contracts/apps/SuperTokenV1Library.sol";

/* FLUID Contracts & Interfaces */
import { IFontaine } from "./interfaces/IFontaine.sol";

using SuperTokenV1Library for ISuperToken;

/**
 * @title Fontaine Contract
 * @author Superfluid
 * @notice Contract responsible for flowing the token being unlocked from the locker to the locker owner
 *
 */
contract Fontaine is Initializable, IFontaine {
    //     _____ __        __
    //    / ___// /_____ _/ /____  _____
    //    \__ \/ __/ __ `/ __/ _ \/ ___/
    //   ___/ / /_/ /_/ / /_/  __(__  )
    //  /____/\__/\__,_/\__/\___/____/

    /// @notice $FLUID SuperToken interface
    ISuperToken public immutable FLUID;

    /// @notice Superfluid pool interface
    ISuperfluidPool public immutable TAX_DISTRIBUTION_POOL;

    //     ______                 __                  __
    //    / ____/___  ____  _____/ /________  _______/ /_____  _____
    //   / /   / __ \/ __ \/ ___/ __/ ___/ / / / ___/ __/ __ \/ ___/
    //  / /___/ /_/ / / / (__  ) /_/ /  / /_/ / /__/ /_/ /_/ / /
    //  \____/\____/_/ /_/____/\__/_/   \__,_/\___/\__/\____/_/

    /**
     * @notice Fontaine contract constructor
     * @param fluid FLUID SuperToken interface
     * @param taxDistributionPool Tax Distribution Pool GDA contract address
     */
    constructor(ISuperToken fluid, ISuperfluidPool taxDistributionPool) {
        // Disable initializers to prevent implementation contract initalization
        _disableInitializers();

        // Sets immutable states
        FLUID = fluid;
        TAX_DISTRIBUTION_POOL = taxDistributionPool;
    }

    /// @inheritdoc IFontaine
    function initialize(address unlockRecipient, int96 unlockFlowRate, int96 taxFlowRate) external initializer {
        // Ensure recipient is not a SuperApp
        if (ISuperfluid(FLUID.getHost()).isApp(ISuperApp(unlockRecipient))) revert CANNOT_UNLOCK_TO_SUPERAPP();

        // Distribute Tax flow to Staker GDA Pool
        FLUID.distributeFlow(address(this), TAX_DISTRIBUTION_POOL, taxFlowRate);

        // Create the unlocking flow from the Fontaine to the locker owner
        FLUID.createFlow(unlockRecipient, unlockFlowRate);
    }
}
