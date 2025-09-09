// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import { Script, console2 } from "forge-std/Script.sol";

import { ERC1967Proxy } from "@openzeppelin-v5/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import { UpgradeableBeacon } from "@openzeppelin/contracts/proxy/beacon/UpgradeableBeacon.sol";

import {
    ISuperfluid,
    ISuperfluidPool,
    ISuperToken
} from "@superfluid-finance/ethereum-contracts/contracts/interfaces/superfluid/ISuperfluid.sol";

import { IEPProgramManager } from "../src/interfaces/IEPProgramManager.sol";
import { FluidEPProgramManager } from "../src/FluidEPProgramManager.sol";
import { FluidLocker } from "../src/FluidLocker.sol";
import { FluidLockerFactory } from "../src/FluidLockerFactory.sol";
import { Fontaine } from "../src/Fontaine.sol";
import { StakingRewardController, IStakingRewardController } from "../src/StakingRewardController.sol";

struct DeploySettings {
    ISuperToken fluid;
    address governor;
    address owner;
    address treasury;
    bool factoryPauseStatus;
    bool unlockStatus;
}

function _deployFontaineBeacon(ISuperToken fluid, ISuperfluidPool taxDistributionPool, address governor)
    returns (address fontaineLogicAddress, address fontaineBeaconAddress)
{
    // Deploy the Fontaine Implementation and associated Beacon contract
    fontaineLogicAddress = address(new Fontaine(fluid, taxDistributionPool));
    UpgradeableBeacon fontaineBeacon = new UpgradeableBeacon(fontaineLogicAddress);
    fontaineBeaconAddress = address(fontaineBeacon);

    // Transfer Fontaine Beacon ownership to the governor
    fontaineBeacon.transferOwnership(governor);
}

function _deployLockerBeacon(
    DeploySettings memory settings,
    ISuperfluidPool taxDistributionPool,
    address programManagerAddress,
    address stakingRewardControllerAddress,
    address fontaineBeaconAddress
) returns (address lockerLogicAddress, address lockerBeaconAddress) {
    // Deploy the Fluid Locker Implementation and associated Beacon contract
    lockerLogicAddress = address(
        new FluidLocker(
            settings.fluid,
            taxDistributionPool,
            IEPProgramManager(programManagerAddress),
            IStakingRewardController(stakingRewardControllerAddress),
            fontaineBeaconAddress,
            settings.unlockStatus
        )
    );
    UpgradeableBeacon lockerBeacon = new UpgradeableBeacon(lockerLogicAddress);
    lockerBeaconAddress = address(lockerBeacon);

    // Transfer Locker Beacon ownership to the governor
    lockerBeacon.transferOwnership(settings.governor);
}

function _deployStakingRewardController(ISuperToken fluid, address owner)
    returns (address stakingRewardControllerLogicAddress, address stakingRewardControllerProxyAddress)
{
    // Deploy the Staking Reward Controller contract
    StakingRewardController stakingRewardControllerLogic = new StakingRewardController(fluid);
    stakingRewardControllerLogicAddress = address(stakingRewardControllerLogic);

    ERC1967Proxy stakingRewardControllerProxy = new ERC1967Proxy(
        stakingRewardControllerLogicAddress, abi.encodeWithSelector(StakingRewardController.initialize.selector, owner)
    );
    stakingRewardControllerProxyAddress = address(stakingRewardControllerProxy);
}

function _deployFluidEPProgramManager(address owner, address treasury, ISuperfluidPool taxDistributionPool)
    returns (address programManagerLogicAddress, address programManagerProxyAddress)
{
    // Deploy the Staking Reward Controller contract
    FluidEPProgramManager programManagerLogic = new FluidEPProgramManager(taxDistributionPool);
    programManagerLogicAddress = address(programManagerLogic);

    ERC1967Proxy programManagerProxy = new ERC1967Proxy(
        programManagerLogicAddress, abi.encodeWithSelector(FluidEPProgramManager.initialize.selector, owner, treasury)
    );
    programManagerProxyAddress = address(programManagerProxy);
}

function _deployLockerFactory(
    bool factoryPauseStatus,
    address governor,
    address lockerBeaconAddress,
    address stakingRewardControllerProxyAddress
) returns (address lockerFactoryLogicAddress, address lockerFactoryProxyAddress) {
    // Deploy the Fluid Locker Factory contract
    FluidLockerFactory lockerFactoryLogic = new FluidLockerFactory(
        lockerBeaconAddress, IStakingRewardController(stakingRewardControllerProxyAddress), factoryPauseStatus
    );

    lockerFactoryLogicAddress = address(lockerFactoryLogic);

    ERC1967Proxy lockerFactoryProxy = new ERC1967Proxy(
        lockerFactoryLogicAddress, abi.encodeWithSelector(FluidLockerFactory.initialize.selector, governor)
    );
    lockerFactoryProxyAddress = address(lockerFactoryProxy);
}

function _deployAll(DeploySettings memory settings)
    returns (
        address programManagerLogicAddress,
        address programManagerProxyAddress,
        address stakingRewardControllerLogicAddress,
        address stakingRewardControllerProxyAddress,
        address lockerFactoryLogicAddress,
        address lockerFactoryProxyAddress,
        address lockerLogicAddress,
        address lockerBeaconAddress,
        address fontaineLogicAddress,
        address fontaineBeaconAddress
    )
{
    (stakingRewardControllerLogicAddress, stakingRewardControllerProxyAddress) =
        _deployStakingRewardController(settings.fluid, settings.owner);

    ISuperfluidPool taxDistributionPool =
        StakingRewardController(stakingRewardControllerProxyAddress).taxDistributionPool();

    // Deploy Ecosystem Partner Program Manager
    (programManagerLogicAddress, programManagerProxyAddress) =
        _deployFluidEPProgramManager(settings.owner, settings.treasury, taxDistributionPool);

    // Deploy the Fontaine Implementation and associated Beacon contract
    (fontaineLogicAddress, fontaineBeaconAddress) =
        _deployFontaineBeacon(settings.fluid, taxDistributionPool, settings.governor);

    // Deploy the Fluid Locker Implementation and associated Beacon contract
    (lockerLogicAddress, lockerBeaconAddress) = _deployLockerBeacon(
        settings,
        taxDistributionPool,
        programManagerProxyAddress,
        stakingRewardControllerProxyAddress,
        fontaineBeaconAddress
    );

    (lockerFactoryLogicAddress, lockerFactoryProxyAddress) = _deployLockerFactory(
        settings.factoryPauseStatus, settings.governor, lockerBeaconAddress, stakingRewardControllerProxyAddress
    );

    // Sets the FluidLockerFactory address in the StakingRewardController
    StakingRewardController(stakingRewardControllerProxyAddress).setLockerFactory(lockerFactoryProxyAddress);

    // Sets the FluidLockerFactory address in the ProgramManager
    FluidEPProgramManager(programManagerProxyAddress).setLockerFactory(lockerFactoryProxyAddress);
}

// forge script script/Deploy.s.sol:DeployScript --ffi --rpc-url $BASE_SEPOLIA_RPC_URL --broadcast --verify -vvvv
contract DeployScript is Script {
    error GOVERNOR_IS_ZERO_ADDRESS();

    function setUp() public { }

    function run() public {
        _showGitRevision();

        // Deployer settings
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        address deployer = vm.addr(deployerPrivateKey);

        // Deployment parameters
        address governor = vm.envAddress("GOVERNOR_ADDRESS");
        address treasury = vm.envAddress("TREASURY_ADDRESS");
        ISuperToken fluid = ISuperToken(vm.envAddress("FLUID_ADDRESS"));
        bool factoryPauseStatus = vm.envBool("PAUSE_FACTORY_LOCKER_CREATION");
        bool unlockStatus = vm.envBool("FLUID_UNLOCK_STATUS");

        // Purposedly not enforcing this at contract level in case governance decides to forfeit ownership of the contracts
        if (governor == address(0)) {
            revert GOVERNOR_IS_ZERO_ADDRESS();
        }

        DeploySettings memory settings = DeploySettings({
            fluid: fluid,
            governor: governor,
            owner: deployer,
            treasury: treasury,
            factoryPauseStatus: factoryPauseStatus,
            unlockStatus: unlockStatus
        });

        _logDeploymentSettings(deployer, address(fluid), governor, treasury, factoryPauseStatus, unlockStatus);

        vm.startBroadcast(deployerPrivateKey);
        (
            address programManagerLogicAddress,
            address programManagerProxyAddress,
            address stakingRewardControllerLogicAddress,
            address stakingRewardControllerProxyAddress,
            address lockerFactoryLogicAddress,
            address lockerFactoryProxyAddress,
            address lockerLogicAddress,
            address lockerBeaconAddress,
            address fontaineLogicAddress,
            address fontaineBeaconAddress
        ) = _deployAll(settings);

        _logDeploymentSummary(
            programManagerLogicAddress,
            programManagerProxyAddress,
            stakingRewardControllerLogicAddress,
            stakingRewardControllerProxyAddress,
            lockerFactoryLogicAddress,
            lockerFactoryProxyAddress,
            lockerLogicAddress,
            lockerBeaconAddress,
            fontaineLogicAddress,
            fontaineBeaconAddress
        );
    }

    function _logDeploymentSettings(
        address deployer,
        address fluid,
        address governor,
        address treasury,
        bool factoryPauseStatus,
        bool unlockStatus
    ) internal pure {
        console2.log("");
        console2.log("*----------------------------------* DEPLOYMENT SETTINGS *---------------------------------*");
        console2.log("|                                                                                    ");
        console2.log("| Deployer Address          : %s", deployer);
        console2.log("| FLUID Token Address       : %s", fluid);
        console2.log("| Governor Address          : %s", governor);
        console2.log("| Treasury Address          : %s", treasury);
        console2.log("| Factory Pause Status      : %s", factoryPauseStatus);
        console2.log("| Locker Unlock Status      : %s", unlockStatus);
        console2.log("*------------------------------------------------------------------------------------------*");
    }

    function _logDeploymentSummary(
        address programManagerLogicAddress,
        address programManagerProxyAddress,
        address stakingRewardControllerLogicAddress,
        address stakingRewardControllerProxyAddress,
        address lockerFactoryLogicAddress,
        address lockerFactoryProxyAddress,
        address lockerLogicAddress,
        address lockerBeaconAddress,
        address fontaineLogicAddress,
        address fontaineBeaconAddress
    ) internal pure {
        console2.log("");
        console2.log("*----------------------------------* DEPLOYMENT SUMMARY *----------------------------------*");
        console2.log("|                                                                                          |");
        console2.log("| FluidEPProgramManager (Logic)   : deployed at %s |", programManagerLogicAddress);
        console2.log("| FluidEPProgramManager (Proxy)   : deployed at %s |", programManagerProxyAddress);
        console2.log("| StakingRewardController (Logic) : deployed at %s |", stakingRewardControllerLogicAddress);
        console2.log("| StakingRewardController (Proxy) : deployed at %s |", stakingRewardControllerProxyAddress);
        console2.log("| FluidLocker (Logic)             : deployed at %s |", lockerLogicAddress);
        console2.log("| FluidLocker (Beacon)            : deployed at %s |", lockerBeaconAddress);
        console2.log("| Fontaine (Logic)                : deployed at %s |", fontaineLogicAddress);
        console2.log("| Fontaine (Beacon)               : deployed at %s |", fontaineBeaconAddress);
        console2.log("| FluidLockerFactory (Logic)      : deployed at %s |", lockerFactoryLogicAddress);
        console2.log("| FluidLockerFactory (Proxy)      : deployed at %s |", lockerFactoryProxyAddress);
        console2.log("*------------------------------------------------------------------------------------------*");
    }

    function _showGitRevision() internal {
        string[] memory inputs = new string[](2);
        inputs[0] = "../tasks/show-git-rev.sh";
        inputs[1] = "forge_ffi_mode";
        try vm.ffi(inputs) returns (bytes memory res) {
            console2.log("GIT REVISION :");
            console2.log(string(res));
        } catch {
            console2.log("!! _showGitRevision: FFI not enabled");
        }
    }
}
