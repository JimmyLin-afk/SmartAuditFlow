// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";

import { ECDSA } from "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import { SafeCast } from "@openzeppelin-v5/contracts/utils/math/SafeCast.sol";
import { UpgradeableBeacon } from "@openzeppelin/contracts/proxy/beacon/UpgradeableBeacon.sol";

import { SuperfluidFrameworkDeployer } from
    "@superfluid-finance/ethereum-contracts/contracts/utils/SuperfluidFrameworkDeployer.sol";
import { ERC1820RegistryCompiled } from
    "@superfluid-finance/ethereum-contracts/contracts/libs/ERC1820RegistryCompiled.sol";
import { ISuperfluidPool } from "@superfluid-finance/ethereum-contracts/contracts/interfaces/superfluid/ISuperfluid.sol";
import { TestToken } from "@superfluid-finance/ethereum-contracts/contracts/utils/TestToken.sol";
import { SuperToken, ISuperToken } from "@superfluid-finance/ethereum-contracts/contracts/superfluid/SuperToken.sol";
import { SuperTokenV1Library } from "@superfluid-finance/ethereum-contracts/contracts/apps/SuperTokenV1Library.sol";

import { FluidEPProgramManager } from "../src/FluidEPProgramManager.sol";
import { FluidLocker } from "../src/FluidLocker.sol";
import { FluidLockerFactory } from "../src/FluidLockerFactory.sol";
import { Fontaine } from "../src/Fontaine.sol";
import { StakingRewardController } from "../src/StakingRewardController.sol";

import { _deployAll, DeploySettings } from "../script/Deploy.s.sol";

using SuperTokenV1Library for SuperToken;
using SuperTokenV1Library for ISuperToken;
using ECDSA for bytes32;
using SafeCast for int256;

contract SFTest is Test {
    uint256 public constant INITIAL_BALANCE = 10000 ether;
    uint256 public constant FLUID_SUPPLY = 1_000_000_000 ether;
    uint256 public constant PROGRAM_DURATION = 90 days;

    SuperfluidFrameworkDeployer.Framework internal _sf;
    SuperfluidFrameworkDeployer internal _deployer;

    bool public constant FACTORY_IS_PAUSED = false;
    bool public constant LOCKER_CAN_UNLOCK = true;

    address public constant ADMIN = address(0x420);
    address public constant ALICE = address(0x1);
    address public constant BOB = address(0x2);
    address public constant CAROL = address(0x3);
    address public constant FLUID_TREASURY = address(0x4);
    address[] internal TEST_ACCOUNTS = [ADMIN, FLUID_TREASURY, ALICE, BOB, CAROL];

    TestToken internal _fluidUnderlying;
    SuperToken internal _fluidSuperToken;
    ISuperToken internal _fluid;

    FluidEPProgramManager internal _programManager;
    FluidLocker internal _fluidLockerLogic;
    Fontaine internal _fontaineLogic;
    FluidLockerFactory internal _fluidLockerFactory;
    StakingRewardController internal _stakingRewardController;
    UpgradeableBeacon internal _lockerBeacon;
    UpgradeableBeacon internal _fontaineBeacon;

    function setUp() public virtual {
        // Superfluid Protocol Deployment Start
        vm.etch(ERC1820RegistryCompiled.at, ERC1820RegistryCompiled.bin);

        _deployer = new SuperfluidFrameworkDeployer();
        _deployer.deployTestFramework();
        _sf = _deployer.getFramework();

        (_fluidUnderlying, _fluidSuperToken) =
            _deployer.deployWrapperSuperToken("Super FLUID", "FLUIDx", 18, type(uint256).max, address(0));

        // Superfluid Protocol Deployment End

        // Mint tokens for test accounts
        for (uint256 i; i < TEST_ACCOUNTS.length; ++i) {
            vm.startPrank(TEST_ACCOUNTS[i]);
            vm.deal(TEST_ACCOUNTS[i], INITIAL_BALANCE);
            vm.stopPrank();
        }

        vm.startPrank(FLUID_TREASURY);
        _fluidUnderlying.mint(FLUID_TREASURY, FLUID_SUPPLY);
        _fluidUnderlying.approve(address(_fluidSuperToken), FLUID_SUPPLY);
        _fluidSuperToken.upgrade(FLUID_SUPPLY);
        vm.stopPrank();

        DeploySettings memory settings = DeploySettings({
            fluid: _fluidSuperToken,
            governor: ADMIN,
            owner: ADMIN,
            treasury: FLUID_TREASURY,
            factoryPauseStatus: FACTORY_IS_PAUSED,
            unlockStatus: LOCKER_CAN_UNLOCK
        });

        // FLUID Contracts Deployment Start
        vm.startPrank(ADMIN);

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

        _programManager = FluidEPProgramManager(programManagerProxyAddress);
        _stakingRewardController = StakingRewardController(stakingRewardControllerProxyAddress);
        _fluidLockerFactory = FluidLockerFactory(lockerFactoryProxyAddress);
        _fluidLockerLogic = FluidLocker(lockerLogicAddress);
        _fontaineLogic = Fontaine(fontaineLogicAddress);
        _fluid = ISuperToken(address(_fluidSuperToken));
        _lockerBeacon = UpgradeableBeacon(lockerBeaconAddress);
        _fontaineBeacon = UpgradeableBeacon(fontaineBeaconAddress);
        vm.stopPrank();

        // FLUID Contracts Deployment End
    }

    //      __  __     __                   ______                 __  _
    //     / / / /__  / /___  ___  _____   / ____/_  ______  _____/ /_(_)___  ____  _____
    //    / /_/ / _ \/ / __ \/ _ \/ ___/  / /_  / / / / __ \/ ___/ __/ / __ \/ __ \/ ___/
    //   / __  /  __/ / /_/ /  __/ /     / __/ / /_/ / / / / /__/ /_/ / /_/ / / / (__  )
    //  /_/ /_/\___/_/ .___/\___/_/     /_/    \__,_/_/ /_/\___/\__/_/\____/_/ /_/____/
    //              /_/

    function _helperCreateProgram(uint256 pId, address admin, address signer)
        internal
        virtual
        returns (ISuperfluidPool pool)
    {
        vm.prank(ADMIN);
        pool = _programManager.createProgram(pId, admin, signer, _fluidSuperToken);
    }

    function _helperCreatePrograms(uint256[] memory pIds, address admin, address signer)
        internal
        returns (ISuperfluidPool[] memory pools)
    {
        vm.startPrank(ADMIN);
        pools = new ISuperfluidPool[](pIds.length);

        for (uint256 i; i < pIds.length; ++i) {
            pools[i] = _programManager.createProgram(pIds[i], admin, signer, _fluidSuperToken);
        }

        vm.stopPrank();
    }

    function _helperGenerateSignature(
        uint256 _signerPkey,
        address _locker,
        uint256 _unitsToGrant,
        uint256 _programId,
        uint256 _nonce
    ) internal pure returns (bytes memory signature) {
        bytes32 message = keccak256(abi.encodePacked(_locker, _unitsToGrant, _programId, _nonce));

        bytes32 digest = message.toEthSignedMessageHash();

        (uint8 v, bytes32 r, bytes32 s) = vm.sign(_signerPkey, digest);
        signature = abi.encodePacked(r, s, v);
    }

    function _helperDistributeToProgramPool(uint256 programId, uint256 amount, uint256 period)
        internal
        returns (int96 actualDistributionFlowRate)
    {
        ISuperfluidPool pool = _programManager.getProgramPool(programId);

        int96 distributionFlowRate = int256(amount / period).toInt96();

        (actualDistributionFlowRate,) =
            _fluid.estimateFlowDistributionActualFlowRate(FLUID_TREASURY, pool, distributionFlowRate);

        vm.startPrank(FLUID_TREASURY);
        _fluid.distributeFlow(FLUID_TREASURY, pool, distributionFlowRate);
        vm.stopPrank();
    }

    function _helperDistributeToProgramPool(
        uint256[] memory programIds,
        uint256[] memory amounts,
        uint256[] memory periods
    ) internal returns (int96[] memory actualDistributionFlowRates) {
        actualDistributionFlowRates = new int96[](programIds.length);

        for (uint256 i; i < programIds.length; ++i) {
            ISuperfluidPool pool = _programManager.getProgramPool(programIds[i]);

            int96 distributionFlowRate = int256(amounts[i] / periods[i]).toInt96();

            (actualDistributionFlowRates[i],) =
                _fluid.estimateFlowDistributionActualFlowRate(FLUID_TREASURY, pool, distributionFlowRate);

            vm.startPrank(FLUID_TREASURY);
            _fluid.distributeFlow(FLUID_TREASURY, pool, distributionFlowRate);
            vm.stopPrank();
        }
    }

    function _helperFundLocker(address locker, uint256 amount) internal {
        vm.prank(FLUID_TREASURY);
        _fluidSuperToken.transfer(locker, amount);
    }
}
