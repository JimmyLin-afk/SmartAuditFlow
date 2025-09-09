// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import { SFTest } from "./SFTest.t.sol";

import { UpgradeableBeacon } from "@openzeppelin/contracts/proxy/beacon/UpgradeableBeacon.sol";
import { SafeCast } from "@openzeppelin-v5/contracts/utils/math/SafeCast.sol";
import { Math } from "@openzeppelin-v5/contracts/utils/math/Math.sol";

import {
    ISuperToken,
    ISuperfluidPool
} from "@superfluid-finance/ethereum-contracts/contracts/interfaces/superfluid/ISuperfluid.sol";
import { SuperTokenV1Library } from "@superfluid-finance/ethereum-contracts/contracts/apps/SuperTokenV1Library.sol";

import { FluidLocker, IFluidLocker } from "../src/FluidLocker.sol";
import { IFontaine } from "../src/interfaces/IFontaine.sol";
import { IEPProgramManager } from "../src/interfaces/IEPProgramManager.sol";
import { IStakingRewardController } from "../src/interfaces/IStakingRewardController.sol";
import { Fontaine } from "../src/Fontaine.sol";

using SuperTokenV1Library for ISuperToken;
using SafeCast for int256;

contract FluidLockerTest is SFTest {
    uint256 public constant PROGRAM_0 = 1;
    uint256 public constant PROGRAM_1 = 2;
    uint256 public constant PROGRAM_2 = 3;
    uint256 public constant signerPkey = 0x69;

    uint128 internal constant _MIN_UNLOCK_PERIOD = 7 days;

    uint128 internal constant _MAX_UNLOCK_PERIOD = 540 days;

    ISuperfluidPool[] public programPools;
    IFluidLocker public aliceLocker;
    IFluidLocker public bobLocker;

    function setUp() public virtual override {
        super.setUp();

        uint256[] memory pIds = new uint256[](3);
        pIds[0] = PROGRAM_0;
        pIds[1] = PROGRAM_1;
        pIds[2] = PROGRAM_2;

        programPools = _helperCreatePrograms(pIds, ADMIN, vm.addr(signerPkey));

        vm.prank(ALICE);
        aliceLocker = IFluidLocker(_fluidLockerFactory.createLockerContract());

        vm.prank(BOB);
        bobLocker = IFluidLocker(_fluidLockerFactory.createLockerContract());
    }

    function testInitialize(address owner) external virtual {
        vm.expectRevert();
        FluidLocker(address(aliceLocker)).initialize(owner);
    }

    function testClaim(uint256 units) external virtual {
        units = bound(units, 1, 1_000_000);

        uint256 nonce = _programManager.getNextValidNonce(PROGRAM_0, ALICE);
        bytes memory signature = _helperGenerateSignature(signerPkey, ALICE, units, PROGRAM_0, nonce);

        vm.prank(ALICE);
        aliceLocker.claim(PROGRAM_0, units, nonce, signature);

        assertEq(programPools[0].getUnits(address(aliceLocker)), units, "units not updated");
        assertEq(aliceLocker.getUnitsPerProgram(PROGRAM_0), units, "getUnitsPerProgram invalid");

        int96 distributionFlowrate = _helperDistributeToProgramPool(PROGRAM_0, 1_000_000e18, _MAX_UNLOCK_PERIOD);

        assertEq(aliceLocker.getFlowRatePerProgram(PROGRAM_0), distributionFlowrate, "getFlowRatePerProgram invalid");
    }

    function testClaimBatch(uint256 units) external virtual {
        units = bound(units, 1, 1_000_000);

        uint256[] memory programIds = new uint256[](3);
        uint256[] memory newUnits = new uint256[](3);
        uint256[] memory nonces = new uint256[](3);
        bytes[] memory signatures = new bytes[](3);

        uint256[] memory distributionAmounts = new uint256[](3);
        uint256[] memory distributionPeriods = new uint256[](3);

        for (uint8 i = 0; i < 3; ++i) {
            programIds[i] = i + 1;
            newUnits[i] = units;
            nonces[i] = _programManager.getNextValidNonce(programIds[i], ALICE);
            signatures[i] = _helperGenerateSignature(signerPkey, ALICE, units, programIds[i], nonces[i]);
            distributionAmounts[i] = 1_000_000e18;
            distributionPeriods[i] = _MAX_UNLOCK_PERIOD;
        }

        vm.prank(ALICE);
        aliceLocker.claim(programIds, newUnits, nonces, signatures);

        int96[] memory distributionFlowrates =
            _helperDistributeToProgramPool(programIds, distributionAmounts, distributionPeriods);

        uint128[] memory unitsPerProgram = aliceLocker.getUnitsPerProgram(programIds);
        int96[] memory flowratePerProgram = aliceLocker.getFlowRatePerProgram(programIds);

        for (uint8 i = 0; i < 3; ++i) {
            assertEq(newUnits[i], programPools[i].getUnits(address(aliceLocker)), "incorrect units amounts");
            assertEq(newUnits[i], unitsPerProgram[i], "getUnitsPerProgram invalid");
            assertEq(distributionFlowrates[i], flowratePerProgram[i], "getFlowRatePerProgram invalid");
        }
    }

    function testLock(uint256 amount) external virtual {
        amount = bound(amount, 1e18, 1e24);
        assertEq(_fluidSuperToken.balanceOf(address(aliceLocker)), 0, "incorrect balance before operation");

        vm.startPrank(FLUID_TREASURY);
        _fluidSuperToken.approve(address(aliceLocker), amount);
        aliceLocker.lock(amount);
        vm.stopPrank();

        assertEq(_fluidSuperToken.balanceOf(address(aliceLocker)), amount, "incorrect balance after operation");
    }

    function testInstantUnlock() external virtual {
        _helperFundLocker(address(aliceLocker), 10_000e18);
        _helperBobStaking();

        assertEq(_fluidSuperToken.balanceOf(address(ALICE)), 0, "incorrect Alice bal before op");
        assertEq(_fluidSuperToken.balanceOf(address(aliceLocker)), 10_000e18, "incorrect Locker bal before op");
        assertEq(bobLocker.getAvailableBalance(), 0, "incorrect Bob bal before op");

        vm.prank(BOB);
        vm.expectRevert(IFluidLocker.NOT_LOCKER_OWNER.selector);
        aliceLocker.unlock(0, ALICE);

        vm.prank(ALICE);
        vm.expectRevert(IFluidLocker.FORBIDDEN.selector);
        aliceLocker.unlock(0, address(0));

        vm.prank(ALICE);
        aliceLocker.unlock(0, ALICE);

        assertEq(_fluidSuperToken.balanceOf(address(ALICE)), 2_000e18, "incorrect Alice bal after op");
        assertEq(_fluidSuperToken.balanceOf(address(aliceLocker)), 0, "incorrect bal after op");
        assertEq(bobLocker.getAvailableBalance(), 8_000e18, "incorrect Bob bal after op");

        vm.prank(ALICE);
        vm.expectRevert(IFluidLocker.NO_FLUID_TO_UNLOCK.selector);
        aliceLocker.unlock(0, ALICE);
    }

    function testVestUnlock(uint128 unlockPeriod) external virtual {
        unlockPeriod = uint128(bound(unlockPeriod, _MIN_UNLOCK_PERIOD, _MAX_UNLOCK_PERIOD));
        uint256 funding = 10_000e18;
        _helperFundLocker(address(aliceLocker), funding);
        _helperBobStaking();

        assertEq(_fluidSuperToken.balanceOf(address(aliceLocker)), funding, "incorrect Locker bal before op");
        assertEq(FluidLocker(address(aliceLocker)).fontaineCount(), 0, "incorrect fontaine count");
        assertEq(
            address(FluidLocker(address(aliceLocker)).fontaines(0)),
            address(IFontaine(address(0))),
            "incorrect fontaine addrfoess"
        );

        (int96 taxFlowRate, int96 unlockFlowRate) = _helperCalculateUnlockFlowRates(funding, unlockPeriod);

        vm.prank(ALICE);
        aliceLocker.unlock(unlockPeriod, ALICE);

        IFontaine newFontaine = FluidLocker(address(aliceLocker)).fontaines(0);

        assertEq(_fluidSuperToken.balanceOf(address(aliceLocker)), 0, "incorrect bal after op");
        assertEq(
            ISuperToken(_fluidSuperToken).getFlowRate(address(newFontaine), ALICE),
            unlockFlowRate,
            "incorrect unlock flowrate"
        );
        assertApproxEqAbs(
            FluidLocker(address(aliceLocker)).TAX_DISTRIBUTION_POOL().getMemberFlowRate(address(bobLocker)),
            taxFlowRate,
            funding / 1e16,
            "incorrect tax flowrate"
        );
    }

    function testInvalidUnlockPeriod(uint128 unlockPeriod) external virtual {
        uint256 funding = 10_000e18;
        _helperFundLocker(address(aliceLocker), funding);

        unlockPeriod = uint128(bound(unlockPeriod, 0 + 1, _MIN_UNLOCK_PERIOD - 1));
        vm.prank(ALICE);
        vm.expectRevert(IFluidLocker.INVALID_UNLOCK_PERIOD.selector);
        aliceLocker.unlock(unlockPeriod, ALICE);

        unlockPeriod = uint128(bound(unlockPeriod, _MAX_UNLOCK_PERIOD + 1, 100_000 days));
        vm.prank(ALICE);
        vm.expectRevert(IFluidLocker.INVALID_UNLOCK_PERIOD.selector);
        aliceLocker.unlock(unlockPeriod, ALICE);
    }

    function testStake() external virtual {
        uint256 funding = 10_000e18;
        _helperFundLocker(address(aliceLocker), funding);
        assertEq(_fluidSuperToken.balanceOf(address(aliceLocker)), funding, "incorrect Locker bal before op");
        assertEq(aliceLocker.getAvailableBalance(), funding, "incorrect available bal before op");
        assertEq(aliceLocker.getStakedBalance(), 0, "incorrect staked bal before op");

        vm.prank(ALICE);
        aliceLocker.stake();

        assertEq(aliceLocker.getAvailableBalance(), 0, "incorrect available bal after op");
        assertEq(aliceLocker.getStakedBalance(), funding, "incorrect staked bal after op");
        assertEq(
            _stakingRewardController.taxDistributionPool().getUnits(address(aliceLocker)),
            funding / 1e16,
            "incorrect units"
        );

        vm.prank(ALICE);
        vm.expectRevert(IFluidLocker.NO_FLUID_TO_STAKE.selector);
        aliceLocker.stake();
    }

    function testUnstake() external virtual {
        uint256 funding = 10_000e18;
        _helperFundLocker(address(aliceLocker), funding);
        vm.startPrank(ALICE);
        aliceLocker.stake();

        assertEq(aliceLocker.getAvailableBalance(), 0, "incorrect available bal before op");
        assertEq(aliceLocker.getStakedBalance(), funding, "incorrect staked bal before op");
        assertEq(
            _stakingRewardController.taxDistributionPool().getUnits(address(aliceLocker)),
            funding / 1e16,
            "incorrect units before op"
        );

        vm.expectRevert(IFluidLocker.STAKING_COOLDOWN_NOT_ELAPSED.selector);
        aliceLocker.unstake();

        vm.warp(uint256(FluidLocker(address(aliceLocker)).stakingUnlocksAt()) + 1);
        aliceLocker.unstake();

        assertEq(aliceLocker.getAvailableBalance(), funding, "incorrect available bal after op");
        assertEq(aliceLocker.getStakedBalance(), 0, "incorrect staked bal after op");
        assertEq(
            _stakingRewardController.taxDistributionPool().getUnits(address(aliceLocker)), 0, "incorrect units after op"
        );

        vm.expectRevert(IFluidLocker.NO_FLUID_TO_UNSTAKE.selector);
        aliceLocker.unstake();

        vm.stopPrank();
    }

    function testGetFontaineBeaconImplementation() external view virtual {
        assertEq(_fluidLockerLogic.getFontaineBeaconImplementation(), address(_fontaineLogic));
    }

    function _helperBobStaking() internal {
        _helperFundLocker(address(bobLocker), 10_000e18);
        vm.prank(BOB);
        bobLocker.stake();
    }

    function _helperCalculateUnlockFlowRates(uint256 amountToUnlock, uint128 unlockPeriod)
        internal
        pure
        returns (int96 taxFlowRate, int96 unlockFlowRate)
    {
        uint256 unlockingPercentageBP =
            (100 * (((80 * 1e18) / Math.sqrt(540 * 1e18)) * (Math.sqrt(unlockPeriod * 1e18) / 1e18) + 20 * 1e18)) / 1e18;

        uint256 amountToUser = (amountToUnlock * unlockingPercentageBP) / 10_000;
        uint256 penaltyAmount = amountToUnlock - amountToUser;

        taxFlowRate = int256(penaltyAmount / unlockPeriod).toInt96();
        unlockFlowRate = int256(amountToUser / unlockPeriod).toInt96();
    }
}

contract FluidLockerTTETest is SFTest {
    address internal _nonUnlockableLockerLogic;
    address internal _unlockableLockerLogic;

    uint256 public constant PROGRAM_0 = 1;
    uint256 public constant PROGRAM_1 = 2;
    uint256 public constant PROGRAM_2 = 3;
    uint256 public constant signerPkey = 0x69;

    uint128 internal constant _MIN_UNLOCK_PERIOD = 7 days;

    uint128 internal constant _MAX_UNLOCK_PERIOD = 540 days;

    ISuperfluidPool[] public programPools;
    IFluidLocker public aliceLocker;
    IFluidLocker public bobLocker;

    function setUp() public override {
        super.setUp();

        // Deploy the non-unlockable Fluid Locker Implementation contract
        _nonUnlockableLockerLogic = address(
            new FluidLocker(
                _fluid,
                _stakingRewardController.taxDistributionPool(),
                IEPProgramManager(address(_programManager)),
                IStakingRewardController(address(_stakingRewardController)),
                address(_fontaineLogic),
                !LOCKER_CAN_UNLOCK
            )
        );

        _unlockableLockerLogic = address(_fluidLockerLogic);

        UpgradeableBeacon beacon = _fluidLockerFactory.LOCKER_BEACON();

        vm.prank(ADMIN);
        beacon.upgradeTo(_nonUnlockableLockerLogic);

        uint256[] memory pIds = new uint256[](3);
        pIds[0] = PROGRAM_0;
        pIds[1] = PROGRAM_1;
        pIds[2] = PROGRAM_2;

        programPools = _helperCreatePrograms(pIds, ADMIN, vm.addr(signerPkey));

        vm.prank(ALICE);
        aliceLocker = IFluidLocker(_fluidLockerFactory.createLockerContract());

        vm.prank(BOB);
        bobLocker = IFluidLocker(_fluidLockerFactory.createLockerContract());
    }

    function testClaim(uint256 units) external {
        units = bound(units, 1, 1_000_000);

        uint256 nonce = _programManager.getNextValidNonce(PROGRAM_0, ALICE);
        bytes memory signature = _helperGenerateSignature(signerPkey, ALICE, units, PROGRAM_0, nonce);

        vm.prank(ALICE);
        aliceLocker.claim(PROGRAM_0, units, nonce, signature);

        assertEq(programPools[0].getUnits(address(aliceLocker)), units, "units not updated");
        assertEq(aliceLocker.getUnitsPerProgram(PROGRAM_0), units, "getUnitsPerProgram invalid");

        int96 distributionFlowrate = _helperDistributeToProgramPool(PROGRAM_0, 1_000_000e18, _MAX_UNLOCK_PERIOD);

        assertEq(aliceLocker.getFlowRatePerProgram(PROGRAM_0), distributionFlowrate, "getFlowRatePerProgram invalid");
    }

    function testClaimBatch(uint256 units) external {
        units = bound(units, 1, 1_000_000);

        uint256[] memory programIds = new uint256[](3);
        uint256[] memory newUnits = new uint256[](3);
        uint256[] memory nonces = new uint256[](3);
        bytes[] memory signatures = new bytes[](3);

        uint256[] memory distributionAmounts = new uint256[](3);
        uint256[] memory distributionPeriods = new uint256[](3);

        for (uint8 i = 0; i < 3; ++i) {
            programIds[i] = i + 1;
            newUnits[i] = units;
            nonces[i] = _programManager.getNextValidNonce(programIds[i], ALICE);
            signatures[i] = _helperGenerateSignature(signerPkey, ALICE, units, programIds[i], nonces[i]);
            distributionAmounts[i] = 1_000_000e18;
            distributionPeriods[i] = _MAX_UNLOCK_PERIOD;
        }

        vm.prank(ALICE);
        aliceLocker.claim(programIds, newUnits, nonces, signatures);

        int96[] memory distributionFlowrates =
            _helperDistributeToProgramPool(programIds, distributionAmounts, distributionPeriods);

        uint128[] memory unitsPerProgram = aliceLocker.getUnitsPerProgram(programIds);
        int96[] memory flowratePerProgram = aliceLocker.getFlowRatePerProgram(programIds);

        for (uint8 i = 0; i < 3; ++i) {
            assertEq(newUnits[i], programPools[i].getUnits(address(aliceLocker)), "incorrect units amounts");
            assertEq(newUnits[i], unitsPerProgram[i], "getUnitsPerProgram invalid");
            assertEq(distributionFlowrates[i], flowratePerProgram[i], "getFlowRatePerProgram invalid");
        }
    }

    function testLock(uint256 amount) external {
        amount = bound(amount, 1e18, 1e24);
        assertEq(_fluidSuperToken.balanceOf(address(aliceLocker)), 0, "incorrect balance before operation");

        vm.startPrank(FLUID_TREASURY);
        _fluidSuperToken.approve(address(aliceLocker), amount);
        aliceLocker.lock(amount);
        vm.stopPrank();

        assertEq(_fluidSuperToken.balanceOf(address(aliceLocker)), amount, "incorrect balance after operation");
    }

    function testInstantUnlock() external {
        _helperFundLocker(address(aliceLocker), 10_000e18);

        assertEq(_fluidSuperToken.balanceOf(address(ALICE)), 0, "incorrect Alice bal before op");
        assertEq(_fluidSuperToken.balanceOf(address(aliceLocker)), 10_000e18, "incorrect Locker bal before op");
        assertEq(bobLocker.getAvailableBalance(), 0, "incorrect Bob bal before op");

        vm.prank(ALICE);
        vm.expectRevert(IFluidLocker.TTE_NOT_ACTIVATED.selector);
        aliceLocker.unlock(0, ALICE);

        _helperUpgradeLocker();
        _helperBobStaking();

        vm.prank(ALICE);
        aliceLocker.unlock(0, ALICE);

        assertEq(_fluidSuperToken.balanceOf(address(ALICE)), 2_000e18, "incorrect Alice bal after op");
        assertEq(_fluidSuperToken.balanceOf(address(aliceLocker)), 0, "incorrect bal after op");
        assertEq(bobLocker.getAvailableBalance(), 8_000e18, "incorrect Bob bal after op");
    }

    function testVestUnlock(uint128 unlockPeriod) external {
        unlockPeriod = uint128(bound(unlockPeriod, _MIN_UNLOCK_PERIOD, _MAX_UNLOCK_PERIOD));
        uint256 funding = 10_000e18;
        _helperFundLocker(address(aliceLocker), funding);

        assertEq(_fluidSuperToken.balanceOf(address(aliceLocker)), funding, "incorrect Locker bal before op");
        assertEq(FluidLocker(address(aliceLocker)).fontaineCount(), 0, "incorrect fontaine count");
        assertEq(
            address(FluidLocker(address(aliceLocker)).fontaines(0)),
            address(IFontaine(address(0))),
            "incorrect fontaine addrfoess"
        );

        (int96 taxFlowRate, int96 unlockFlowRate) = _helperCalculateUnlockFlowRates(funding, unlockPeriod);

        vm.prank(ALICE);
        vm.expectRevert(IFluidLocker.TTE_NOT_ACTIVATED.selector);
        aliceLocker.unlock(unlockPeriod, ALICE);

        _helperUpgradeLocker();
        _helperBobStaking();

        vm.prank(ALICE);
        aliceLocker.unlock(unlockPeriod, ALICE);

        IFontaine newFontaine = FluidLocker(address(aliceLocker)).fontaines(0);

        assertEq(_fluidSuperToken.balanceOf(address(aliceLocker)), 0, "incorrect bal after op");
        assertEq(
            ISuperToken(_fluidSuperToken).getFlowRate(address(newFontaine), ALICE),
            unlockFlowRate,
            "incorrect unlock flowrate"
        );
        assertApproxEqAbs(
            FluidLocker(address(aliceLocker)).TAX_DISTRIBUTION_POOL().getMemberFlowRate(address(bobLocker)),
            taxFlowRate,
            funding / 1e16,
            "incorrect tax flowrate"
        );
    }

    function testStake() external {
        uint256 funding = 10_000e18;
        _helperFundLocker(address(aliceLocker), funding);
        assertEq(_fluidSuperToken.balanceOf(address(aliceLocker)), funding, "incorrect Locker bal before op");
        assertEq(aliceLocker.getAvailableBalance(), funding, "incorrect available bal before op");
        assertEq(aliceLocker.getStakedBalance(), 0, "incorrect staked bal before op");

        vm.prank(ALICE);
        vm.expectRevert(IFluidLocker.TTE_NOT_ACTIVATED.selector);
        aliceLocker.stake();

        _helperUpgradeLocker();

        vm.prank(ALICE);
        aliceLocker.stake();

        assertEq(aliceLocker.getAvailableBalance(), 0, "incorrect available bal after op");
        assertEq(aliceLocker.getStakedBalance(), funding, "incorrect staked bal after op");
        assertEq(
            _stakingRewardController.taxDistributionPool().getUnits(address(aliceLocker)),
            funding / 1e16,
            "incorrect units"
        );

        vm.prank(ALICE);
        vm.expectRevert(IFluidLocker.NO_FLUID_TO_STAKE.selector);
        aliceLocker.stake();
    }

    function _helperUpgradeLocker() internal {
        UpgradeableBeacon beacon = _fluidLockerFactory.LOCKER_BEACON();

        vm.prank(ADMIN);
        beacon.upgradeTo(_unlockableLockerLogic);
    }

    function _helperBobStaking() internal {
        _helperFundLocker(address(bobLocker), 10_000e18);
        vm.prank(BOB);
        bobLocker.stake();
    }

    function _helperCalculateUnlockFlowRates(uint256 amountToUnlock, uint128 unlockPeriod)
        internal
        pure
        returns (int96 taxFlowRate, int96 unlockFlowRate)
    {
        uint256 unlockingPercentageBP =
            (100 * (((80 * 1e18) / Math.sqrt(540 * 1e18)) * (Math.sqrt(unlockPeriod * 1e18) / 1e18) + 20 * 1e18)) / 1e18;

        uint256 amountToUser = (amountToUnlock * unlockingPercentageBP) / 10_000;
        uint256 penaltyAmount = amountToUnlock - amountToUser;

        taxFlowRate = int256(penaltyAmount / unlockPeriod).toInt96();
        unlockFlowRate = int256(amountToUser / unlockPeriod).toInt96();
    }
}

contract FluidLockerLayoutTest is FluidLocker {
    constructor()
        FluidLocker(
            ISuperToken(address(0)),
            ISuperfluidPool(address(0)),
            IEPProgramManager(address(0)),
            IStakingRewardController(address(0)),
            address(0),
            true
        )
    { }

    function testStorageLayout() external pure {
        uint256 slot;
        uint256 offset;

        // FluidLocker storage

        assembly {
            slot := lockerOwner.slot
            offset := lockerOwner.offset
        }
        require(slot == 0 && offset == 0, "lockerOwner changed location");

        assembly {
            slot := stakingUnlocksAt.slot
            offset := stakingUnlocksAt.offset
        }
        require(slot == 0 && offset == 20, "stakingUnlocksAt changed location");

        assembly {
            slot := fontaineCount.slot
            offset := fontaineCount.offset
        }
        require(slot == 0 && offset == 30, "fontaineCount changed location");

        // private state : _stakedBalance
        // slot = 1 - offset = 0

        assembly {
            slot := fontaines.slot
            offset := fontaines.offset
        }
        require(slot == 2 && offset == 0, "fontaines changed location");
    }
}
