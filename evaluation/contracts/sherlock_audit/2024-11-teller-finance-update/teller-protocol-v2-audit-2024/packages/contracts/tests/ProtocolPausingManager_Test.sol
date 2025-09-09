// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "forge-std/Test.sol"; 

import "../contracts/pausing/ProtocolPausingManager.sol";


contract ProtocolPausingManagerTest is Test {
    ProtocolPausingManager protocolPausingManager;
    address owner = address(0x1);
    address pauser = address(0x2);
    address nonPauser = address(0x3);

    // Events
    event PausedProtocol(address indexed account);
    event UnpausedProtocol(address indexed account);
    event PausedLiquidations(address indexed account);
    event UnpausedLiquidations(address indexed account);
    event PauserAdded(address indexed account);
    event PauserRemoved(address indexed account);

    function setUp() public {
        vm.startPrank(owner);
        protocolPausingManager = new ProtocolPausingManager();
        protocolPausingManager.initialize();
        protocolPausingManager.addPauser(pauser);
        vm.stopPrank();
    }

    function test_addPauser() public {
        vm.startPrank(owner);
        vm.expectEmit(true, true, true, true);
        emit PauserAdded(nonPauser);
        protocolPausingManager.addPauser(nonPauser);
        assertTrue(protocolPausingManager.isPauser(nonPauser));
        vm.stopPrank();
    }

    function test_removePauser() public {
        vm.startPrank(owner);
        protocolPausingManager.addPauser(nonPauser);
        vm.expectEmit(true, true, true, true);
        emit PauserRemoved(nonPauser);
        protocolPausingManager.removePauser(nonPauser);
        assertFalse(protocolPausingManager.isPauser(nonPauser));
        vm.stopPrank();
    }

    function test_pauseProtocol() public {
        vm.startPrank(pauser);
        vm.expectEmit(true, true, true, true);
        emit PausedProtocol(pauser);
        protocolPausingManager.pauseProtocol();
        assertTrue(protocolPausingManager.protocolPaused());
        vm.stopPrank();
    }

    function test_unpauseProtocol() public {
        vm.startPrank(pauser);
        protocolPausingManager.pauseProtocol();
        vm.expectEmit(true, true, true, true);
        emit UnpausedProtocol(pauser);
        protocolPausingManager.unpauseProtocol();
        assertFalse(protocolPausingManager.protocolPaused());
        vm.stopPrank();
    }

    function test_pauseLiquidations() public {
        vm.startPrank(pauser);
        vm.expectEmit(true, true, true, true);
        emit PausedLiquidations(pauser);
        protocolPausingManager.pauseLiquidations();
        assertTrue(protocolPausingManager.liquidationsPaused());
        vm.stopPrank();
    }

    function test_unpauseLiquidations() public {
        vm.startPrank(pauser);
        protocolPausingManager.pauseLiquidations();
        vm.expectEmit(true, true, true, true);
        emit UnpausedLiquidations(pauser);
        protocolPausingManager.unpauseLiquidations();
        assertFalse(protocolPausingManager.liquidationsPaused());
        vm.stopPrank();
    }

    function test_pauseProtocolFailsIfNotPauser() public {
        vm.startPrank(nonPauser);
        vm.expectRevert();
        protocolPausingManager.pauseProtocol();
        vm.stopPrank();
    }

    function test_unpauseProtocolFailsIfNotPauser() public {
        vm.startPrank(pauser);
        protocolPausingManager.pauseProtocol();
        vm.stopPrank();

        vm.startPrank(nonPauser);
        vm.expectRevert();
        protocolPausingManager.unpauseProtocol();
        vm.stopPrank();
    }

    function test_pauseLiquidationsFailsIfNotPauser() public {
        vm.startPrank(nonPauser);
        vm.expectRevert();
        protocolPausingManager.pauseLiquidations();
        vm.stopPrank();
    }

    function test_unpauseLiquidationsFailsIfNotPauser() public {
        vm.startPrank(pauser);
        protocolPausingManager.pauseLiquidations();
        vm.stopPrank();

        vm.startPrank(nonPauser);
        vm.expectRevert();
        protocolPausingManager.unpauseLiquidations();
        vm.stopPrank();
    }
}
