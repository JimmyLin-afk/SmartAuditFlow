// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { StdStorage, stdStorage } from "forge-std/StdStorage.sol";
import { Testable } from "../Testable.sol";
import { TellerV2_Override } from "./TellerV2_Override.sol";
import { Bid, BidState, Collateral } from "../../contracts/TellerV2.sol";

contract TellerV2_initialize is Testable {
    using stdStorage for StdStorage;

    TellerV2_Override tellerV2;

    uint16 protocolFee = 5;

    Contract marketRegistry;
    Contract reputationManager;
    Contract lenderCommitmentForwarder;
    Contract collateralManager;
    Contract lenderManager;
    Contract escrowVault;
    Contract protocolPausingManager;

    function setUp() public {
        tellerV2 = new TellerV2_Override();

        //stdstore.target(address(tellerV2)).sig("marketRegistry()").checked_write(address(0x1234));
    }

    function test_initialize() public {
        marketRegistry = new Contract();
        reputationManager = new Contract();
        lenderCommitmentForwarder = new Contract();
        collateralManager = new Contract();
        lenderManager = new Contract();
        escrowVault = new Contract();
        protocolPausingManager = new Contract();

        tellerV2.initialize(
            protocolFee,
            address(marketRegistry),
            address(reputationManager),
            address(lenderCommitmentForwarder),
            address(collateralManager),
            address(lenderManager),
            address(escrowVault),
            address(protocolPausingManager)
        );

        assertEq(address(tellerV2.marketRegistry()), address(marketRegistry));
        assertEq(address(tellerV2.lenderManager()), address(lenderManager));
        assertEq(address(tellerV2.escrowVault()), address(escrowVault));
    }

    function test_initialize_lender_commitment_forwarder_not_contract() public {
        marketRegistry = new Contract();
        reputationManager = new Contract();

        collateralManager = new Contract();
        lenderManager = new Contract();
        escrowVault = new Contract();
        protocolPausingManager = new Contract();

        vm.expectRevert("LCF_ic");

        tellerV2.initialize(
            protocolFee,
            address(marketRegistry),
            address(reputationManager),
            address(lenderCommitmentForwarder),
            address(collateralManager),
            address(lenderManager),
            address(escrowVault),
            address(protocolPausingManager)
        );
    }

    function test_initialize_market_registry_not_contract() public {
        reputationManager = new Contract();

        lenderCommitmentForwarder = new Contract();
        collateralManager = new Contract();
        lenderManager = new Contract();
        escrowVault = new Contract();
        protocolPausingManager = new Contract();

        vm.expectRevert("MR_ic");

        tellerV2.initialize(
            protocolFee,
            address(marketRegistry),
            address(reputationManager),
            address(lenderCommitmentForwarder),
            address(collateralManager),
            address(lenderManager),
            address(escrowVault),
            address(protocolPausingManager)
        );
    }

    function test_initialize_reputation_manager_not_contract() public {
        marketRegistry = new Contract();

        lenderCommitmentForwarder = new Contract();
        collateralManager = new Contract();
        lenderManager = new Contract();
        escrowVault = new Contract();
        protocolPausingManager = new Contract();

        vm.expectRevert("RM_ic");

        tellerV2.initialize(
            protocolFee,
            address(marketRegistry),
            address(reputationManager),
            address(lenderCommitmentForwarder),
            address(collateralManager),
            address(lenderManager),
            address(escrowVault),
            address(protocolPausingManager)
        );
    }

    function test_initialize_collateral_manager_not_contract() public {
        marketRegistry = new Contract();

        lenderCommitmentForwarder = new Contract();
        reputationManager = new Contract();
        lenderManager = new Contract();
        escrowVault = new Contract();
        protocolPausingManager = new Contract();

        vm.expectRevert("CM_ic");

        tellerV2.initialize(
            protocolFee,
            address(marketRegistry),
            address(reputationManager),
            address(lenderCommitmentForwarder),
            address(collateralManager),
            address(lenderManager),
            address(escrowVault),
            address(protocolPausingManager)
        );
    }

    function test_initialize_lender_manager_not_contract() public {
        marketRegistry = new Contract();

        lenderCommitmentForwarder = new Contract();
        reputationManager = new Contract();
        collateralManager = new Contract();
        escrowVault = new Contract();
        protocolPausingManager = new Contract();

        vm.expectRevert("LM_ic");

        tellerV2.initialize(
            protocolFee,
            address(marketRegistry),
            address(reputationManager),
            address(lenderCommitmentForwarder),
            address(collateralManager),
            address(lenderManager),
            address(escrowVault),
            address(protocolPausingManager)
        );
    }

    function test_setLenderManager_external() public {
        //how to mock self as the owner ?
        //tellerV2.setLenderManager(address(lenderManager));
    }

    function test_setReputationManager_external() public {
        //how to mock self as the owner ?
        //tellerV2.setReputationManager(address(reputationManager));
    }

    function test_setLenderManager_internal() public {
        lenderManager = new Contract();

        tellerV2.setLenderManagerSuper(address(lenderManager));

        assertEq(address(tellerV2.lenderManager()), address(lenderManager));
    }
}

contract User {}

contract Contract {
    //adding some storage so this is recognizes as a contract
    address public target;

    constructor() {
        target = msg.sender;
    }

    function execute(bytes memory data) public returns (bytes memory) {
        (bool success, bytes memory result) = target.delegatecall(data);
        require(success, "User: delegatecall failed");
        return result;
    }
}
