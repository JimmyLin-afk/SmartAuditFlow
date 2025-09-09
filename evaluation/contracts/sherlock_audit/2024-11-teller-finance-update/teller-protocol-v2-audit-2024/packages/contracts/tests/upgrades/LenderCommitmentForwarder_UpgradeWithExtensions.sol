// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { Testable } from "../Testable.sol";

import "@openzeppelin/contracts/proxy/transparent/ProxyAdmin.sol";
import "@openzeppelin/contracts/proxy/transparent/TransparentUpgradeableProxy.sol";

import "../../contracts/LenderCommitmentForwarder/LenderCommitmentForwarder.sol";
import { LenderCommitmentForwarder_G1 } from "../../contracts/LenderCommitmentForwarder/LenderCommitmentForwarder_G1.sol";
import { LenderCommitmentForwarder_G3 } from "../../contracts/LenderCommitmentForwarder/LenderCommitmentForwarder_G3.sol";

contract LenderCommitmentForwarder_UpgradeToG2 is Testable {
    ProxyAdmin internal admin;
    TransparentUpgradeableProxy internal proxy;

    address internal constant tellerV2 = address(1);
    address internal constant marketRegistry = address(2);

    LenderCommitmentForwarder_G1 internal logicV1 =
        new LenderCommitmentForwarder_G1(tellerV2, marketRegistry);
    LenderCommitmentForwarder_G3 internal logicV2 =
        new LenderCommitmentForwarder_G3(tellerV2, marketRegistry);

    function setUp() public {
        admin = new ProxyAdmin();
        proxy = new TransparentUpgradeableProxy(
            address(logicV1),
            address(admin),
            ""
        );

        // Create 2 commitments to test the ID being incremented
        assertEq(_createCommitment(), 0, "Commitment ID should be 0");
        assertEq(_createCommitment(), 1, "Commitment ID should be 1");
    }

    function _createCommitment() internal returns (uint256 commitmentId_) {
        LenderCommitmentForwarder_G1.Commitment memory commitment;
        commitment.marketId = 1;
        commitment.principalTokenAddress = address(123);
        commitment.maxPrincipal = 10_000_000;
        commitment.maxDuration = 60 days;
        commitment.minInterestRate = 100;
        commitment.expiration = uint32(block.timestamp + 60 days);
        commitment.lender = address(this);

        address[] memory borrowers;
        commitmentId_ = LenderCommitmentForwarder_G1(address(proxy))
            .createCommitment(commitment, borrowers);
    }

    function test_storage_slot_data_after_upgrade() public {
        // This should revert because the contract has not been upgraded yet with that function

        // Upgrade contract to V2
         admin.upgrade(proxy, address(logicV2));
       // admin.upgrade(ITransparentUpgradeableProxy(address(proxy)), address(logicV2));

        // Verify the commitment principalTokenAddress is set to the correct address after the upgrade
        (
            ,
            ,
            ,
            ,
            ,
            ,
            ,
            ,
            ,
            ,
            address principalTokenAddress
        ) = LenderCommitmentForwarder_G3(address(proxy)).commitments(1);
        assertEq(
            principalTokenAddress,
            address(123),
            "Principal token address should be set to 123"
        );
    }
}
