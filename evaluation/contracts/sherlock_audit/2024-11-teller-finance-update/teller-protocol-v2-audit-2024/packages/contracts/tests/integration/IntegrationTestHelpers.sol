pragma solidity >=0.8.0 <0.9.0;
// SPDX-License-Identifier: MIT

import { TellerV2 } from "../../contracts/TellerV2.sol";

import "../../contracts/EAS/TellerAS.sol";
import "../../contracts/EAS/TellerASEIP712Verifier.sol";
import "../../contracts/EAS/TellerASRegistry.sol";

import "@openzeppelin/contracts/proxy/beacon/UpgradeableBeacon.sol";
import "@openzeppelin/contracts/proxy/beacon/BeaconProxy.sol";

import { MarketRegistry } from "../../contracts/MarketRegistry.sol";

import { ProtocolPausingManager } from "../../contracts/pausing/ProtocolPausingManager.sol";
import { EscrowVault } from "../../contracts/EscrowVault.sol";
import { LenderManager } from "../../contracts/LenderManager.sol";
import { LenderCommitmentForwarder_G3 } from "../../contracts/LenderCommitmentForwarder/LenderCommitmentForwarder_G3.sol";
import { CollateralManager } from "../../contracts/CollateralManager.sol";
import { CollateralEscrowV1 } from "../../contracts/escrow/CollateralEscrowV1.sol";

import { ReputationManager } from "../../contracts/ReputationManager.sol";
import { IMarketRegistry } from "../../contracts/interfaces/IMarketRegistry.sol";

library IntegrationTestHelpers {
    function deployMarketRegistry() public returns (address) {
        IASRegistry iasRegistry = new TellerASRegistry();
        IEASEIP712Verifier ieaseip712verifier = new TellerASEIP712Verifier();

        TellerAS tellerAS = new TellerAS((iasRegistry), (ieaseip712verifier));
        MarketRegistry marketRegistry = new MarketRegistry();

        marketRegistry.initialize(tellerAS);

        return address(marketRegistry);
    }

    function deployIntegrationSuite() public returns (TellerV2 tellerV2_) {
        address trustedForwarder = address(0);
        TellerV2 tellerV2 = new TellerV2(trustedForwarder);

        uint16 _protocolFee = 100;
        address _marketRegistry = deployMarketRegistry();
        ReputationManager _reputationManager = new ReputationManager();

        LenderCommitmentForwarder_G3 _lenderCommitmentForwarder = new LenderCommitmentForwarder_G3(
                address(tellerV2),
                address(_marketRegistry)
            );

        CollateralEscrowV1 escrowImplementation = new CollateralEscrowV1();
        // Deploy beacon contract with implementation
        UpgradeableBeacon escrowBeacon = new UpgradeableBeacon(
            address(escrowImplementation)
        );

        CollateralManager _collateralManager = new CollateralManager();
        LenderManager _lenderManager = new LenderManager(
            IMarketRegistry(_marketRegistry)
        );
        EscrowVault _escrowVault = new EscrowVault();
        ProtocolPausingManager _protocolPausingManager = new ProtocolPausingManager();

        _protocolPausingManager.initialize();  //need to xfer ownership ..?
        _collateralManager.initialize(address(escrowBeacon), address(tellerV2));
        _lenderManager.initialize();
        _reputationManager.initialize(address(tellerV2));

        tellerV2.initialize(
            _protocolFee,
            address(_marketRegistry),
            address(_reputationManager),
            address(_lenderCommitmentForwarder),
            address(_collateralManager),
            address(_lenderManager),
            address(_escrowVault),
            address(_protocolPausingManager)
        );

        return tellerV2;
    }
}
