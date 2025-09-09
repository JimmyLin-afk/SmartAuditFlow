// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { Testable } from "../Testable.sol";

import { TellerV2 } from "../../contracts/TellerV2.sol";
import { MarketRegistry } from "../../contracts/MarketRegistry.sol";
import { ReputationManager } from "../../contracts/ReputationManager.sol";

import "../../contracts/interfaces/IMarketRegistry.sol";
import "../../contracts/interfaces/IReputationManager.sol";

import "../../contracts/EAS/TellerAS.sol";

import "../../contracts/mock/WethMock.sol";
import "../../contracts/interfaces/IWETH.sol";

import { User } from "../Test_Helpers.sol";

import "../../contracts/escrow/CollateralEscrowV1.sol";
import "@openzeppelin/contracts/proxy/beacon/UpgradeableBeacon.sol";
import "../../contracts/LenderCommitmentForwarder/LenderCommitmentForwarder_G1.sol";
import "../tokens/TestERC20Token.sol";

import {CollateralManager} from  "../../contracts/CollateralManager.sol";
import { Collateral } from "../../contracts/interfaces/escrow/ICollateralEscrowV1.sol";
import { PaymentType } from "../../contracts/libraries/V2Calculations.sol";
import { BidState, Payment } from "../../contracts/TellerV2Storage.sol";

import "../../contracts/MetaForwarder.sol";
import { LenderManager } from "../../contracts/LenderManager.sol";
import { EscrowVault } from "../../contracts/EscrowVault.sol";
import { ProtocolPausingManager } from "../../contracts/pausing/ProtocolPausingManager.sol";

contract TellerV2_Test is Testable {
    TellerV2User private marketOwner;
    TellerV2User private borrower;
    TellerV2User private lender;

    TellerV2 tellerV2;

    WethMock wethMock;
    TestERC20Token daiMock;
    CollateralManager collateralManager;
    ProtocolPausingManager protocolPausingManager;

    uint256 marketId1;
    uint256 collateralAmount = 10;

    function setUp() public {
        // Deploy test tokens
        wethMock = new WethMock();
        daiMock = new TestERC20Token("Dai", "DAI", 10000000, 18);

        // Deploy Escrow beacon
        CollateralEscrowV1 escrowImplementation = new CollateralEscrowV1();
        UpgradeableBeacon escrowBeacon = new UpgradeableBeacon(
            address(escrowImplementation)
        );

        // Deploy protocol
        tellerV2 = new TellerV2(address(0));

        // Deploy MarketRegistry & ReputationManager
        IMarketRegistry marketRegistry = IMarketRegistry(new MarketRegistry());
        IReputationManager reputationManager = IReputationManager(
            new ReputationManager()
        );
        reputationManager.initialize(address(tellerV2));

        // Deploy Collateral manager
        collateralManager = new CollateralManager();
        collateralManager.initialize(address(escrowBeacon), address(tellerV2));

        // Deploy Lender manager
        MetaForwarder metaforwarder = new MetaForwarder();
        metaforwarder.initialize();
        LenderManager lenderManager = new LenderManager((marketRegistry));
        lenderManager.initialize();
        lenderManager.transferOwnership(address(tellerV2));

        EscrowVault escrowVault = new EscrowVault();
        escrowVault.initialize();

        protocolPausingManager = new ProtocolPausingManager();
        protocolPausingManager.initialize();

        // Deploy LenderCommitmentForwarder
        LenderCommitmentForwarder_G1 lenderCommitmentForwarder = new LenderCommitmentForwarder_G1(
                address(tellerV2),
                address(marketRegistry)
            );

        // Initialize protocol
        tellerV2.initialize(
            50,
            address(marketRegistry),
            address(reputationManager),
            address(lenderCommitmentForwarder),
            address(collateralManager),
            address(lenderManager),
            address(escrowVault),
            address(protocolPausingManager)
        );

        // Instantiate users & balances
        marketOwner = new TellerV2User(address(tellerV2), wethMock);
        borrower = new TellerV2User(address(tellerV2), wethMock);
        lender = new TellerV2User(address(tellerV2), wethMock);

        uint256 balance = 50000;
        payable(address(borrower)).transfer(balance);
        payable(address(lender)).transfer(balance * 10);
        borrower.depositToWeth(balance);
        lender.depositToWeth(balance * 10);

        daiMock.transfer(address(lender), balance * 10);
        daiMock.transfer(address(borrower), balance);
        // Approve Teller V2 for the lender's dai
        lender.addAllowance(address(daiMock), address(tellerV2), balance * 10);

        // Create a market
        marketId1 = marketOwner.createMarket(
            address(marketRegistry),
            8000,
            7000,
            5000,
            500,
            false,
            false,
            PaymentType.EMI,
            PaymentCycleType.Seconds,
            "uri://"
        );
    }

    function submitCollateralBid() public returns (uint256 bidId_) {
        Collateral memory info;
        info._amount = collateralAmount;
        info._tokenId = 0;
        info._collateralType = CollateralType.ERC20;
        info._collateralAddress = address(wethMock);

        Collateral[] memory collateralInfo = new Collateral[](1);
        collateralInfo[0] = info;

        uint256 bal = wethMock.balanceOf(address(borrower));

        // Increase allowance
        // Approve the collateral manager for the borrower's weth
        borrower.addAllowance(
            address(wethMock),
            address(collateralManager),
            info._amount
        );

        bidId_ = borrower.submitCollateralBid(
            address(daiMock),
            marketId1,
            100,
            10000,
            500,
            "metadataUri://",
            address(borrower),
            collateralInfo
        );
    }

    function acceptBid(uint256 _bidId) public {
        // Accept bid
        lender.acceptBid(_bidId);
    }

    function test_collateralEscrow() public {

          

        // Submit bid as borrower
        uint256 bidId = submitCollateralBid();
        // Accept bid as lender
        acceptBid(bidId);

        // Get newly created escrow
        address escrowAddress = collateralManager._escrows(bidId);
        CollateralEscrowV1 escrow = CollateralEscrowV1(escrowAddress);

        uint256 storedBidId = escrow.getBid();

        // Test that the created escrow has the same bidId and collateral stored
        assertEq(bidId, storedBidId, "Collateral escrow was not created");

        uint256 escrowBalance = wethMock.balanceOf(escrowAddress);

        assertEq(collateralAmount, escrowBalance, "Collateral was not stored");

        vm.warp(100000);

        // Repay loan
        uint256 borrowerBalanceBefore = wethMock.balanceOf(address(borrower));
        Payment memory amountOwed = tellerV2.calculateAmountOwed(
            bidId,
            block.timestamp
        );
        borrower.addAllowance(
            address(daiMock),
            address(tellerV2),
            amountOwed.principal + amountOwed.interest
        );
        borrower.repayLoanFull(bidId);

        // Check escrow balance
        uint256 escrowBalanceAfter = wethMock.balanceOf(escrowAddress);
        assertEq(
            0,
            escrowBalanceAfter,
            "Collateral was not withdrawn from escrow on repayment"
        );

        // Check borrower balance for collateral
        uint256 borrowerBalanceAfter = wethMock.balanceOf(address(borrower));
        assertEq(
            collateralAmount,
            borrowerBalanceAfter - borrowerBalanceBefore,
            "Collateral was not sent to borrower after repayment"
        );
    }

    function test_commit_collateral_frontrun_exploit() public {
        // The original borrower balance for the DAI principal and WETH collateral
        assertEq(daiMock.balanceOf(address(borrower)), 50000);
        assertEq(wethMock.balanceOf(address(borrower)), 50000);

        // The original lender balance for the DAI principal -> This will be stolen
        assertEq(daiMock.balanceOf(address(lender)), 500000);

        // Submit bid as borrower
        uint256 bidId = submitCollateralBid();

        // The original bid is for 10 WETH
        uint256 originalCollateralAmount = collateralManager
            .getCollateralAmount(bidId, address(wethMock));
        assertEq(originalCollateralAmount, 10);

        // This is just to illustrate that some time passes (but it is irrelevant)
        vm.warp(100);

        // A potential lender finds the bid attractive and decides to accept the bid

        // The attack begins here
        // The malicious borrower sees the transaction in the mempool and frontruns it

        // The borrower prepares the malicious bid lowering the amount to the minimum possible
        Collateral memory info;
        info._amount = 1; // @audit minimum amount
        info._tokenId = 0;
        info._collateralAddress = address(wethMock);
        info._collateralType = CollateralType.ERC20;

        Collateral[] memory collateralInfo = new Collateral[](1);
        collateralInfo[0] = info;

        // The malicious borrower performs the attack by frontrunning the tx and updating the bid collateral amount
        vm.prank(address(borrower));
        vm.expectRevert();
        collateralManager.commitCollateral(bidId, info);

        // The lender is now victim to the frontrunning and accepts the malicious bid
        /*  acceptBid(bidId);

        // The borrower now has the expected 95 DAI from the loan (5 DAI are gone in fees)
        // But he only provided 1 WETH as collateral instead of the original amount of 10 WETH
        assertEq(daiMock.balanceOf(address(borrower)), 50095);
        assertEq(wethMock.balanceOf(address(borrower)), 49999); // @audit only provided 1 WETH

        // The lender lost his principal of 100 DAI, as the loan is only collateralized by 1 WETH instead of 10 WETH
        assertEq(daiMock.balanceOf(address(lender)), 499900);*/
    }
}

contract TellerV2User is User {
    WethMock public immutable wethMock;

    constructor(address _tellerV2, WethMock _wethMock) User(_tellerV2) {
        wethMock = _wethMock;
    }

    function depositToWeth(uint256 amount) public {
        wethMock.deposit{ value: amount }();
    }
}
