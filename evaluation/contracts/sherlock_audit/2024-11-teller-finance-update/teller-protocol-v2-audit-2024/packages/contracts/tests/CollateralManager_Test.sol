// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { Testable } from "./Testable.sol";

import { CollateralEscrowV1 } from "../contracts/escrow/CollateralEscrowV1.sol";
import "../contracts/mock/WethMock.sol";
import "@openzeppelin/contracts/proxy/beacon/UpgradeableBeacon.sol";
import "@openzeppelin/contracts/proxy/beacon/BeaconProxy.sol";

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

import "./tokens/TestERC20Token.sol";
import "./tokens/TestERC721Token.sol";
import "./tokens/TestERC1155Token.sol";

import "../contracts/mock/TellerV2SolMock.sol";
import "../contracts/CollateralManager.sol";
import "../contracts/pausing/ProtocolPausingManager.sol";
import "./CollateralManager_Override.sol";

contract CollateralManager_Test is Testable {
    CollateralManager_Override collateralManager;
    User private borrower;
    User private lender;
    User private liquidator;

    TestERC20Token wethMock;
    TestERC20Token otherErc20Mock;
    TestERC721Token erc721Mock;
    TestERC1155Token erc1155Mock;

    TellerV2_Mock tellerV2Mock;

    CollateralEscrowV1_Mock escrowImplementation =
        new CollateralEscrowV1_Mock();

    event CollateralCommitted(
        uint256 _bidId,
        CollateralType _type,
        address _collateralAddress,
        uint256 _amount,
        uint256 _tokenId
    );
    event CollateralClaimed(uint256 _bidId);
    event CollateralDeposited(
        uint256 _bidId,
        CollateralType _type,
        address _collateralAddress,
        uint256 _amount,
        uint256 _tokenId
    );
    event CollateralWithdrawn(
        uint256 _bidId,
        CollateralType _type,
        address _collateralAddress,
        uint256 _amount,
        uint256 _tokenId,
        address _recipient
    );

    function setUp() public {
        // Deploy beacon contract with implementation
        UpgradeableBeacon escrowBeacon = new UpgradeableBeacon(
            address(escrowImplementation)
        );

        wethMock = new TestERC20Token("wrappedETH", "WETH", 1e24, 18);
        otherErc20Mock = new TestERC20Token("wrappedETH", "WETH", 1e24, 18);
        erc721Mock = new TestERC721Token("ERC721", "ERC721");
        erc1155Mock = new TestERC1155Token("ERC1155");

        tellerV2Mock = new TellerV2_Mock();
        borrower = new User();
        lender = new User();
        liquidator = new User();

        ProtocolPausingManager protocolPausingManager = new ProtocolPausingManager();
        protocolPausingManager.initialize();

        tellerV2Mock.setProtocolPausingManager(address(protocolPausingManager));
      
        //  uint256 borrowerBalance = 50000;
        //   payable(address(borrower)).transfer(borrowerBalance);

        collateralManager = new CollateralManager_Override();

        collateralManager.initialize(
            address(escrowBeacon),
            address(tellerV2Mock)
        );
    }

    function test_initialize_valid() public {
        UpgradeableBeacon eBeacon = new UpgradeableBeacon(
            address(escrowImplementation)
        );

        CollateralManager_Override tempCManager = new CollateralManager_Override();
        tempCManager.initialize(address(eBeacon), address(tellerV2Mock));

        address managerTellerV2 = address(tempCManager.tellerV2());
        assertEq(
            managerTellerV2,
            address(tellerV2Mock),
            "CollateralManager was not initialized"
        );
    }

    function test_setCollateralEscrowBeacon() public {
        // Deploy implementation
        CollateralEscrowV1 escrowImplementation = new CollateralEscrowV1_Mock();
        // Deploy beacon contract with implementation
        UpgradeableBeacon escrowBeacon = new UpgradeableBeacon(
            address(escrowImplementation)
        );

        collateralManager.setCollateralEscrowBeacon(address(escrowBeacon));

        //how to test ?
    }

    function test_setCollateralEscrowBeacon_invalid_twice() public {
        CollateralEscrowV1 escrowImplementation = new CollateralEscrowV1_Mock();
        // Deploy beacon contract with implementation
        UpgradeableBeacon escrowBeacon = new UpgradeableBeacon(
            address(escrowImplementation)
        );
        collateralManager.setCollateralEscrowBeacon(address(escrowBeacon));

        vm.expectRevert("Initializable: contract is already initialized");
        collateralManager.setCollateralEscrowBeacon(address(escrowBeacon));
        //
    }

    function test_deposit() public {
        uint256 bidId = 0;
        uint256 amount = 1000;
        wethMock.transfer(address(borrower), amount);

        borrower.approveERC20(
            address(wethMock),
            address(collateralManager),
            amount
        );

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: amount,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        tellerV2Mock.setBorrower(address(borrower));

        collateralManager.setGlobalEscrowProxyAddress(
            address(escrowImplementation)
        );

        vm.expectEmit(false, false, false, false);
        emit CollateralDeposited(
            bidId,
            collateral._collateralType,
            collateral._collateralAddress,
            collateral._amount,
            collateral._tokenId
        );
        collateralManager._depositSuper(bidId, collateral);
    }

    function test_deposit_collateral_erc20_not_validated() public {
        uint256 bidId = 0;
        uint256 amount = 0;
        wethMock.transfer(address(borrower), amount);

        borrower.approveERC20(
            address(wethMock),
            address(collateralManager),
            amount
        );

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: amount,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        tellerV2Mock.setBorrower(address(borrower));

        collateralManager.setGlobalEscrowProxyAddress(
            address(escrowImplementation)
        );

        vm.expectRevert("Collateral not validated");
        collateralManager._depositSuper(bidId, collateral);
    }

    function test_deposit_erc20() public {
        uint256 bidId = 0;
        uint256 amount = 1000;
        wethMock.transfer(address(borrower), amount);

        borrower.approveERC20(
            address(wethMock),
            address(collateralManager),
            amount
        );

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: amount,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        tellerV2Mock.setBorrower(address(borrower));

        collateralManager.setGlobalEscrowProxyAddress(
            address(escrowImplementation)
        );

        vm.expectEmit(false, false, false, false);
        emit CollateralDeposited(
            bidId,
            collateral._collateralType,
            collateral._collateralAddress,
            collateral._amount,
            collateral._tokenId
        );
        vm.prank(address(borrower));
        collateralManager._depositSuper(bidId, collateral);

        assertEq(
            escrowImplementation.depositAssetWasCalled(),
            true,
            "deposit token was not called"
        );
    }

    function test_deposit_erc721() public {
        uint256 bidId = 0;
        uint256 amount = 1000;

        uint256 tokenId = erc721Mock.mint(address(borrower));

        vm.prank(address(borrower));
        erc721Mock.approve(address(collateralManager), tokenId);

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC721,
            _amount: amount,
            _tokenId: tokenId,
            _collateralAddress: address(erc721Mock)
        });

        tellerV2Mock.setBorrower(address(borrower));

        collateralManager.setGlobalEscrowProxyAddress(
            address(escrowImplementation)
        );

        vm.expectEmit(false, false, false, false);
        emit CollateralDeposited(
            bidId,
            collateral._collateralType,
            collateral._collateralAddress,
            collateral._amount,
            collateral._tokenId
        );
        vm.prank(address(borrower));
        collateralManager._depositSuper(bidId, collateral);

        assertEq(
            escrowImplementation.depositAssetWasCalled(),
            true,
            "deposit asset was not called"
        );
    }

    function test_deposit_erc721_amount_invalid() public {
        uint256 bidId = 0;
        uint256 amount = 1000;

        uint256 tokenId = erc721Mock.mint(address(borrower));

        vm.prank(address(borrower));
        erc721Mock.approve(address(collateralManager), tokenId);

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC721,
            _amount: 0,
            _tokenId: tokenId,
            _collateralAddress: address(erc721Mock)
        });

        tellerV2Mock.setBorrower(address(borrower));

        collateralManager.setGlobalEscrowProxyAddress(
            address(escrowImplementation)
        );

        vm.expectRevert("Collateral not validated");
        vm.prank(address(borrower));
        collateralManager._depositSuper(bidId, collateral);
    }

    function test_deposit_erc1155() public {
        uint256 bidId = 0;
        uint256 amount = 1000;

        erc1155Mock.mint(address(borrower), 1);

        vm.prank(address(borrower));
        erc1155Mock.setApprovalForAll(address(collateralManager), true);

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC1155,
            _amount: 1,
            _tokenId: 0,
            _collateralAddress: address(erc1155Mock)
        });

        tellerV2Mock.setBorrower(address(borrower));

        collateralManager.setGlobalEscrowProxyAddress(
            address(escrowImplementation)
        );

        vm.expectEmit(false, false, false, false);
        emit CollateralDeposited(
            bidId,
            collateral._collateralType,
            collateral._collateralAddress,
            collateral._amount,
            collateral._tokenId
        );
        vm.prank(address(borrower));
        collateralManager._depositSuper(bidId, collateral);

        assertEq(
            escrowImplementation.depositAssetWasCalled(),
            true,
            "deposit asset was not called"
        );
    }

    function test_deposit_erc1155_amount_invalid() public {
        uint256 bidId = 0;
        uint256 amount = 1000;

        vm.prank(address(borrower));
        erc1155Mock.setApprovalForAll(address(collateralManager), true);

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC1155,
            _amount: 0,
            _tokenId: 0,
            _collateralAddress: address(erc1155Mock)
        });

        tellerV2Mock.setBorrower(address(borrower));

        collateralManager.setGlobalEscrowProxyAddress(
            address(escrowImplementation)
        );

        vm.expectRevert("Collateral not validated");
        vm.prank(address(borrower));
        collateralManager._depositSuper(bidId, collateral);
    }

    /*function test_deposit_invalid_bid() public  {
        uint256 bidId = 0 ;
        uint256 amount = 1000;
        wethMock.transfer(address(borrower), amount);
        wethMock.approve(address(collateralManager), amount);

        //borrower.approveERC20( address(wethMock), address(collateralManager), amount  );
    

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: amount,
            _tokenId: 0, 
            _collateralAddress: address(wethMock)
        });


        tellerV2Mock.setBorrower(address(borrower));

        vm.expectRevert("Bid does not exist");
        vm.prank(address(borrower));
        collateralManager._depositSuper(bidId, collateral);

       
         
    }*/

    function test_deployAndDeposit_invalid_sender() public {
        vm.prank(address(lender));
        vm.expectRevert("Sender not authorized");
        collateralManager.deployAndDeposit(0);
    }

    function test_deployAndDeposit_not_backed() public {
        uint256 bidId = 0;
        uint256 amount = 1000;
        wethMock.transfer(address(borrower), amount);
        wethMock.approve(address(collateralManager), amount);

        collateralManager.setBidsCollateralBackedGlobally(false);

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: amount,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        tellerV2Mock.setBorrower(address(borrower));

        vm.prank(address(tellerV2Mock));
        collateralManager.deployAndDeposit(bidId);

        assertFalse(
            collateralManager.deployEscrowInternalWasCalled(),
            "deploy escrow internal was called"
        );
    }

    function test_deployAndDeposit_backed() public {
        uint256 bidId = 0;
        uint256 amount = 1000;
        wethMock.transfer(address(borrower), amount);
        wethMock.approve(address(collateralManager), amount);

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: amount,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        collateralManager.commitCollateralSuper(bidId, collateral);

        tellerV2Mock.setBorrower(address(borrower));

        collateralManager.setBidsCollateralBackedGlobally(true);

        vm.prank(address(tellerV2Mock));
        collateralManager.deployAndDeposit(bidId);

        assertTrue(
            collateralManager.deployEscrowInternalWasCalled(),
            "deploy escrow internal was not called"
        );

        assertTrue(
            collateralManager.depositInternalWasCalled(),
            "deposit internal was not called"
        );
    }

    function test_deployAndDeposit_backed_empty_array() public {
        uint256 bidId = 0;
        uint256 amount = 1000;
        wethMock.transfer(address(borrower), amount);
        wethMock.approve(address(collateralManager), amount);

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: amount,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        tellerV2Mock.setBorrower(address(borrower));

        collateralManager.setBidsCollateralBackedGlobally(true);

        vm.prank(address(tellerV2Mock));
        collateralManager.deployAndDeposit(bidId);

        assertTrue(
            collateralManager.deployEscrowInternalWasCalled(),
            "deploy escrow internal was called"
        );
        assertFalse(
            collateralManager.depositInternalWasCalled(),
            "deposit internal was called"
        );
    }

    function test_initialize_again() public {
        vm.expectRevert("Initializable: contract is already initialized");
        collateralManager.initialize(address(0), address(0));
    }

    function test_withdraw_external_invalid_bid_state() public {
        uint256 bidId = 0;

        tellerV2Mock.setGlobalBidState(BidState.PENDING);

        vm.expectRevert("collateral cannot be withdrawn");
        collateralManager.withdraw(bidId);
    }

    function test_withdraw_external_invalid_bid_state_liquidated() public {
        uint256 bidId = 0;

        tellerV2Mock.setGlobalBidState(BidState.LIQUIDATED);

        vm.expectRevert("collateral cannot be withdrawn");
        collateralManager.withdraw(bidId);
    }

    function test_withdraw_external_state_paid() public {
        uint256 bidId = 0;

        tellerV2Mock.setBorrower(address(borrower));
        tellerV2Mock.setGlobalBidState(BidState.PAID);

        collateralManager.withdraw(bidId);

        assertEq(
            collateralManager.withdrawInternalWasCalledToRecipient(),
            address(borrower),
            "withdraw internal was not called with correct recipient"
        );
    }

    function test_lenderClaimCollateral_invalid_sender() public {
        uint256 bidId = 0;

        tellerV2Mock.setGlobalBidState(BidState.CLOSED);

        vm.expectRevert("Sender not authorized");
        collateralManager.lenderClaimCollateral(bidId);
    }

    function test_lenderClaimCollateral() public {
        uint256 bidId = 0;

        tellerV2Mock.setLender(address(lender));

        tellerV2Mock.setGlobalBidState(BidState.CLOSED);
        collateralManager.setBidsCollateralBackedGlobally(true);

        vm.expectEmit(true, false, false, false);
        emit CollateralClaimed(bidId);
        vm.prank(address(tellerV2Mock));
        collateralManager.lenderClaimCollateral(bidId);

        assertEq(
            collateralManager.withdrawInternalWasCalledToRecipient(),
            address(lender),
            "withdraw internal was not called with correct recipient"
        );
    }

    function test_liquidateCollateral_invalid_sender() public {
        uint256 bidId = 0;

        tellerV2Mock.setGlobalBidState(BidState.LIQUIDATED);

        vm.expectRevert("Sender not authorized");
        collateralManager.liquidateCollateral(bidId, address(liquidator));
    }

    function test_liquidateCollateral_invalid_state() public {
        uint256 bidId = 0;

        collateralManager.setBidsCollateralBackedGlobally(true);

        vm.prank(address(tellerV2Mock));
        //tellerV2Mock.setGlobalBidState(BidState.LIQUIDATED);

        vm.expectRevert("Loan has not been liquidated");
        collateralManager.liquidateCollateral(bidId, address(liquidator));
    }

    function test_liquidateCollateral_not_backed() public {
        uint256 bidId = 0;

        collateralManager.setBidsCollateralBackedGlobally(false);

        tellerV2Mock.setGlobalBidState(BidState.PENDING);

        vm.prank(address(tellerV2Mock));
        collateralManager.liquidateCollateral(bidId, address(liquidator));

        assertTrue(
            collateralManager.withdrawInternalWasCalledToRecipient() ==
                address(0),
            "withdraw internal should not have been called"
        );
    }

    function test_liquidateCollateral() public {
        uint256 bidId = 0;

        collateralManager.setBidsCollateralBackedGlobally(true);

        tellerV2Mock.setGlobalBidState(BidState.LIQUIDATED);

        vm.prank(address(tellerV2Mock));
        collateralManager.liquidateCollateral(bidId, address(liquidator));

        assertTrue(
            collateralManager.withdrawInternalWasCalledToRecipient() ==
                address(liquidator),
            "withdraw internal was not called with correct recipient"
        );
    }

    function test_commit_collateral_address_multiple_times() public {
        uint256 bidId = 0;
        address recipient = address(borrower);

        wethMock.transfer(address(escrowImplementation), 1000);

        Collateral memory collateralInfo = Collateral({
            _collateralType: CollateralType.ERC721,
            _amount: 1,
            _tokenId: 2,
            _collateralAddress: address(wethMock)
        });

        collateralManager._commitCollateralSuper(bidId, collateralInfo);

        vm.expectRevert(
            "Cannot commit multiple collateral with the same address"
        );
        collateralManager._commitCollateralSuper(bidId, collateralInfo);
    }

    function test_commit_collateral_ERC721_amount_1() public {
        uint256 bidId = 0;
        address recipient = address(borrower);

        wethMock.transfer(address(escrowImplementation), 1000);

        Collateral memory collateralInfo = Collateral({
            _collateralType: CollateralType.ERC721,
            _amount: 1000,
            _tokenId: 2,
            _collateralAddress: address(wethMock)
        });

        vm.expectRevert("ERC721 collateral must have amount of 1");
        collateralManager._commitCollateralSuper(bidId, collateralInfo);
    }

    function test_withdraw_internal() public {
        uint256 bidId = 0;
        address recipient = address(borrower);

        wethMock.transfer(address(escrowImplementation), 1000);

        Collateral memory collateralInfo = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        collateralManager._commitCollateralSuper(bidId, collateralInfo);

        collateralManager.forceSetEscrowAddress(
            bidId,
            address(escrowImplementation)
        );
        collateralManager._withdrawSuper(bidId, recipient);

        assertTrue(
            escrowImplementation.withdrawWasCalled(),
            "withdraw was not called on escrow imp"
        );
    }

    function test_withdraw_internal_emptyArray() public {
        uint256 bidId = 0;
        address recipient = address(borrower);

        wethMock.transfer(address(escrowImplementation), 1000);

        collateralManager.forceSetEscrowAddress(
            bidId,
            address(escrowImplementation)
        );
        collateralManager._withdrawSuper(bidId, recipient);

        assertFalse(
            escrowImplementation.withdrawWasCalled(),
            "withdraw was not called on escrow imp"
        );
    }

    /*
    function test_withdraw_internal_invalid_bid() public {

        uint256 bidId = 0;
        address recipient = address(borrower);

        vm.expectRevert("Bid does not exist");
        collateralManager.withdraw_internal(bidId,recipient);

    }
*/

    function test_getCollateralInfo() public {
        uint256 bidId = 0;

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        collateralManager.commitCollateralSuper(bidId, collateral);

        Collateral[] memory collateralInfo = collateralManager
            .getCollateralInfo(bidId);

        assertTrue(
            collateralInfo[0]._collateralType == CollateralType.ERC20,
            "collateral type is not correct"
        );
        assertTrue(
            collateralInfo[0]._amount == 1000,
            "collateral amount is not correct"
        );
        assertTrue(
            collateralInfo[0]._tokenId == 0,
            "collateral tokenId is not correct"
        );
        assertTrue(
            collateralInfo[0]._collateralAddress == address(wethMock),
            "collateral address is not correct"
        );
    }

    function test_getCollateralAmount() public {
        uint256 bidId = 0;

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        collateralManager.commitCollateralSuper(bidId, collateral);

        uint256 collateralAmount = collateralManager.getCollateralAmount(
            bidId,
            address(wethMock)
        );

        assertTrue(
            collateralAmount == 1000,
            "collateral amount is not correct"
        );
    }

    function test_getEscrow() public {
        uint256 bidId = 0;

        address escrow = collateralManager.getEscrow(bidId);

        assertTrue(escrow == address(0), "escrow is not correct");
    }

    function test_deployEscrow_internal() public {
        uint256 bidId = 0;

        tellerV2Mock.setBorrower(address(borrower));

        //use the mock escrow imp
        collateralManager.setGlobalEscrowProxyAddress(address(0));

        (address proxy, address borrower) = collateralManager
            ._deployEscrowSuper(bidId);

        assertTrue(proxy != address(0), "proxy was not depoloyed");
    }

    function test_deployEscrow_internal_nonexisting_bid() public {
        uint256 bidId = 0;

        vm.expectRevert("Bid does not exist");
        (address proxy, address borrower) = collateralManager
            ._deployEscrowSuper(bidId);
    }

    function test_deployEscrow_internal_proxy_already_deployed() public {
        uint256 bidId = 0;

        tellerV2Mock.setBorrower(address(borrower));

        //use the mock escrow imp
        collateralManager.setGlobalEscrowProxyAddress(
            address(escrowImplementation)
        );

        (address proxy, address borrower) = collateralManager
            ._deployEscrowSuper(bidId);
    }

    function test_isBidCollateralBacked_empty() public {
        uint256 bidId = 0;

        bool collateralBacked = collateralManager.isBidCollateralBackedSuper(
            bidId
        );

        assertTrue(
            collateralBacked == false,
            "collateral backed is not correct"
        );
    }

    function test_isBidCollateralBacked_populated() public {
        uint256 bidId = 0;

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        collateralManager.commitCollateralSuper(bidId, collateral);

        bool collateralBacked = collateralManager.isBidCollateralBackedSuper(
            bidId
        );

        assertTrue(collateralBacked, "collateral backed is not correct");
    }

    function test_onERC721Received() public {
        bytes4 response = collateralManager.onERC721Received(
            address(this),
            address(borrower),
            0,
            ""
        );

        assertEq(
            response,
            bytes4(
                keccak256("onERC721Received(address,address,uint256,bytes)")
            ),
            "response is not correct"
        );
    }

    function test_onERC1155Received() public {
        bytes4 response = collateralManager.onERC1155Received(
            address(this),
            address(borrower),
            0,
            0,
            ""
        );

        assertEq(
            response,
            bytes4(
                keccak256(
                    "onERC1155Received(address,address,uint256,uint256,bytes)"
                )
            ),
            "response is not correct"
        );
    }

    function test_onERC1155BatchReceived() public {
        bytes4 response = collateralManager.onERC1155BatchReceived(
            address(this),
            address(borrower),
            new uint256[](1),
            new uint256[](1),
            ""
        );

        assertEq(
            response,
            bytes4(
                keccak256(
                    "onERC1155BatchReceived(address,address,uint256[],uint256[],bytes)"
                )
            ),
            "response is not correct"
        );
    }

    function test_onERC1155BatchReceived_invalid_batch_length() public {
        vm.expectRevert(
            "Only allowed one asset batch transfer per transaction."
        );
        bytes4 response = collateralManager.onERC1155BatchReceived(
            address(this),
            address(borrower),
            new uint256[](2),
            new uint256[](2),
            ""
        );
    }

    function test_checkBalances_empty() public {
        Collateral[] memory collateralArray;

        (bool valid, bool[] memory checks) = collateralManager.checkBalances(
            address(borrower),
            collateralArray
        );

        assertTrue(valid);
    }



    function test_checkBalances_has_partial() public {
        Collateral[] memory collateralArray = new Collateral[](2);


         wethMock.transfer(address(borrower), 1000000);

         uint256 borrowerBalance = otherErc20Mock.balanceOf(address(borrower));
         assertEq( borrowerBalance, 0 ) ;

            //does not have this 
         collateralArray[0] = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(otherErc20Mock)
        });

        //does have this 
         collateralArray[1] = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });


        (bool valid, bool[] memory checks) = collateralManager._checkBalancesSuper(
            address(borrower),
            collateralArray,
            false
        );  

        // assertFalse(checks[0]);
       // assertTrue(checks[1]);

        //check balances should return false (even with short circuit = false)
        assertFalse(valid);
    }


    function test_checkBalances_has_both() public {
        Collateral[] memory collateralArray = new Collateral[](2);


         wethMock.transfer(address(borrower), 1000000);
          otherErc20Mock.transfer(address(borrower), 1000000);

            //does have this 
         collateralArray[0] = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(otherErc20Mock)
        });

        //does have this 
         collateralArray[1] = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });


        (bool valid, bool[] memory checks) = collateralManager._checkBalancesSuper(
            address(borrower),
            collateralArray,
            false
        );

        //check balances should return true (  with short circuit = false)
        assertTrue(valid);
    }

   /* function test_checkBalances_public() public {
        Collateral[] memory collateralArray = new Collateral[](1);

        collateralArray[0] = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        (bool valid, bool[] memory checks) = collateralManager.checkBalances(
            address(borrower),
            collateralArray
        );

        assertTrue(
            collateralManager.checkBalancesWasCalled(),
            "Check balances was not called"
        );
    }

      function test_checkBalance_internal_invalid_type() public {
    
        wethMock.transfer(address(borrower), 1000);

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType(uint16(4)),
            _amount: 1000,
            _tokenId: 0, 
            _collateralAddress: address(wethMock)
        });
 
 
        bool valid = collateralManager._checkBalanceSuper(
            address(borrower),
            collateral
        );
 
         assertFalse(valid, "Check balance should be invalid");
     }*/

    function test_checkBalance_internal_erc20() public {
        wethMock.transfer(address(borrower), 1000);

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        bool valid = collateralManager._checkBalanceSuper(
            address(borrower),
            collateral
        );

        assertTrue(valid, "Check balance should be valid");
    }

    function test_checkBalance_internal_erc20_invalid() public {
        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        bool valid = collateralManager._checkBalanceSuper(
            address(borrower),
            collateral
        );

        assertFalse(valid, "Check balance should be invalid");
    }

    function test_checkBalance_internal_erc721() public {
        erc721Mock.mint(address(borrower));

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC721,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(erc721Mock)
        });

        bool valid = collateralManager._checkBalanceSuper(
            address(borrower),
            collateral
        );

        assertEq(valid, true, "Check balance should be valid");
    }

    function test_checkBalance_internal_erc721_invalid() public {
        erc721Mock.mint(address(this));

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC721,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(erc721Mock)
        });

        bool valid = collateralManager._checkBalanceSuper(
            address(borrower),
            collateral
        );

        assertEq(valid, false, "Check balance should be invalid");
    }

    function test_checkBalance_internal_erc1155() public {
        erc1155Mock.mint(address(borrower), 5);

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC1155,
            _amount: 5,
            _tokenId: 0,
            _collateralAddress: address(erc1155Mock)
        });

        bool valid = collateralManager._checkBalanceSuper(
            address(borrower),
            collateral
        );

        assertTrue(valid, "Check balance should be valid");
    }

    function test_checkBalance_internal_erc1155_invalid() public {
        //erc1155Mock.mint(address(this), 5);

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC1155,
            _amount: 5,
            _tokenId: 0,
            _collateralAddress: address(erc1155Mock)
        });

        bool valid = collateralManager._checkBalanceSuper(
            address(borrower),
            collateral
        );

        assertFalse(valid, "Check balance should be invalid");
    }

    function test_checkBalances_internal_short_circuit_valid() public {
        bool shortCircuit = true;


         wethMock.transfer(address(borrower), 1000000);

        Collateral[] memory collateralArray = new Collateral[](2);
        collateralArray[0] = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });
        collateralArray[1] = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 222,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        collateralManager.setCheckBalanceGlobalValid(true);

        (bool valid, bool[] memory checks) = collateralManager
            ._checkBalancesSuper(
                address(borrower),
                collateralArray,
                shortCircuit
            );

        assertTrue(valid, "Check balance should be valid");
        assertEq(checks.length, 2, "Checks length should be 2");
    }

    function test_checkBalances_internal_short_circuit_invalid() public {
        bool shortCircuit = true;

        Collateral[] memory collateralArray = new Collateral[](2);
        collateralArray[0] = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });
        collateralArray[1] = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 222,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

      //  collateralManager.setCheckBalanceGlobalValid(false);

        (bool valid, bool[] memory checks) = collateralManager
            ._checkBalancesSuper(
                address(borrower),
                collateralArray,
                shortCircuit
            );

        assertFalse(valid, "Check balance should be invalid");
        assertEq(checks.length, 2, "Checks length should be 2");
    }

    function test_checkBalances_internal_valid() public {


         wethMock.transfer(address(borrower), 1000000);

        Collateral[] memory collateralArray = new Collateral[](2);

        collateralArray[0] = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });
        collateralArray[1] = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 222,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        bool shortCircuit = false;

      //  collateralManager.setCheckBalanceGlobalValid(true);

        (bool valid, bool[] memory checks) = collateralManager
            ._checkBalancesSuper(
                address(borrower),
                collateralArray,
                shortCircuit
            );

        assertTrue(valid, "Check balance should be valid");
        assertEq(checks.length, 2, "Checks length should be 2");
    }

    function test_checkBalances_internal_invalid() public {
        Collateral[] memory collateralArray = new Collateral[](1);
        collateralArray[0] = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        bool shortCircuit = false;

        collateralManager.setCheckBalanceGlobalValid(false);

        (bool valid, bool[] memory checks) = collateralManager
            ._checkBalancesSuper(
                address(borrower),
                collateralArray,
                shortCircuit
            );

        assertFalse(valid, "Check balance should be invalid");
    }

    /*function test_checkBalance_internal_none() public {

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType(uint16(5)),
            _amount: 1000,
            _tokenId: 0, 
            _collateralAddress: address(wethMock)
        }); 
 
        bool valid = collateralManager._checkBalanceSuper(
            address(borrower),
            collateral
        );
 
         assertFalse(valid, "Check balance should be false");

      }*/

    function test_checkBalance_internal_insufficient_assets() public {
        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        bool valid = collateralManager._checkBalanceSuper(
            address(borrower),
            collateral
        );

        assertFalse(valid, "check balance super should be invalid");
    }

    function test_checkBalance_internal_sufficient_assets() public {
        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        wethMock.transfer(address(borrower), 1000);

        bool valid = collateralManager._checkBalanceSuper(
            address(borrower),
            collateral
        );

        //need to inject state

        assertTrue(valid, "check balance super not valid");
    }

    function test_revalidateCollateral_valid() public {
        Collateral[] memory collateralArray;

        uint256 bidId = 0;

        bool valid = collateralManager.revalidateCollateral(bidId);

        assertTrue(valid);
    }

    function test_revalidateCollateral_invalid() public {
       

        uint256 bidId = 0;


         Collateral[] memory collateralArray = new Collateral[](1);

        collateralArray[0] = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1e32,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });


         collateralManager.forceSetBidCollateral(bidId ,collateralArray );

        bool valid = collateralManager.revalidateCollateral(bidId);

        assertFalse(valid, "collateral should be invalid");
    }

    function test_commit_collateral_single() public {
        uint256 bidId = 0;

        wethMock.transfer(address(borrower), 1000000);



        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        tellerV2Mock.setBorrower(address(borrower));

        collateralManager.setCheckBalanceGlobalValid(true);
        vm.prank(address(tellerV2Mock));
        collateralManager.commitCollateral(bidId, collateral);

        assertTrue(
            collateralManager.commitCollateralInternalWasCalled(),
            "commit collateral was not called"
        );
    }

    function test_commit_collateral_single_invalid_bid() public {
        uint256 bidId = 0;

           wethMock.transfer(address(borrower), 1000000);



        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        collateralManager.setCheckBalanceGlobalValid(true);
        vm.prank(address(tellerV2Mock));
        vm.expectRevert("Loan has no borrower");
        collateralManager.commitCollateral(bidId, collateral);
    }

    function test_commit_collateral_single_not_teller() public {
        uint256 bidId = 0;

           wethMock.transfer(address(borrower), 1000000);



        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        tellerV2Mock.setBorrower(address(borrower));

        collateralManager.setCheckBalanceGlobalValid(true);
        vm.expectRevert("Sender not authorized");
        collateralManager.commitCollateral(bidId, collateral);
    }

    function test_commit_collateral_single_invalid() public {
        uint256 bidId = 0;

        Collateral memory collateral = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        tellerV2Mock.setBorrower(address(borrower));

        collateralManager.setCheckBalanceGlobalValid(false);
        vm.prank(address(tellerV2Mock));
        collateralManager.commitCollateral(bidId, collateral);

        assertFalse(
            collateralManager.commitCollateralInternalWasCalled(),
            "commit collateral was not called"
        );
    }

    function test_commit_collateral_array() public {
        uint256 bidId = 0;

        Collateral[] memory collateralArray = new Collateral[](1);


         wethMock.transfer(address(borrower), 1000000);



        collateralArray[0] = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        tellerV2Mock.setBorrower(address(borrower));

      //  collateralManager.setCheckBalanceGlobalValid(true);
        vm.prank(address(tellerV2Mock));
        collateralManager.commitCollateral(bidId, collateralArray);

        assertTrue(
            collateralManager.commitCollateralInternalWasCalled(),
            "commit collateral was not called"
        );
    }

    function test_commit_collateral_invalid_bid() public {
        uint256 bidId = 0;

        Collateral[] memory collateralArray = new Collateral[](1);

        collateralArray[0] = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        tellerV2Mock.setBorrower(address(0));

        collateralManager.setCheckBalanceGlobalValid(true);
        vm.prank(address(tellerV2Mock));
        vm.expectRevert("Loan has no borrower");
        collateralManager.commitCollateral(bidId, collateralArray);
    }

    function test_commit_collateral_array_empty() public {
        uint256 bidId = 0;

        tellerV2Mock.setBorrower(address(borrower));

        Collateral[] memory collateralArray = new Collateral[](0);

        collateralManager.setCheckBalanceGlobalValid(true);
        vm.prank(address(tellerV2Mock));
        collateralManager.commitCollateral(bidId, collateralArray);

        assertFalse(
            collateralManager.commitCollateralInternalWasCalled(),
            "commit collateral was not called"
        );
    }

    function test_commit_collateral_array_invalid() public {
        uint256 bidId = 0;

        Collateral[] memory collateralArray = new Collateral[](1);

        collateralArray[0] = Collateral({
            _collateralType: CollateralType.ERC20,
            _amount: 1000,
            _tokenId: 0,
            _collateralAddress: address(wethMock)
        });

        collateralManager.setCheckBalanceGlobalValid(false);

        tellerV2Mock.setBorrower(address(borrower));

        vm.prank(address(tellerV2Mock));
        collateralManager.commitCollateral(bidId, collateralArray);

        assertFalse(
            collateralManager.commitCollateralInternalWasCalled(),
            "commit collateral should not have been called"
        );
    }

    function test_commit_collateral_array_empty_invalid() public {
        uint256 bidId = 0;

        Collateral[] memory collateralArray = new Collateral[](0);

        tellerV2Mock.setBorrower(address(borrower));

        collateralManager.setCheckBalanceGlobalValid(false);
        vm.prank(address(tellerV2Mock));
        collateralManager.commitCollateral(bidId, collateralArray);

        assertFalse(
            collateralManager.commitCollateralInternalWasCalled(),
            "commit collateral should not have been called"
        );
    }
}

contract User {
    constructor() {}

    function approveERC20(address tokenAddress, address to, uint256 amount)
        public
    {
        ERC20(tokenAddress).approve(address(to), amount);
    }

    receive() external payable {}

    //receive 721
    function onERC721Received(address, address, uint256, bytes calldata)
        external
        returns (bytes4)
    {
        return this.onERC721Received.selector;
    }

    //receive 1155
    function onERC1155Received(
        address,
        address,
        uint256,
        uint256,
        bytes calldata
    ) external returns (bytes4) {
        return this.onERC1155Received.selector;
    }
}

contract CollateralEscrowV1_Mock is CollateralEscrowV1 {
    bool public depositAssetWasCalled;
    bool public withdrawWasCalled;

    constructor() CollateralEscrowV1() {}

    function depositAsset(
        CollateralType _collateralType,
        address _collateralAddress,
        uint256 _amount,
        uint256 _tokenId
    ) external payable override {
        depositAssetWasCalled = true;
    }

    function withdraw(
        address _collateralAddress,
        uint256 _amount,
        address _recipient
    ) external override {
        withdrawWasCalled = true;
    }
}

contract TellerV2_Mock is TellerV2SolMock {
    address public globalBorrower;
    address public globalLender;
    bool public bidsDefaultedGlobally;
    BidState public globalBidState;

    constructor() TellerV2SolMock() {}

    function setBorrower(address borrower) public {
        globalBorrower = borrower;
    }

    function setLender(address lender) public {
        globalLender = lender;
    }

    function getLoanBorrower(uint256 bidId)
        public
        view
        override
        returns (address)
    {
        return address(globalBorrower);
    }

    function getLoanLender(uint256 bidId)
        public
        view
        override
        returns (address)
    {
        return address(globalLender);
    }

    function isLoanDefaulted(uint256 _bidId)
        public
        view
        override
        returns (bool)
    {
        return bidsDefaultedGlobally;
    }

    function getBidState(uint256 _bidId)
        public
        view
        override
        returns (BidState)
    {
        return globalBidState;
    }

    function setGlobalBidState(BidState _state) public {
        globalBidState = _state;
    }

    function setBidsDefaultedGlobally(bool _defaulted) public {
        bidsDefaultedGlobally = _defaulted;
    }
}
