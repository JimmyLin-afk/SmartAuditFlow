pragma solidity ^0.8.0;

import {Test, console} from "forge-std/Test.sol";
import {VotingEscrow} from "@aerodrome/VotingEscrow.sol";
import {BuyOrder, buyOrderFactory} from "@contracts/buyOrders/buyOrderFactory.sol";
// DutchAuction_veNFT
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
// import ERC20
import {ERC20Mock} from "@openzeppelin/contracts/mocks/token/ERC20Mock.sol";
import {veNFTAerodrome} from "@contracts/Non-Fungible-Receipts/veNFTS/Aerodrome/Receipt-veNFT.sol";
import {veNFTVault} from "@contracts/Non-Fungible-Receipts/veNFTS/Aerodrome/veNFTAerodrome.sol";
import "@openzeppelin/contracts/token/ERC721/utils/ERC721Holder.sol";
import {DynamicData} from "../../interfaces/getDynamicData.sol";

contract BuyOrderTest is Test {
    VotingEscrow public ABIERC721Contract;
    buyOrderFactory public factory;
    BuyOrder public buyOrder;
    veNFTAerodrome public receiptContract;
    DynamicData public allDynamicData;
    ERC20Mock public AEROContract;
    BuyOrder public buyOrderContract;

    address signer = 0x5F35576Ae82553209224d85Bbe9657565ab16a4f;
    address seller = 0x81B2c95353d69580875a7aFF5E8f018F1761b7D1;
    address buyer = address(0x02);
    address veAERO = 0xeBf418Fe2512e7E6bd9b87a8F0f294aCDC67e6B4;
    address AERO = 0x940181a94A35A4569E4529A3CDfB74e38FD98631;
    uint receiptID;
    function setUp() public {
        deal(AERO, seller, 100e18, false);
        deal(AERO, buyer, 100e18, false);
        BuyOrder instanceDeployment = new BuyOrder();
        factory = new buyOrderFactory(address(instanceDeployment));
        allDynamicData = new DynamicData();
        AEROContract = ERC20Mock(AERO);

        receiptContract = new veNFTAerodrome(veAERO, AERO);

        ABIERC721Contract = VotingEscrow(veAERO);
        vm.startPrank(seller);
        ERC20Mock(AERO).approve(address(ABIERC721Contract), 1000e18);
        uint veNFTID = ABIERC721Contract.createLock(100e18, 365 * 4 * 86400);

        ABIERC721Contract.approve(address(receiptContract), veNFTID);
        uint[] memory nftID = allDynamicData.getDynamicUintArray(1);
        nftID[0] = veNFTID;
        receiptContract.deposit(nftID);
        receiptID = receiptContract.lastReceiptID();

        vm.stopPrank();

        vm.startPrank(buyer);
        AEROContract.approve(address(factory), 1000e18);
        address _buyOrderAddress = factory.createBuyOrder(
            AERO,
            address(receiptContract),
            100e18,
            7e17
        );
        buyOrderContract = BuyOrder(_buyOrderAddress);

        vm.stopPrank();
    }

    function testValues() public {
        BuyOrder.BuyInfo memory buyInfo = buyOrderContract.getBuyInfo();
        assertEq(buyInfo.buyOrderAddress, address(buyOrderContract));
        assertEq(buyInfo.wantedToken, address(receiptContract));
        assertEq(buyInfo.buyRatio, 7e17);
        assertEq(buyInfo.availableAmount, 100e18);
        assertEq(buyInfo.capturedAmount, 0);
        assertEq(buyInfo.owner, buyer);
        assertEq(buyInfo.buyToken, AERO);
        assertEq(buyInfo.isActive, true);
        vm.prank(buyer);
        buyOrderContract.deleteBuyOrder();
        BuyOrder.BuyInfo memory buyInfo2 = buyOrderContract.getBuyInfo();

        assertEq(buyInfo2.buyOrderAddress, address(buyOrderContract));
        assertEq(buyInfo2.wantedToken, address(receiptContract));
        assertEq(buyInfo2.buyRatio, 7e17);
        assertEq(buyInfo2.availableAmount, 0);
        assertEq(buyInfo2.capturedAmount, 0);
        assertEq(buyInfo2.owner, buyer);
        assertEq(buyInfo2.buyToken, AERO);
        assertEq(buyInfo2.isActive, false);
    }

    function testSellReceipt() public {
        vm.startPrank(seller);
        receiptContract.approve(address(buyOrderContract), receiptID);
        uint balanceBeforeAero = AEROContract.balanceOf(seller);
        buyOrderContract.sellNFT(receiptID);
        uint balanceAfterAero = AEROContract.balanceOf(seller);
        vm.stopPrank();
        BuyOrder.BuyInfo memory buyInfo = buyOrderContract.getBuyInfo();

        uint fee = (70e18 * 50) / 10000;
        assertEq(buyInfo.capturedAmount, 100e18);
        assertEq(buyInfo.availableAmount, 100e18 - 70e18);
        assertEq(balanceBeforeAero + 70e18 - fee, balanceAfterAero);
    }

    function testSellAndThenCancelIT() public {
        vm.startPrank(seller);
        receiptContract.approve(address(buyOrderContract), receiptID);
        buyOrderContract.sellNFT(receiptID);
        vm.stopPrank();

        vm.startPrank(buyer);

        uint balanceBeforeAero = AEROContract.balanceOf(buyer);
        buyOrderContract.deleteBuyOrder();
        uint balanceAfterAero = AEROContract.balanceOf(buyer);
        BuyOrder.BuyInfo memory buyInfo = buyOrderContract.getBuyInfo();
        vm.stopPrank();

        assertEq(buyInfo.availableAmount, 0);
        assertEq(buyInfo.capturedAmount, 100e18);
        assertEq(balanceBeforeAero + 30e18, balanceAfterAero);
    }

    function testCancelItAndTryToUseIt() public {
        vm.startPrank(buyer);
        buyOrderContract.deleteBuyOrder();
        vm.stopPrank();

        vm.startPrank(seller);
        receiptContract.approve(address(buyOrderContract), receiptID);
        vm.expectRevert("Buy order is not active");
        buyOrderContract.sellNFT(receiptID);
        vm.stopPrank();
    }

    function testTryingToSellABiggerAmount() public {
        vm.startPrank(seller);
        deal(AERO, seller, 1000e18, false);
        ERC20Mock(AERO).approve(address(ABIERC721Contract), 1000e18);
        uint veNFTID = ABIERC721Contract.createLock(1000e18, 365 * 4 * 86400);

        ABIERC721Contract.approve(address(receiptContract), veNFTID);
        uint[] memory nftID = allDynamicData.getDynamicUintArray(1);
        nftID[0] = veNFTID;
        receiptContract.deposit(nftID);
        receiptID = receiptContract.lastReceiptID();
        receiptContract.approve(address(buyOrderContract), receiptID);
        vm.expectRevert("Amount exceeds available amount");
        buyOrderContract.sellNFT(receiptID);
        vm.stopPrank();
    }

    function testTryToCancelItTwice() public {
        vm.startPrank(buyer);
        buyOrderContract.deleteBuyOrder();
        vm.stopPrank();

        vm.startPrank(buyer);
        vm.expectRevert("Buy order is not active");
        buyOrderContract.deleteBuyOrder();
        vm.stopPrank();
    }

    function testTryToCancelItAsSeller() public {
        vm.startPrank(seller);
        vm.expectRevert("Only owner");
        buyOrderContract.deleteBuyOrder();
        vm.stopPrank();
    }
}
