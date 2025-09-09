pragma solidity ^0.8.0;

import {Test, console} from "forge-std/Test.sol";
import {VotingEscrow} from "@aerodrome/VotingEscrow.sol";
import {auctionFactoryDebita, DutchAuction_veNFT} from "@contracts/auctions/AuctionFactory.sol";

// DutchAuction_veNFT
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
// import ERC20
import {ERC20Mock} from "@openzeppelin/contracts/mocks/token/ERC20Mock.sol";
import {DebitaV3Aggregator} from "@contracts/DebitaV3Aggregator.sol";
import "@openzeppelin/contracts/token/ERC721/utils/ERC721Holder.sol";

contract Auction is Test {
    VotingEscrow public ABIERC721Contract;
    auctionFactoryDebita public factory;
    DutchAuction_veNFT public auction;
    DebitaV3Aggregator public DebitaV3AggregatorContract;

    address signer = 0x5F35576Ae82553209224d85Bbe9657565ab16a4f;
    address secondSigner = 0x81B2c95353d69580875a7aFF5E8f018F1761b7D1;
    address buyer = address(0x02);
    address veAERO = 0xeBf418Fe2512e7E6bd9b87a8F0f294aCDC67e6B4;
    address AERO = 0x940181a94A35A4569E4529A3CDfB74e38FD98631;
    address USDC = 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913;

    function setUp() public {
        deal(AERO, signer, 100e18, false);
        deal(AERO, buyer, 100e18, false);
        factory = new auctionFactoryDebita();
        ABIERC721Contract = VotingEscrow(veAERO);

        DebitaV3AggregatorContract = new DebitaV3Aggregator(
            address(0x0),
            address(0x0),
            address(0x0),
            address(0x0),
            address(0x0),
            address(0x0)
        );
        factory.setAggregator(address(DebitaV3AggregatorContract));
        vm.startPrank(signer);

        ERC20Mock(AERO).approve(address(ABIERC721Contract), 1000e18);
        uint id = ABIERC721Contract.createLock(100e18, 365 * 4 * 86400);
        ABIERC721Contract.approve(address(factory), id);
        address _auction = factory.createAuction(
            id,
            veAERO,
            AERO,
            100e18,
            10e18,
            86400
        );
        auction = DutchAuction_veNFT(_auction);
        vm.stopPrank();
    }

    function testGetInfo() public {
        vm.startPrank(buyer);
        ERC20Mock(AERO).approve(address(auction), 100e18);
        uint balanceBefore = ERC20Mock(AERO).balanceOf(signer);
        uint balanceBeforeThisAddress = ERC20Mock(AERO).balanceOf(
            address(this)
        );
        uint paymentAmount = auction.getCurrentPrice();
        auction.buyNFT();
        uint balanceAfter = ERC20Mock(AERO).balanceOf(signer);
        uint balanceAfterThisAddress = ERC20Mock(AERO).balanceOf(address(this));
        vm.stopPrank();
        uint publicFee = factory.publicAuctionFee();
        uint fee = (paymentAmount * publicFee) / 10000;
        console.logUint(paymentAmount);
        assertEq(balanceBefore + paymentAmount - fee, balanceAfter);
        assertEq(balanceBeforeThisAddress + fee, balanceAfterThisAddress);
    }

    function testRandomBuyDuringAuction() public {
        vm.startPrank(buyer);
        ERC20Mock(AERO).approve(address(auction), 100e18);
        uint balanceBefore = ERC20Mock(AERO).balanceOf(signer);
        uint balanceBeforeThisAddress = ERC20Mock(AERO).balanceOf(
            address(this)
        );
        vm.warp(block.timestamp + 43200);
        uint paymentAmount = auction.getCurrentPrice();
        auction.buyNFT();
        uint balanceAfter = ERC20Mock(AERO).balanceOf(signer);
        uint balanceAfterThisAddress = ERC20Mock(AERO).balanceOf(address(this));
        vm.stopPrank();

        uint publicFee = factory.publicAuctionFee();
        uint fee = (paymentAmount * publicFee) / 10000;
        // get auction info
        DutchAuction_veNFT.dutchAuction_INFO memory m_currentAuction = auction
            .getAuctionData();
        assertEq(balanceBefore + paymentAmount - fee, balanceAfter);
        assertEq(balanceBeforeThisAddress + fee, balanceAfterThisAddress);
        assertEq(m_currentAuction.isActive, false);
        assertEq(m_currentAuction.isLiquidation, false);
        assertEq(paymentAmount > 55e18 && paymentAmount < 551e17, true);
    }

    function testFloorPrice() public {
        vm.startPrank(buyer);
        ERC20Mock(AERO).approve(address(auction), 100e18);
        uint balanceBefore = ERC20Mock(AERO).balanceOf(signer);
        uint balanceBeforeThisAddress = ERC20Mock(AERO).balanceOf(
            address(this)
        );
        vm.warp(block.timestamp + 86401);
        uint paymentAmount = auction.getCurrentPrice();
        auction.buyNFT();
        uint balanceAfter = ERC20Mock(AERO).balanceOf(signer);
        uint balanceAfterThisAddress = ERC20Mock(AERO).balanceOf(address(this));
        vm.stopPrank();

        uint publicFee = factory.publicAuctionFee();
        uint fee = (paymentAmount * publicFee) / 10000;
        // get auction info
        DutchAuction_veNFT.dutchAuction_INFO memory m_currentAuction = auction
            .getAuctionData();

        // get owner of veNFT
        address owner = ABIERC721Contract.ownerOf(
            m_currentAuction.nftCollateralID
        );

        assertEq(owner, buyer);
        assertEq(balanceBefore + paymentAmount - fee, balanceAfter);
        assertEq(balanceBeforeThisAddress + fee, balanceAfterThisAddress);
        assertEq(m_currentAuction.isActive, false);
        assertEq(m_currentAuction.isLiquidation, false);
        assertEq(paymentAmount, 10e18);
    }

    function testEditFloorPriceDuringAuction() public {
        // get current price
        uint currentPrice = auction.getCurrentPrice();
        // edit floor price
        vm.startPrank(signer);
        auction.editFloorPrice(5e18);
        vm.stopPrank();
        // get current price
        uint newPrice = auction.getCurrentPrice();

        assertEq(currentPrice, newPrice);
    }

    function testCancelAuction() public {
        DutchAuction_veNFT.dutchAuction_INFO[] memory auctionsBefore = factory
            .getActiveAuctionOrders(0, 100);
        vm.startPrank(signer);
        auction.cancelAuction();
        vm.stopPrank();
        DutchAuction_veNFT.dutchAuction_INFO memory m_currentAuction = auction
            .getAuctionData();
        DutchAuction_veNFT.dutchAuction_INFO[] memory auctionsAfter = factory
            .getActiveAuctionOrders(0, 100);
        address owner = ABIERC721Contract.ownerOf(
            m_currentAuction.nftCollateralID
        );

        assertEq(auctionsBefore.length, 1);
        assertEq(auctionsBefore[0].isActive, true);
        assertEq(auctionsBefore[0].initAmount, 100e18);
        assertEq(auctionsAfter.length, 0);
        assertEq(owner, signer);
        assertEq(m_currentAuction.isActive, false);
    }

    function testEditFloorPriceAfterReachedIt() public {
        vm.startPrank(signer);
        vm.warp(block.timestamp + 8640000);
        auction.editFloorPrice(5e18);
        // get auction info
        DutchAuction_veNFT.dutchAuction_INFO memory m_currentAuction = auction
            .getAuctionData();
        vm.stopPrank();
        uint currentPrice = auction.getCurrentPrice();

        vm.warp(block.timestamp + 86400);
        uint newPrice = auction.getCurrentPrice();
        uint expectedDuration = (100e18 - 5e18) / m_currentAuction.tickPerBlock;
        assertEq(
            m_currentAuction.endBlock,
            m_currentAuction.initialBlock + expectedDuration
        );
        assertEq(newPrice, 5e18);
        assertEq(currentPrice > 10e18 && currentPrice < 101e17, true);
    }

    function testReadMultipleAuctions() public {
        DutchAuction_veNFT secondAuction;
        DutchAuction_veNFT thirdAuction;
        vm.startPrank(secondSigner);
        secondAuction = DutchAuction_veNFT(
            createAuctionInternal(300e18, 15e18, 106400)
        );

        thirdAuction = DutchAuction_veNFT(
            createAuctionInternal(200e18, 10e18, 106400)
        );

        DutchAuction_veNFT.dutchAuction_INFO[] memory auctions = factory
            .getActiveAuctionOrders(0, 100);

        assertEq(auctions.length, 3);
        assertEq(auctions[0].initAmount, 100e18);
        assertEq(auctions[1].initAmount, 300e18);
        assertEq(auctions[2].initAmount, 200e18);

        assertEq(auctions[0].floorAmount, 10e18);
        assertEq(auctions[1].floorAmount, 15e18);
        assertEq(auctions[2].floorAmount, 10e18);

        secondAuction.cancelAuction();

        auctions = factory.getActiveAuctionOrders(0, 100);

        assertEq(auctions.length, 2);
        assertEq(auctions[0].initAmount, 100e18);
        assertEq(auctions[1].initAmount, 200e18);

        assertEq(auctions[0].floorAmount, 10e18);
        assertEq(auctions[1].floorAmount, 10e18);

        secondAuction = DutchAuction_veNFT(
            createAuctionInternal(300e18, 15e18, 106400)
        );

        auctions = factory.getActiveAuctionOrders(0, 100);

        assertEq(auctions.length, 3);
        assertEq(auctions[0].initAmount, 100e18);
        assertEq(auctions[1].initAmount, 200e18);
        assertEq(auctions[2].initAmount, 300e18);

        secondAuction.cancelAuction();
        DutchAuction_veNFT.dutchAuction_INFO[] memory historical = factory
            .getHistoricalAuctions(0, 100);

        assertEq(historical.length, 4);

        vm.stopPrank();
    }

    function testUSDCAuctionWith6Decimals() public {
        deal(USDC, signer, 100e6, false);
        deal(AERO, signer, 100e18, false);

        deal(USDC, buyer, 100e6, false);
        vm.startPrank(signer);
        ERC20Mock(AERO).approve(address(ABIERC721Contract), 100e18);
        uint id = ABIERC721Contract.createLock(100e18, 365 * 4 * 86400);
        ABIERC721Contract.approve(address(factory), id);
        address _auction = factory.createAuction(
            id,
            veAERO,
            USDC,
            100e6,
            10e6,
            86400
        );
        DutchAuction_veNFT usdcAuction = DutchAuction_veNFT(_auction);
        vm.stopPrank();
        vm.warp(block.timestamp + 86401);
        vm.startPrank(buyer);
        ERC20Mock(USDC).approve(address(usdcAuction), 100e6);
        uint balanceBefore = ERC20Mock(USDC).balanceOf(signer);
        uint balanceBeforeThisAddress = ERC20Mock(USDC).balanceOf(
            address(this)
        );
        uint paymentAmount = usdcAuction.getCurrentPrice();
        usdcAuction.buyNFT();
        uint balanceAfter = ERC20Mock(USDC).balanceOf(signer);
        uint balanceAfterThisAddress = ERC20Mock(USDC).balanceOf(address(this));
        vm.stopPrank();
        uint publicFee = factory.publicAuctionFee();
        uint fee = (paymentAmount * publicFee) / 10000;
        assertEq(paymentAmount, 10e6);
        assertEq(balanceBefore + paymentAmount - fee, balanceAfter);
        assertEq(balanceBeforeThisAddress + fee, balanceAfterThisAddress);
    }

    function testEditFloorUSDCAndBuyIt() public {
        deal(USDC, signer, 100e6, false);
        deal(AERO, signer, 100e18, false);

        deal(USDC, buyer, 100e6, false);
        vm.startPrank(signer);
        ERC20Mock(AERO).approve(address(ABIERC721Contract), 100e18);
        uint id = ABIERC721Contract.createLock(100e18, 365 * 4 * 86400);
        ABIERC721Contract.approve(address(factory), id);
        address _auction = factory.createAuction(
            id,
            veAERO,
            USDC,
            100e6,
            10e6,
            86400
        );
        DutchAuction_veNFT usdcAuction = DutchAuction_veNFT(_auction);
        vm.stopPrank();
        vm.warp(block.timestamp + 86401);
        vm.startPrank(signer);
        usdcAuction.editFloorPrice(5e6);
        vm.stopPrank();

        vm.warp(block.timestamp + 106401);
        vm.startPrank(buyer);
        ERC20Mock(USDC).approve(address(usdcAuction), 100e6);
        uint balanceBefore = ERC20Mock(USDC).balanceOf(signer);
        uint balanceBeforeThisAddress = ERC20Mock(USDC).balanceOf(
            address(this)
        );
        uint paymentAmount = usdcAuction.getCurrentPrice();
        usdcAuction.buyNFT();
        uint balanceAfter = ERC20Mock(USDC).balanceOf(signer);
        uint balanceAfterThisAddress = ERC20Mock(USDC).balanceOf(address(this));
        vm.stopPrank();
        uint publicFee = factory.publicAuctionFee();
        uint fee = (paymentAmount * publicFee) / 10000;

        assertEq(paymentAmount, 5e6);
        assertEq(balanceBefore + paymentAmount - fee, balanceAfter);
        assertEq(balanceBeforeThisAddress + fee, balanceAfterThisAddress);
    }

    function buyAuctionWithUSDCAfterHalfOfTheTime() public {
        deal(USDC, signer, 100e6, false);
        deal(AERO, signer, 100e18, false);

        deal(USDC, buyer, 100e6, false);
        vm.startPrank(signer);
        ERC20Mock(AERO).approve(address(ABIERC721Contract), 100e18);
        uint id = ABIERC721Contract.createLock(100e18, 365 * 4 * 86400);
        ABIERC721Contract.approve(address(factory), id);
        address _auction = factory.createAuction(
            id,
            veAERO,
            USDC,
            100e6,
            10e6,
            86400
        );
        DutchAuction_veNFT usdcAuction = DutchAuction_veNFT(_auction);
        vm.stopPrank();
        vm.warp(block.timestamp + 43200);
        vm.startPrank(buyer);
        ERC20Mock(USDC).approve(address(usdcAuction), 100e6);
        uint balanceBefore = ERC20Mock(USDC).balanceOf(signer);
        uint balanceBeforeThisAddress = ERC20Mock(USDC).balanceOf(
            address(this)
        );
        uint paymentAmount = usdcAuction.getCurrentPrice();
        usdcAuction.buyNFT();
        uint balanceAfter = ERC20Mock(USDC).balanceOf(signer);
        uint balanceAfterThisAddress = ERC20Mock(USDC).balanceOf(address(this));
        vm.stopPrank();
        uint publicFee = factory.publicAuctionFee();
        uint fee = (paymentAmount * publicFee) / 10000;
        assertEq(paymentAmount > 55e6 && paymentAmount < 551e15, true);
        assertEq(balanceBefore + paymentAmount - fee, balanceAfter);
        assertEq(balanceBeforeThisAddress + fee, balanceAfterThisAddress);
    }

    function createAuctionInternal(
        uint initAmount,
        uint floorAmount,
        uint timelapse
    ) internal returns (address) {
        deal(AERO, secondSigner, 1000e18, false);
        ERC20Mock(AERO).approve(address(ABIERC721Contract), 1000e18);
        uint id = ABIERC721Contract.createLock(100e18, 365 * 4 * 86400);
        ABIERC721Contract.approve(address(factory), id);
        address _auction = factory.createAuction(
            id,
            veAERO,
            AERO,
            initAmount,
            floorAmount,
            timelapse
        );
        return _auction;
    }
}
