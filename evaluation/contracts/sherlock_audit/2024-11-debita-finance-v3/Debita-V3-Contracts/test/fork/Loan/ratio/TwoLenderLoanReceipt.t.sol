pragma solidity ^0.8.0;

import {Test, console} from "forge-std/Test.sol";
import {veNFTEqualizer} from "@contracts/Non-Fungible-Receipts/veNFTS/Equalizer/Receipt-veNFT.sol";

import {veNFTVault} from "@contracts/Non-Fungible-Receipts/veNFTS/Equalizer/veNFTEqualizer.sol";
import {DBOFactory} from "@contracts/DebitaBorrowOffer-Factory.sol";
import {DBOImplementation} from "@contracts/DebitaBorrowOffer-Implementation.sol";
import {DLOFactory} from "@contracts/DebitaLendOfferFactory.sol";
import {DLOImplementation} from "@contracts/DebitaLendOffer-Implementation.sol";
import {DebitaV3Aggregator} from "@contracts/DebitaV3Aggregator.sol";
import {Ownerships} from "@contracts/DebitaLoanOwnerships.sol";
import {auctionFactoryDebita} from "@contracts/auctions/AuctionFactory.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {DynamicData} from "../../../interfaces/getDynamicData.sol";
// import ERC20
import {ERC20Mock} from "@openzeppelin/contracts/mocks/token/ERC20Mock.sol";
import {DebitaV3Loan} from "@contracts/DebitaV3Loan.sol";
import {DebitaIncentives} from "@contracts/DebitaIncentives.sol";
import {VotingEscrow} from "@aerodrome/VotingEscrow.sol";
import {DebitaV3Loan} from "@contracts/DebitaV3Loan.sol";
import {DutchAuction_veNFT} from "@contracts/auctions/Auction.sol";

contract DebitaAggregatorTest is Test, DynamicData {
    VotingEscrow public ABIERC721Contract;
    veNFTEqualizer public receiptContract;
    DBOFactory public DBOFactoryContract;
    DLOFactory public DLOFactoryContract;
    Ownerships public ownershipsContract;
    DebitaIncentives public incentivesContract;
    DebitaV3Aggregator public DebitaV3AggregatorContract;
    auctionFactoryDebita public auctionFactoryDebitaContract;
    DynamicData public allDynamicData;
    DebitaV3Loan public DebitaV3LoanContract;
    ERC20Mock public AEROContract;
    ERC20Mock public USDCContract;
    DLOImplementation public LendOrder;
    DLOImplementation public SecondLendOrder;

    DBOImplementation public BorrowOrder;

    address veAERO = 0xeBf418Fe2512e7E6bd9b87a8F0f294aCDC67e6B4;
    address AERO = 0x940181a94A35A4569E4529A3CDfB74e38FD98631;
    address USDC = 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913;
    address borrower = address(0x02);
    address firstLender = address(this);
    address secondLender = 0x5C235931376b21341fA00d8A606e498e1059eCc0;
    address buyer = 0x5C235931376b21341fA00d8A606e498e1059eCc0;

    address feeAddress = address(this);

    uint receiptID;

    function setUp() public {
        allDynamicData = new DynamicData();
        ownershipsContract = new Ownerships();
        receiptContract = new veNFTEqualizer(veAERO, AERO);
        ABIERC721Contract = VotingEscrow(veAERO);
        incentivesContract = new DebitaIncentives();
        DBOImplementation borrowOrderImplementation = new DBOImplementation();
        DBOFactoryContract = new DBOFactory(address(borrowOrderImplementation));
        DLOImplementation proxyImplementation = new DLOImplementation();
        DLOFactoryContract = new DLOFactory(address(proxyImplementation));
        auctionFactoryDebitaContract = new auctionFactoryDebita();
        AEROContract = ERC20Mock(AERO);
        USDCContract = ERC20Mock(USDC);
        DebitaV3Loan loanInstance = new DebitaV3Loan();
        DebitaV3AggregatorContract = new DebitaV3Aggregator(
            address(DLOFactoryContract),
            address(DBOFactoryContract),
            address(incentivesContract),
            address(ownershipsContract),
            address(auctionFactoryDebitaContract),
            address(loanInstance)
        );

        ownershipsContract.setDebitaContract(
            address(DebitaV3AggregatorContract)
        );
        auctionFactoryDebitaContract.setAggregator(
            address(DebitaV3AggregatorContract)
        );
        DLOFactoryContract.setAggregatorContract(
            address(DebitaV3AggregatorContract)
        );
        DBOFactoryContract.setAggregatorContract(
            address(DebitaV3AggregatorContract)
        );

        incentivesContract.setAggregatorContract(
            address(DebitaV3AggregatorContract)
        );
        DebitaV3AggregatorContract.setValidNFTCollateral(
            address(receiptContract),
            true
        );

        deal(AERO, firstLender, 1000e18, false);
        deal(AERO, secondLender, 1000e18, false);
        deal(AERO, borrower, 1000e18, false);

        vm.startPrank(borrower);
        IERC20(AERO).approve(address(ABIERC721Contract), 100e18);
        uint id = ABIERC721Contract.createLock(10e18, 365 * 4 * 86400);
        ABIERC721Contract.approve(address(receiptContract), id);
        uint[] memory nftID = allDynamicData.getDynamicUintArray(1);
        nftID[0] = id;
        receiptContract.deposit(nftID);

        receiptID = receiptContract.lastReceiptID();

        IERC20(AERO).approve(address(DBOFactoryContract), 100e18);

        bool[] memory oraclesActivated = allDynamicData.getDynamicBoolArray(1);
        uint[] memory ltvs = allDynamicData.getDynamicUintArray(1);
        uint[] memory ratio = allDynamicData.getDynamicUintArray(1);

        address[] memory acceptedPrinciples = allDynamicData
            .getDynamicAddressArray(1);
        address[] memory acceptedCollaterals = allDynamicData
            .getDynamicAddressArray(1);
        address[] memory oraclesPrinciples = allDynamicData
            .getDynamicAddressArray(1);

        ratio[0] = 5e17;
        oraclesPrinciples[0] = address(0x0);
        acceptedPrinciples[0] = AERO;
        acceptedCollaterals[0] = address(receiptContract);
        oraclesActivated[0] = false;
        ltvs[0] = 0;
        receiptContract.approve(address(DBOFactoryContract), receiptID);
        address borrowOrderAddress = DBOFactoryContract.createBorrowOrder(
            oraclesActivated,
            ltvs,
            1400,
            864000,
            acceptedPrinciples,
            address(receiptContract),
            true,
            receiptID,
            oraclesPrinciples,
            ratio,
            address(0x0),
            1
        );
        vm.stopPrank();

        AEROContract.approve(address(DLOFactoryContract), 5e18);
        ratio[0] = 65e16;

        address lendOrderAddress = DLOFactoryContract.createLendOrder(
            false,
            oraclesActivated,
            false,
            ltvs,
            2000,
            8640000,
            86400,
            acceptedCollaterals,
            AERO,
            oraclesPrinciples,
            ratio,
            address(0x0),
            5e18
        );

        vm.startPrank(secondLender);
        AEROContract.approve(address(DLOFactoryContract), 5e18);
        ratio[0] = 4e17;
        address SecondlendOrderAddress = DLOFactoryContract.createLendOrder(
            false,
            oraclesActivated,
            false,
            ltvs,
            500,
            9640000,
            86400,
            acceptedCollaterals,
            AERO,
            oraclesPrinciples,
            ratio,
            address(0x0),
            5e18
        );
        vm.stopPrank();
        LendOrder = DLOImplementation(lendOrderAddress);
        BorrowOrder = DBOImplementation(borrowOrderAddress);
        SecondLendOrder = DLOImplementation(SecondlendOrderAddress);
    }

    function testReceiptLoan() public {
        MatchOffers();
        uint[] memory indexes = allDynamicData.getDynamicUintArray(2);
        indexes[0] = 0;
        indexes[1] = 1;
        vm.startPrank(borrower);
        AEROContract.approve(address(DebitaV3LoanContract), 100e18);
        DebitaV3LoanContract.payDebt(indexes);

        // claim the NFT
        address ownerBefore = receiptContract.ownerOf(receiptID);
        DebitaV3LoanContract.claimCollateralAsBorrower(indexes);
        address ownerAfter = receiptContract.ownerOf(receiptID);
        vm.stopPrank();

        // claim Debt
        uint balanceBefore = AEROContract.balanceOf(firstLender);
        DebitaV3LoanContract.claimDebt(0);
        uint balanceAfter = AEROContract.balanceOf(firstLender);

        vm.startPrank(secondLender);
        uint balanceBeforeSecondLender = AEROContract.balanceOf(secondLender);
        DebitaV3LoanContract.claimDebt(1);
        uint balanceAfterSecondLender = AEROContract.balanceOf(secondLender);
        vm.stopPrank();
        // 1000 is the apr of borrow order
        uint amountPerLender = 25e17;
        uint interestToPayFirstLender = calculateInterest(
            LendOrder.getLendInfo().apr,
            86400,
            amountPerLender
        );

        uint interestToPaySecondLender = calculateInterest(
            SecondLendOrder.getLendInfo().apr,
            86400,
            amountPerLender
        );
        uint fee = (interestToPayFirstLender * 1500) / 10000;
        uint feeSecondLender = (interestToPaySecondLender * 1500) / 10000;

        assertEq(
            balanceBefore + amountPerLender + interestToPayFirstLender - fee,
            balanceAfter
        );
        // second lender
        assertEq(
            balanceBeforeSecondLender +
                amountPerLender +
                interestToPaySecondLender -
                feeSecondLender,
            balanceAfterSecondLender
        );
        assertEq(ownerBefore, address(DebitaV3LoanContract));
        assertEq(ownerAfter, borrower);
    }

    function testDefaultAndAuctionCall() public {
        MatchOffers();

        vm.warp(block.timestamp + 8640010);
        DebitaV3LoanContract.createAuctionForCollateral(0);
        DutchAuction_veNFT auction = DutchAuction_veNFT(
            DebitaV3LoanContract.getAuctionData().auctionAddress
        );
        DutchAuction_veNFT.dutchAuction_INFO memory auctionData = auction
            .getAuctionData();

        vm.warp(block.timestamp + (86400 * 10) + 1);

        deal(AERO, buyer, 100e18);
        vm.startPrank(buyer);

        AEROContract.approve(address(auction), 100e18);
        uint balanceBeforeFeeAddress = AEROContract.balanceOf(feeAddress);
        auction.buyNFT();
        uint balanceAfterFeeAddress = AEROContract.balanceOf(feeAddress);
        vm.stopPrank();
        uint expectedFee = (15e17 * 200) / 10000;
        address ownerOfNFT = receiptContract.ownerOf(receiptID);
        DebitaV3Loan.AuctionData memory _auctionData = DebitaV3LoanContract
            .getAuctionData();

        // claim sold Amount
        uint balanceBefore = AEROContract.balanceOf(firstLender);
        DebitaV3LoanContract.claimCollateralAsLender(0);
        uint balanceAfter = AEROContract.balanceOf(firstLender);
        uint soldAmount = 15e17;
        DebitaV3Loan.LoanData memory _loanData = DebitaV3LoanContract
            .getLoanData();
        uint decimalsValuableAsset = 18;

        // use ratio from first lend order --> multiply it with the

        uint collateralUsed = _loanData._acceptedOffers[0].collateralUsed;
        uint payment = (_auctionData.tokenPerCollateralUsed * collateralUsed) /
            (10 ** decimalsValuableAsset);
        vm.warp(block.timestamp + 8640000);
        uint balanceBeforeSecondLender = AEROContract.balanceOf(secondLender);
        vm.prank(secondLender);
        DebitaV3LoanContract.claimCollateralAsLender(1);
        uint balanceAfterSecondLender = AEROContract.balanceOf(secondLender);

        uint CollateralUsedSecondLender = _loanData
            ._acceptedOffers[1]
            .collateralUsed;
        uint paymentSecondLender = (_auctionData.tokenPerCollateralUsed *
            CollateralUsedSecondLender) / (10 ** decimalsValuableAsset);

        uint balanceContract = AEROContract.balanceOf(
            address(DebitaV3LoanContract)
        );
        assertEq(balanceBefore + payment, balanceAfter);
        assertEq(
            balanceBeforeSecondLender + paymentSecondLender,
            balanceAfterSecondLender
        );
        assertEq(
            balanceBeforeFeeAddress + expectedFee,
            balanceAfterFeeAddress,
            "Fee address balance incorrect"
        );
        assertEq(ownerOfNFT, buyer);
        assertEq(auctionData.initAmount, 10e18);
        assertEq(auctionData.isLiquidation, true);
        assertEq(auctionData.sellingToken, AERO);
    }

    function testPartialDefaultAndAuctionCall() public {
        MatchOffers();
        uint[] memory indexes = allDynamicData.getDynamicUintArray(1);
        indexes[0] = 0;
        vm.startPrank(borrower);
        AEROContract.approve(address(DebitaV3LoanContract), 100e18);
        DebitaV3LoanContract.payDebt(indexes);
        vm.stopPrank();

        vm.warp(block.timestamp + 9640010);
        vm.prank(borrower);
        DebitaV3LoanContract.createAuctionForCollateral(0);
        DutchAuction_veNFT auction = DutchAuction_veNFT(
            DebitaV3LoanContract.getAuctionData().auctionAddress
        );
        DutchAuction_veNFT.dutchAuction_INFO memory auctionData = auction
            .getAuctionData();
        vm.startPrank(buyer);
        deal(AERO, buyer, 1000e18);
        AEROContract.approve(address(auction), 100e18);
        auction.buyNFT();
        vm.stopPrank();

        vm.startPrank(firstLender);
        vm.expectRevert();
        DebitaV3LoanContract.claimCollateralAsLender(0);
        vm.stopPrank();

        vm.startPrank(borrower);
        uint[] memory hackIndexes = allDynamicData.getDynamicUintArray(2);
        hackIndexes[0] = 0;
        hackIndexes[1] = 0;
        vm.expectRevert("Already executed");
        DebitaV3LoanContract.claimCollateralAsBorrower(hackIndexes);
        vm.stopPrank();

        vm.startPrank(borrower);
        uint balanceBefore_Borrower = AEROContract.balanceOf(borrower);
        DebitaV3LoanContract.claimCollateralAsBorrower(indexes);
        uint balanceAfter_Borrower = AEROContract.balanceOf(borrower);
        vm.stopPrank();

        vm.startPrank(secondLender);
        uint balanceBefore = AEROContract.balanceOf(secondLender);
        DebitaV3LoanContract.claimCollateralAsLender(1);
        uint balanceAfter = AEROContract.balanceOf(secondLender);
        vm.stopPrank();
    }

    function testInteractWithReceiptWhileLoan() public {
        MatchOffers();
        DebitaV3Loan.LoanData memory _loanData = DebitaV3LoanContract
            .getLoanData();
        veNFTEqualizer.receiptInstance memory receiptData = receiptContract
            .getDataByReceipt(_loanData.NftID);
        address[] memory vaults = allDynamicData.getDynamicAddressArray(1);
        vaults[0] = receiptData.vault;
        vm.prank(borrower);
        receiptContract.resetMultiple(vaults);
    }

    function MatchOffers() internal {
        address[] memory lendOrders = allDynamicData.getDynamicAddressArray(2);
        uint[] memory lendAmountPerOrder = allDynamicData.getDynamicUintArray(
            2
        );
        uint[] memory porcentageOfRatioPerLendOrder = allDynamicData
            .getDynamicUintArray(2);
        address[] memory principles = allDynamicData.getDynamicAddressArray(1);
        uint[] memory indexForPrinciple_BorrowOrder = allDynamicData
            .getDynamicUintArray(2);
        uint[] memory indexForCollateral_LendOrder = allDynamicData
            .getDynamicUintArray(2);
        uint[] memory indexPrinciple_LendOrder = allDynamicData
            .getDynamicUintArray(2);

        lendOrders[0] = address(LendOrder);
        lendAmountPerOrder[0] = 25e17;
        porcentageOfRatioPerLendOrder[0] = 10000;
        principles[0] = AERO;

        lendOrders[1] = address(SecondLendOrder);
        lendAmountPerOrder[1] = 25e17;
        porcentageOfRatioPerLendOrder[1] = 10000;

        address loan = DebitaV3AggregatorContract.matchOffersV3(
            lendOrders,
            lendAmountPerOrder,
            porcentageOfRatioPerLendOrder,
            address(BorrowOrder),
            principles,
            indexForPrinciple_BorrowOrder,
            indexForCollateral_LendOrder,
            indexPrinciple_LendOrder
        );

        DebitaV3LoanContract = DebitaV3Loan(loan);
    }

    function testIncentivesTwoLendersReceipt() public {
        address[] memory principles = allDynamicData.getDynamicAddressArray(2);
        address[] memory collateral = allDynamicData.getDynamicAddressArray(2);
        address[] memory incentiveToken = allDynamicData.getDynamicAddressArray(
            2
        );

        bool[] memory isLend = allDynamicData.getDynamicBoolArray(2);
        uint[] memory amount = allDynamicData.getDynamicUintArray(2);
        uint[] memory epochs = allDynamicData.getDynamicUintArray(2);

        principles[0] = AERO;
        collateral[0] = address(receiptContract);
        incentiveToken[0] = USDC;
        isLend[0] = true;
        amount[0] = 100e6;
        epochs[0] = 2;

        principles[1] = AERO;
        collateral[1] = address(receiptContract);
        incentiveToken[1] = USDC;
        isLend[1] = false;
        amount[1] = 100e6;
        epochs[1] = 2;

        address[] memory tokensUsedAsBribes = allDynamicData
            .getDynamicAddressArray(1);
        tokensUsedAsBribes[0] = USDC;

        incentivesContract.whitelListCollateral(
            AERO,
            address(receiptContract),
            true
        );
        deal(USDC, address(this), 1000e18, false);
        IERC20(USDC).approve(address(incentivesContract), 1000e18);
        incentivesContract.incentivizePair(
            principles,
            incentiveToken,
            isLend,
            amount,
            epochs
        );

        vm.warp(block.timestamp + 15 days);
        uint currentEpoch = incentivesContract.currentEpoch();
        address[][] memory tokensIncentives = new address[][](
            incentiveToken.length
        );

        tokensIncentives[0] = tokensUsedAsBribes;
        MatchOffers();
        vm.prank(firstLender);
        vm.expectRevert();
        incentivesContract.claimIncentives(principles, tokensIncentives, 2);
        vm.warp(block.timestamp + 30 days);
        DebitaV3Loan.LoanData memory loanData = DebitaV3LoanContract
            .getLoanData();

        uint porcentageOfLending = (loanData
            ._acceptedOffers[0]
            .principleAmount * 10000) / 50e17;
        uint porcentageOfLendingSecond = (loanData
            ._acceptedOffers[1]
            .principleAmount * 10000) / 50e17;

        uint incentivesFirstLender = (100e6 * porcentageOfLending) / 10000;
        uint incentivesSecondLender = (100e6 * porcentageOfLendingSecond) /
            10000;

        uint balanceBeforeFirstLender = IERC20(USDC).balanceOf(firstLender);
        vm.prank(firstLender);
        incentivesContract.claimIncentives(principles, tokensIncentives, 2);
        uint balanceAfterFirstLender = IERC20(USDC).balanceOf(firstLender);

        uint balanceBeforeSecondLender = IERC20(USDC).balanceOf(secondLender);
        vm.prank(secondLender);
        incentivesContract.claimIncentives(principles, tokensIncentives, 2);
        uint balanceAfterSecondLender = IERC20(USDC).balanceOf(secondLender);

        uint balanceBeforeBorrower = IERC20(USDC).balanceOf(borrower);
        vm.prank(borrower);
        incentivesContract.claimIncentives(principles, tokensIncentives, 2);
        uint balanceAfterBorrower = IERC20(USDC).balanceOf(borrower);

        assertEq(
            balanceBeforeSecondLender + incentivesSecondLender,
            balanceAfterSecondLender
        );

        assertEq(
            balanceBeforeFirstLender + incentivesFirstLender,
            balanceAfterFirstLender
        );

        assertEq(balanceBeforeBorrower + 100e6, balanceAfterBorrower);
    }

    // calculate interest with input as anual interest, days borrowed & amount lent per lender
    function calculateInterest(
        uint anualInterestPorcentage,
        uint daysBorrowed,
        uint amountPerLender
    ) internal pure returns (uint) {
        uint anualInterest = (amountPerLender * anualInterestPorcentage) /
            10000;
        uint interestToPay = (anualInterest * daysBorrowed) / 31536000;
        return interestToPay;
    }
}
