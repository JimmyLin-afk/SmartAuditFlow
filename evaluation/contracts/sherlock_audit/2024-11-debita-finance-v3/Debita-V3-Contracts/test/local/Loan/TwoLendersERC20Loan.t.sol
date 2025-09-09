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
import {DynamicData} from "../../interfaces/getDynamicData.sol";
// import ERC20
import {ERC20Mock} from "@openzeppelin/contracts/mocks/token/ERC20Mock.sol";
import {DebitaV3Loan} from "@contracts/DebitaV3Loan.sol";
import {DebitaIncentives} from "@contracts/DebitaIncentives.sol";
import {DebitaV3Loan} from "@contracts/DebitaV3Loan.sol";

contract TwoLendersERC20Loan is Test, DynamicData {
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

    address AERO;
    address USDC;
    address borrower = address(0x02);
    address firstLender = address(this);
    address secondLender = 0x5C235931376b21341fA00d8A606e498e1059eCc0;
    address buyer = 0x5C235931376b21341fA00d8A606e498e1059eCc0;

    address feeAddress = address(this);

    uint receiptID;

    function setUp() public {
        allDynamicData = new DynamicData();
        ownershipsContract = new Ownerships();
        incentivesContract = new DebitaIncentives();
        DBOImplementation borrowOrderImplementation = new DBOImplementation();
        DBOFactoryContract = new DBOFactory(address(borrowOrderImplementation));
        DLOImplementation proxyImplementation = new DLOImplementation();
        DLOFactoryContract = new DLOFactory(address(proxyImplementation));
        auctionFactoryDebitaContract = new auctionFactoryDebita();
        AEROContract = new ERC20Mock();
        deal(address(AEROContract), address(this), 1000e18, true);
        USDCContract = new ERC20Mock();
        DebitaV3Loan loanInstance = new DebitaV3Loan();
        DebitaV3AggregatorContract = new DebitaV3Aggregator(
            address(DLOFactoryContract),
            address(DBOFactoryContract),
            address(incentivesContract),
            address(ownershipsContract),
            address(auctionFactoryDebitaContract),
            address(loanInstance)
        );

        AERO = address(AEROContract);
        USDC = address(USDCContract);

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
        deal(USDC, borrower, 1000e18, false);

        vm.startPrank(borrower);

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
        acceptedCollaterals[0] = USDC;
        oraclesActivated[0] = false;
        ltvs[0] = 0;

        USDCContract.approve(address(DBOFactoryContract), 11e18);
        address borrowOrderAddress = DBOFactoryContract.createBorrowOrder(
            oraclesActivated,
            ltvs,
            1400,
            864000,
            acceptedPrinciples,
            USDC,
            false,
            0,
            oraclesPrinciples,
            ratio,
            address(0x0),
            10e18
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

    function testMultipleLenders() public {
        matchOffers();
        DebitaV3Loan.LoanData memory loanData = DebitaV3LoanContract
            .getLoanData();
        uint[] memory indexes = allDynamicData.getDynamicUintArray(2);
        indexes[0] = 0;
        indexes[1] = 1;
        vm.startPrank(borrower);
        AEROContract.approve(address(DebitaV3LoanContract), 6e18);
        DebitaV3LoanContract.payDebt(indexes);

        uint balanceBeforeBorrower = IERC20(USDC).balanceOf(borrower);
        DebitaV3LoanContract.claimCollateralAsBorrower(indexes);
        uint balanceAfterBorrower = IERC20(USDC).balanceOf(borrower);

        vm.stopPrank();

        uint balanceBeforeFirstLender = IERC20(AERO).balanceOf(firstLender);
        vm.prank(firstLender);
        DebitaV3LoanContract.claimDebt(0);
        uint balanceAfterFirstLender = IERC20(AERO).balanceOf(firstLender);

        uint balanceBeforeSecondLender = IERC20(AERO).balanceOf(secondLender);
        vm.prank(secondLender);
        DebitaV3LoanContract.claimDebt(1);
        uint balanceAfterSecondLender = IERC20(AERO).balanceOf(secondLender);

        console.logUint(loanData.valuableCollateralUsed);

        assertEq(
            balanceBeforeBorrower + loanData.valuableCollateralUsed,
            balanceAfterBorrower,
            "Balance Borrower not equal"
        );
        // 10%
        uint anualInterest = (loanData._acceptedOffers[0].principleAmount *
            loanData._acceptedOffers[0].apr) / 10000;
        uint interest = (anualInterest * 86400) / 31536000;
        uint feeAmountFirstLender = (interest * 1500) / 10000;
        assertEq(
            balanceBeforeFirstLender +
                loanData._acceptedOffers[0].principleAmount +
                interest -
                feeAmountFirstLender,
            balanceAfterFirstLender,
            "Balance First lender not equal"
        );

        uint anualInterestSecondLender = (loanData
            ._acceptedOffers[1]
            .principleAmount * loanData._acceptedOffers[1].apr) / 10000;
        uint interestAmountSecondLender = (anualInterestSecondLender * 86400) /
            31536000;
        uint feeAmountSecondLender = (interestAmountSecondLender * 1500) /
            10000;
        assertEq(
            balanceBeforeSecondLender +
                loanData._acceptedOffers[1].principleAmount +
                interestAmountSecondLender -
                feeAmountSecondLender,
            balanceAfterSecondLender,
            "Balance Second lender not equal"
        );
    }

    function testFullDefault() public {
        matchOffers();
        DebitaV3Loan.LoanData memory loanData = DebitaV3LoanContract
            .getLoanData();
        // next deadline
        vm.warp(block.timestamp + 8640010);
        uint[] memory indexes = allDynamicData.getDynamicUintArray(2);
        indexes[0] = 0;
        indexes[1] = 1;
        vm.startPrank(borrower);
        AEROContract.approve(address(DebitaV3LoanContract), 6e18);
        vm.expectRevert();
        DebitaV3LoanContract.payDebt(indexes);
        vm.stopPrank();

        vm.expectRevert();
        vm.prank(borrower);
        DebitaV3LoanContract.claimCollateralAsBorrower(indexes);

        vm.expectRevert();
        vm.prank(firstLender);
        DebitaV3LoanContract.claimDebt(0);

        vm.expectRevert();
        vm.prank(secondLender);
        DebitaV3LoanContract.claimDebt(1);

        vm.expectRevert();
        vm.prank(firstLender);
        DebitaV3LoanContract.claimCollateralAsLender(1);

        vm.expectRevert();
        vm.prank(secondLender);
        DebitaV3LoanContract.claimCollateralAsLender(0);

        // claim collateral as lender
        uint balanceBeforeFirstLender = IERC20(USDC).balanceOf(firstLender);
        vm.prank(firstLender);
        DebitaV3LoanContract.claimCollateralAsLender(0);
        uint balanceAfterFirstLender = IERC20(USDC).balanceOf(firstLender);

        uint balanceBeforeSecondLender = IERC20(USDC).balanceOf(secondLender);
        vm.prank(secondLender);
        DebitaV3LoanContract.claimCollateralAsLender(1);
        uint balanceAfterSecondLender = IERC20(USDC).balanceOf(secondLender);

        uint collateralUsedFirstLender = loanData
            ._acceptedOffers[0]
            .collateralUsed;
        uint collateralUsedSecondLender = loanData
            ._acceptedOffers[1]
            .collateralUsed;
        assert(collateralUsedSecondLender > collateralUsedFirstLender);
        assertEq(
            balanceBeforeFirstLender + collateralUsedFirstLender,
            balanceAfterFirstLender
        );
        assertEq(
            balanceBeforeSecondLender + collateralUsedSecondLender,
            balanceAfterSecondLender
        );
    }

    function testPartialDefault() public {
        matchOffers();
        DebitaV3Loan.LoanData memory loanData = DebitaV3LoanContract
            .getLoanData();
        uint[] memory indexes = allDynamicData.getDynamicUintArray(1);
        indexes[0] = 0;

        vm.startPrank(borrower);
        AEROContract.approve(address(DebitaV3LoanContract), 6e18);
        DebitaV3LoanContract.payDebt(indexes);
        vm.stopPrank();

        uint balanceBeforeBorrower = IERC20(USDC).balanceOf(borrower);
        vm.prank(borrower);
        DebitaV3LoanContract.claimCollateralAsBorrower(indexes);
        uint balanceAfterBorrower = IERC20(USDC).balanceOf(borrower);

        uint balanceBeforeFirstLender = IERC20(AERO).balanceOf(firstLender);
        vm.prank(firstLender);
        DebitaV3LoanContract.claimDebt(0);
        uint balanceAfterFirstLender = IERC20(AERO).balanceOf(firstLender);

        vm.expectRevert();
        vm.prank(firstLender);
        DebitaV3LoanContract.claimCollateralAsLender(0);

        // deadline first offer
        vm.warp(block.timestamp + 8640010);
        vm.expectRevert();
        vm.prank(firstLender);
        DebitaV3LoanContract.claimCollateralAsLender(0);

        // deadline all loans

        vm.warp(block.timestamp + 9640010);
        uint balanceBeforeSecondLender = IERC20(USDC).balanceOf(secondLender);
        vm.prank(secondLender);
        DebitaV3LoanContract.claimCollateralAsLender(1);
        uint balanceAfterSecondLender = IERC20(USDC).balanceOf(secondLender);

        uint collateralUsedSecondLender = loanData
            ._acceptedOffers[1]
            .collateralUsed;

        uint collateralUsedFirstLender = loanData
            ._acceptedOffers[0]
            .collateralUsed;

        uint anualInterest = (loanData._acceptedOffers[0].principleAmount *
            loanData._acceptedOffers[0].apr) / 10000;
        uint firstLenderInterest = (anualInterest * 86400) / 31536000;
        uint fee = (firstLenderInterest * 1500) / 10000;

        assertEq(
            balanceBeforeFirstLender +
                loanData._acceptedOffers[0].principleAmount +
                firstLenderInterest -
                fee,
            balanceAfterFirstLender
        );
        assertEq(
            balanceBeforeBorrower + collateralUsedFirstLender,
            balanceAfterBorrower
        );
        assertEq(
            balanceBeforeSecondLender + collateralUsedSecondLender,
            balanceAfterSecondLender
        );
    }

    function testCheckIncentivesWithTwoLenders() public {
        address[] memory principles = allDynamicData.getDynamicAddressArray(2);
        address[] memory collateral = allDynamicData.getDynamicAddressArray(2);
        address[] memory incentiveToken = allDynamicData.getDynamicAddressArray(
            2
        );

        bool[] memory isLend = allDynamicData.getDynamicBoolArray(2);
        uint[] memory amount = allDynamicData.getDynamicUintArray(2);
        uint[] memory epochs = allDynamicData.getDynamicUintArray(2);

        principles[0] = AERO;
        collateral[0] = USDC;
        incentiveToken[0] = USDC;
        isLend[0] = true;
        amount[0] = 100e6;
        epochs[0] = 2;

        principles[1] = AERO;
        collateral[1] = USDC;
        incentiveToken[1] = USDC;
        isLend[1] = false;
        amount[1] = 100e6;
        epochs[1] = 2;

        address[] memory tokensUsedAsBribes = allDynamicData
            .getDynamicAddressArray(1);
        tokensUsedAsBribes[0] = USDC;
        deal(AERO, address(this), 10000e18);
        deal(USDC, address(this), 10000e18);

        incentivesContract.whitelListCollateral(AERO, USDC, true);
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
        matchOffers();
        vm.warp(block.timestamp + 30 days);
        DebitaV3Loan.LoanData memory loanData = DebitaV3LoanContract
            .getLoanData();
        address[][] memory tokensIncentives = new address[][](
            incentiveToken.length
        );

        tokensIncentives[0] = tokensUsedAsBribes;
        uint porcentageOfLending = (loanData
            ._acceptedOffers[0]
            .principleAmount * 10000) / 45e17;
        uint porcentageOfLendingSecond = (loanData
            ._acceptedOffers[1]
            .principleAmount * 10000) / 45e17;

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

    function matchOffers() public {
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
        lendAmountPerOrder[1] = 20e17;
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
}
