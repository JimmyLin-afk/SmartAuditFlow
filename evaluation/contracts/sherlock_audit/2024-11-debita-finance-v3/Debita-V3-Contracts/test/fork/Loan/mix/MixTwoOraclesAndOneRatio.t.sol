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
import {DebitaV3Loan} from "@contracts/DebitaV3Loan.sol";
import {DebitaChainlink} from "@contracts/oracles/DebitaChainlink.sol";
import {DebitaPyth} from "@contracts/oracles/DebitaPyth.sol";

contract testMultiplePrinciples is Test {
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
    ERC20Mock public wETHContract;
    DLOImplementation public LendOrder;
    DLOImplementation public SecondLendOrder;
    DLOImplementation public ThirdLendOrder;

    address DebitaChainlinkOracle;
    address DebitaPythOracle;

    DBOImplementation public BorrowOrder;

    address AERO = 0x940181a94A35A4569E4529A3CDfB74e38FD98631;
    address USDC = 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913;
    address wETH = 0x4200000000000000000000000000000000000006;
    address AEROFEED = 0x4EC5970fC728C5f65ba413992CD5fF6FD70fcfF0;
    address USDCFEED = 0x7e860098F58bBFC8648a4311b374B1D669a2bc6B;
    address WETHFEED = 0x71041dddad3595F9CEd3DcCFBe3D1F4b0a16Bb70;
    address borrower = address(0x02);
    address firstLender = address(this);
    address secondLender = 0x5C235931376b21341fA00d8A606e498e1059eCc0;
    address thirdLender = 0x25ABd53Ea07dc7762DE910f155B6cfbF3B99B296;
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
        AEROContract = ERC20Mock(AERO);
        USDCContract = ERC20Mock(USDC);
        wETHContract = ERC20Mock(wETH);

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
        deal(USDC, borrower, 1000e18, false);
        deal(wETH, secondLender, 1000e18, false);
        deal(wETH, thirdLender, 1000e18, false);
        setOracles();
        vm.startPrank(borrower);

        IERC20(AERO).approve(address(DBOFactoryContract), 100e18);

        bool[] memory oraclesActivated = allDynamicData.getDynamicBoolArray(2);
        uint[] memory ltvs = allDynamicData.getDynamicUintArray(2);
        uint[] memory ratio = allDynamicData.getDynamicUintArray(2);
        uint[] memory ratioLenders = allDynamicData.getDynamicUintArray(1);
        uint[] memory ltvsLenders = allDynamicData.getDynamicUintArray(1);
        bool[] memory oraclesActivatedLenders = allDynamicData
            .getDynamicBoolArray(1);

        address[] memory acceptedPrinciples = allDynamicData
            .getDynamicAddressArray(2);
        address[] memory acceptedCollaterals = allDynamicData
            .getDynamicAddressArray(1);
        address[] memory oraclesCollateral = allDynamicData
            .getDynamicAddressArray(1);
        address[] memory oraclesPrinciples = allDynamicData
            .getDynamicAddressArray(2);

        ltvs[0] = 5000;
        acceptedPrinciples[0] = AERO;
        acceptedCollaterals[0] = USDC;
        oraclesActivated[0] = true;

        ltvs[1] = 7000;

        acceptedPrinciples[1] = wETH;
        oraclesActivated[1] = true;
        oraclesPrinciples[0] = DebitaChainlinkOracle;
        oraclesPrinciples[1] = DebitaChainlinkOracle;
        oraclesCollateral[0] = DebitaChainlinkOracle;

        USDCContract.approve(address(DBOFactoryContract), 101e18);
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
            DebitaChainlinkOracle,
            40e18
        );
        vm.stopPrank();

        AEROContract.approve(address(DLOFactoryContract), 5e18);
        ratioLenders[0] = 5e17;
        ltvsLenders[0] = 5000;
        oraclesCollateral[0] = DebitaChainlinkOracle;
        oraclesActivatedLenders[0] = true;
        address lendOrderAddress = DLOFactoryContract.createLendOrder(
            false,
            oraclesActivatedLenders,
            false,
            ltvsLenders,
            1350,
            8640000,
            86400,
            acceptedCollaterals,
            AERO,
            oraclesCollateral,
            ratioLenders,
            DebitaPythOracle,
            5e18
        );

        vm.startPrank(secondLender);
        wETHContract.approve(address(DLOFactoryContract), 5e18);
        ratioLenders[0] = 4e17;
        ltvsLenders[0] = 6900;

        address SecondlendOrderAddress = DLOFactoryContract.createLendOrder(
            false,
            oraclesActivatedLenders,
            false,
            ltvsLenders,
            1000,
            9640000,
            86400,
            acceptedCollaterals,
            wETH,
            oraclesCollateral,
            ratioLenders,
            DebitaChainlinkOracle,
            5e18
        );
        vm.stopPrank();

        vm.startPrank(thirdLender);
        wETHContract.approve(address(DLOFactoryContract), 5e18);
        ratioLenders[0] = 266e12;
        ltvsLenders[0] = 0;
        oraclesActivatedLenders[0] = false;

        address ThirdlendOrderAddress = DLOFactoryContract.createLendOrder(
            false,
            oraclesActivatedLenders,
            false,
            ltvsLenders,
            1000,
            9640000,
            86400,
            acceptedCollaterals,
            wETH,
            oraclesCollateral,
            ratioLenders,
            DebitaChainlinkOracle,
            5e18
        );
        vm.stopPrank();

        ThirdLendOrder = DLOImplementation(ThirdlendOrderAddress);
        LendOrder = DLOImplementation(lendOrderAddress);
        BorrowOrder = DBOImplementation(borrowOrderAddress);
        SecondLendOrder = DLOImplementation(SecondlendOrderAddress);
    }

    function testOffers() public {
        matchOffers();
        uint[] memory indexes = allDynamicData.getDynamicUintArray(3);
        indexes[0] = 0;
        indexes[1] = 1;
        indexes[2] = 2;
        vm.startPrank(borrower);
        deal(wETH, borrower, 10e18, false);
        AEROContract.approve(address(DebitaV3LoanContract), 10e18);
        wETHContract.approve(address(DebitaV3LoanContract), 10e18);
        DebitaV3LoanContract.payDebt(indexes);

        DebitaV3LoanContract.claimCollateralAsBorrower(indexes);
        vm.stopPrank();

        uint balanceBeforeFirstLender = AEROContract.balanceOf(firstLender);
        vm.prank(firstLender);
        DebitaV3LoanContract.claimDebt(0);
        uint balanceAfterFirstLender = AEROContract.balanceOf(firstLender);

        uint balanceBeforeSecondLender = wETHContract.balanceOf(secondLender);
        vm.prank(secondLender);
        DebitaV3LoanContract.claimDebt(1);
        uint balanceAfterSecondLender = wETHContract.balanceOf(secondLender);

        uint balanceBeforeThirdLender = wETHContract.balanceOf(thirdLender);
        vm.prank(thirdLender);
        DebitaV3LoanContract.claimDebt(2);
        uint balanceAfterThirdLender = wETHContract.balanceOf(thirdLender);

        uint amountFirstLender = 25e17;
        uint interestFirstLender = calculateInterest(0);
        uint feeFirstLender = (interestFirstLender * 1500) / 10000;

        uint amountSecondLender = 38e17;
        uint interestSecondLender = calculateInterest(1);
        uint feeSecondLender = (interestSecondLender * 1500) / 10000;

        uint amountThirdLender = 20e17;
        uint interestThirdLender = calculateInterest(2);
        uint feeThirdLender = (interestThirdLender * 1500) / 10000;

        assertEq(
            balanceAfterThirdLender,
            balanceBeforeThirdLender +
                amountThirdLender +
                interestThirdLender -
                feeThirdLender,
            "Third lender balance"
        );

        assertEq(
            balanceAfterSecondLender,
            balanceBeforeSecondLender +
                amountSecondLender +
                interestSecondLender -
                feeSecondLender,
            "Second lender balance"
        );

        assertEq(
            balanceAfterFirstLender,
            balanceBeforeFirstLender +
                amountFirstLender +
                interestFirstLender -
                feeFirstLender,
            "First lender balance"
        );
    }

    function testPartialDefault() public {
        matchOffers();
        DebitaV3Loan.infoOfOffers[] memory offers = DebitaV3LoanContract
            .getLoanData()
            ._acceptedOffers;
        uint[] memory indexes = allDynamicData.getDynamicUintArray(1);
        vm.startPrank(borrower);
        deal(wETH, borrower, 10e18, false);
        wETHContract.approve(address(DebitaV3LoanContract), 10e18);
        AEROContract.approve(address(DebitaV3LoanContract), 10e18);
        DebitaV3LoanContract.payDebt(indexes);
        vm.stopPrank();

        uint balanceBeforeFirstLender = AEROContract.balanceOf(firstLender);
        vm.prank(firstLender);
        DebitaV3LoanContract.claimDebt(0);
        uint balanceAfterFirstLender = AEROContract.balanceOf(firstLender);
        uint interest = calculateInterest(0);
        uint fee = (interest * 1500) / 10000;

        vm.warp(block.timestamp + 9640001);
        vm.expectRevert();
        indexes[0] = 1;
        vm.prank(borrower);
        DebitaV3LoanContract.payDebt(indexes);

        indexes[0] = 2;
        vm.expectRevert();
        vm.prank(borrower);
        DebitaV3LoanContract.payDebt(indexes);

        vm.expectRevert();
        DebitaV3LoanContract.claimCollateralAsBorrower(indexes);

        uint balanceBeforeSecondLender = USDCContract.balanceOf(secondLender);
        vm.prank(secondLender);
        DebitaV3LoanContract.claimCollateralAsLender(1);
        uint balanceAfterSecondLender = USDCContract.balanceOf(secondLender);

        uint balanceBeforeThirdLender = USDCContract.balanceOf(thirdLender);
        vm.prank(thirdLender);
        DebitaV3LoanContract.claimCollateralAsLender(2);
        uint balanceAfterThirdLender = USDCContract.balanceOf(thirdLender);

        uint collateralUsedSecondOrder = offers[1].collateralUsed;
        uint collateralUsedThirdOrder = offers[2].collateralUsed;
        assertEq(
            balanceBeforeFirstLender + interest - fee + 25e17,
            balanceAfterFirstLender
        );
        assertEq(
            balanceBeforeSecondLender + collateralUsedSecondOrder,
            balanceAfterSecondLender
        );
        assertEq(
            balanceBeforeThirdLender + collateralUsedThirdOrder,
            balanceAfterThirdLender
        );
    }

    function calculateInterest(uint index) internal returns (uint) {
        DebitaV3Loan.infoOfOffers memory offer = DebitaV3LoanContract
            .getLoanData()
            ._acceptedOffers[index];
        uint anualInterest = (offer.principleAmount * offer.apr) / 10000;
        uint activeTime = (BorrowOrder.getBorrowInfo().duration * 1000) / 10000;
        uint interestUsed = (anualInterest * activeTime) / 31536000;
        return interestUsed;
    }

    function matchOffers() public {
        address[] memory lendOrders = allDynamicData.getDynamicAddressArray(3);
        uint[] memory lendAmountPerOrder = allDynamicData.getDynamicUintArray(
            3
        );
        uint[] memory porcentageOfRatioPerLendOrder = allDynamicData
            .getDynamicUintArray(3);
        address[] memory principles = allDynamicData.getDynamicAddressArray(2);
        uint[] memory indexForPrinciple_BorrowOrder = allDynamicData
            .getDynamicUintArray(3);
        uint[] memory indexForCollateral_LendOrder = allDynamicData
            .getDynamicUintArray(3);
        uint[] memory indexPrinciple_LendOrder = allDynamicData
            .getDynamicUintArray(3);

        lendOrders[0] = address(LendOrder);
        lendAmountPerOrder[0] = 25e17;
        porcentageOfRatioPerLendOrder[0] = 10000;
        principles[0] = AERO;
        principles[1] = wETH;

        // 0.1e18 --> 1e18 collateral

        lendOrders[1] = address(SecondLendOrder);
        lendAmountPerOrder[1] = 38e17;
        porcentageOfRatioPerLendOrder[1] = 10000;

        indexForPrinciple_BorrowOrder[1] = 1;
        indexPrinciple_LendOrder[1] = 1;

        lendOrders[2] = address(ThirdLendOrder);
        lendAmountPerOrder[2] = 20e17;
        porcentageOfRatioPerLendOrder[2] = 10000;

        indexForPrinciple_BorrowOrder[2] = 1;
        indexPrinciple_LendOrder[2] = 1;

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

    function setOracles() internal {
        DebitaChainlink oracle = new DebitaChainlink(
            0xBCF85224fc0756B9Fa45aA7892530B47e10b6433,
            address(this)
        );
        DebitaPyth oracle2 = new DebitaPyth(
            0x8250f4aF4B972684F7b336503E2D6dFeDeB1487a,
            address(this)
        );

        DebitaV3AggregatorContract.setOracleEnabled(address(oracle), true);
        DebitaV3AggregatorContract.setOracleEnabled(address(oracle2), true);

        oracle.setPriceFeeds(AERO, AEROFEED);
        oracle.setPriceFeeds(USDC, USDCFEED);
        oracle.setPriceFeeds(wETH, WETHFEED);

        oracle2.setPriceFeeds(
            AERO,
            0x9db37f4d5654aad3e37e2e14ffd8d53265fb3026d1d8f91146539eebaa2ef45f
        );
        oracle2.setPriceFeeds(
            USDC,
            0xeaa020c61cc479712813461ce153894a96a6c00b21ed0cfc2798d1f9a9e9c94a
        );

        oracle2.setPriceFeeds(
            wETH,
            0x9d4294bbcd1174d6f2003ec365831e64cc31d9f6f15a2b85399db8d5000960f6
        );

        DebitaChainlinkOracle = address(oracle);
        DebitaPythOracle = address(oracle2);
    }
}
