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
    address connector = 0x81B2c95353d69580875a7aFF5E8f018F1761b7D1;

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
            DebitaChainlinkOracle,
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
        uint interestSecondLender = calculateInterest(1);
        uint interestThirdLender = calculateInterest(2);
        uint interestFirstLender = calculateInterest(0);
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
        uint feeFirstLender = (interestFirstLender * 1500) / 10000;

        uint amountSecondLender = 38e17;

        uint feeSecondLender = (interestSecondLender * 1500) / 10000;

        uint amountThirdLender = 20e17;
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
        uint interest = calculateInterest(0);

        DebitaV3LoanContract.payDebt(indexes);
        vm.stopPrank();

        uint balanceBeforeFirstLender = AEROContract.balanceOf(firstLender);
        vm.prank(firstLender);
        DebitaV3LoanContract.claimDebt(0);
        uint balanceAfterFirstLender = AEROContract.balanceOf(firstLender);
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

    function testPartialDefaultExtendLoan() public {
        matchOffers();
        uint[] memory indexes = allDynamicData.getDynamicUintArray(3);
        indexes[0] = 0;
        indexes[1] = 1;
        indexes[2] = 2;
        vm.startPrank(borrower);
        deal(wETH, borrower, 10e18, false);
        AEROContract.approve(address(DebitaV3LoanContract), 10e18);
        wETHContract.approve(address(DebitaV3LoanContract), 10e18);
        vm.warp(block.timestamp + 864001);
        vm.expectRevert();
        DebitaV3LoanContract.payDebt(indexes);
    }

    function testExtendLoan() public {
        matchOffers();
        uint[] memory indexes = allDynamicData.getDynamicUintArray(3);
        indexes[0] = 0;
        indexes[1] = 1;
        indexes[2] = 2;
        vm.startPrank(borrower);
        deal(wETH, borrower, 10e18, false);
        AEROContract.approve(address(DebitaV3LoanContract), 10e18);
        wETHContract.approve(address(DebitaV3LoanContract), 10e18);
        vm.warp(block.timestamp + 86401);
        DebitaV3LoanContract.extendLoan();
        vm.warp(block.timestamp + 86501);
        DebitaV3LoanContract.payDebt(indexes);
        vm.stopPrank();
    }

    function testExtendLoanAndRevertAfterDeadlineOfExtend() public {
        matchOffers();
        uint[] memory indexes = allDynamicData.getDynamicUintArray(1);
        indexes[0] = 0;

        vm.startPrank(borrower);
        deal(wETH, borrower, 10e18, false);
        AEROContract.approve(address(DebitaV3LoanContract), 10e18);
        wETHContract.approve(address(DebitaV3LoanContract), 10e18);
        DebitaV3LoanContract.payDebt(indexes);

        vm.warp(block.timestamp + 86401);
        DebitaV3LoanContract.extendLoan();
        vm.warp(block.timestamp + 8640000 + 1);
        indexes[0] = 1;
        DebitaV3LoanContract.payDebt(indexes);
        vm.warp(block.timestamp + 9640000 + 1);

        indexes[0] = 2;
        vm.expectRevert();
        DebitaV3LoanContract.payDebt(indexes);
        indexes[0] = 1;
        DebitaV3LoanContract.claimCollateralAsBorrower(indexes);
        indexes[0] = 0;
        DebitaV3LoanContract.claimCollateralAsBorrower(indexes);

        vm.stopPrank();
    }

    function testIncentivizeOnePair() public {
        address[] memory principles = allDynamicData.getDynamicAddressArray(1);
        address[] memory collateral = allDynamicData.getDynamicAddressArray(1);
        address[] memory incentiveToken = allDynamicData.getDynamicAddressArray(
            1
        );

        bool[] memory isLend = allDynamicData.getDynamicBoolArray(1);
        uint[] memory amount = allDynamicData.getDynamicUintArray(1);
        uint[] memory epochs = allDynamicData.getDynamicUintArray(1);

        principles[0] = AERO;
        collateral[0] = USDC;
        incentiveToken[0] = AERO;
        isLend[0] = true;
        amount[0] = 100e18;
        epochs[0] = 2;
        incentivesContract.whitelListCollateral(AERO, USDC, true);

        IERC20(AERO).approve(address(incentivesContract), 1000e18);
        incentivesContract.incentivizePair(
            principles,
            incentiveToken,
            isLend,
            amount,
            epochs
        );
        vm.warp(block.timestamp + 15 days);
        matchOffers();
        vm.warp(block.timestamp + 32 days);
        address[] memory tokenUsed = allDynamicData.getDynamicAddressArray(1);
        tokenUsed[0] = AERO;
        principles[0] = AERO;

        address[][] memory tokensIncentives = new address[][](tokenUsed.length);

        tokensIncentives[0] = tokenUsed;

        uint balanceBefore = IERC20(AERO).balanceOf(firstLender);
        vm.prank(firstLender);
        incentivesContract.claimIncentives(principles, tokensIncentives, 2);
        uint balanceAfter = IERC20(AERO).balanceOf(firstLender);
        assertEq(balanceAfter, balanceBefore + 100e18);
    }

    function calculateInterest(uint index) internal returns (uint) {
        return DebitaV3LoanContract.calculateInterestToPay(index);
    }

    function testCheckFees() public {
        // borrow info
        DBOImplementation.BorrowInfo memory borrowInfo = BorrowOrder
            .getBorrowInfo();
        uint AEROAmount = 25e17;
        uint wETHAmount = 58e17;

        uint balanceBefore = AEROContract.balanceOf(feeAddress);
        uint balanceBeforeConnector = AEROContract.balanceOf(connector);
        uint balanceBeforeWETH = wETHContract.balanceOf(feeAddress);
        uint balanceBeforeWETHConnector = wETHContract.balanceOf(connector);
        matchOffers();

        uint balanceAfter = AEROContract.balanceOf(feeAddress);
        uint balanceAfterConnector = AEROContract.balanceOf(connector);
        uint balanceAfterWETH = wETHContract.balanceOf(feeAddress);
        uint balanceAfterWETHConnector = wETHContract.balanceOf(connector);

        uint duration = borrowInfo.duration;

        uint feePercentage = (duration * 4) / 86400;
        if (feePercentage > 80) {
            feePercentage = 80;
        }
        if (feePercentage < 20) {
            feePercentage = 20;
        }
        uint fee = (AEROAmount * feePercentage) / 10000;
        uint feeWETH = (wETHAmount * feePercentage) / 10000;

        uint feeConnectorWETH = (feeWETH * 1500) / 10000;
        uint feeConnector = (fee * 1500) / 10000;

        assertEq(
            balanceBefore + fee - feeConnector,
            balanceAfter,
            "Fee Debita"
        );
        assertEq(
            balanceBeforeConnector + feeConnector,
            balanceAfterConnector,
            "Fee Connector"
        );

        assertEq(
            balanceBeforeWETH + feeWETH - feeConnectorWETH,
            balanceAfterWETH,
            "Fee Debita WETH"
        );

        assertEq(
            balanceBeforeWETHConnector + feeConnectorWETH,
            balanceAfterWETHConnector,
            "Fee Connector WETH"
        );

        vm.startPrank(borrower);

        // extendLoan
        vm.warp(block.timestamp + 86401);
        uint balanceBeforeExtend = AEROContract.balanceOf(feeAddress);
        uint balanceWETHBeforeExtend = wETHContract.balanceOf(feeAddress);
        wETHContract.approve(address(DebitaV3LoanContract), 100e18);
        AEROContract.approve(address(DebitaV3LoanContract), 100e18);
        uint interest = calculateInterest(0);
        uint interestSecondOffer = calculateInterest(1);
        uint interestThirdOffer = calculateInterest(2);
        DebitaV3LoanContract.extendLoan();

        uint feeInterest = (interest * 1500) / 10000;
        uint feeInterestSecondOffer = (interestSecondOffer * 1500) / 10000;
        uint feeInterestThirdOffer = (interestThirdOffer * 1500) / 10000;

        uint balanceAfterExtend = AEROContract.balanceOf(feeAddress);
        uint balanceWETHAfterExtend = wETHContract.balanceOf(feeAddress);
        vm.stopPrank();
        uint feePercentageExtend = 80 - feePercentage;
        uint balanceBeforeFirstLender = AEROContract.balanceOf(firstLender);
        vm.startPrank(firstLender);
        DebitaV3LoanContract.claimDebt(0);
        uint balanceAfterFirstLender = AEROContract.balanceOf(firstLender);
        vm.expectRevert();
        DebitaV3LoanContract.claimDebt(0);
        vm.expectRevert();
        DebitaV3LoanContract.claimDebt(1);
        vm.stopPrank();
        vm.startPrank(borrower);
        uint feeExtend = (AEROAmount * feePercentageExtend) / 10000;
        uint feeExtendWETH = (wETHAmount * feePercentageExtend) / 10000;

        vm.expectRevert();
        DebitaV3LoanContract.extendLoan();
        vm.stopPrank();
        assertEq(
            balanceAfterFirstLender,
            balanceBeforeFirstLender + interest - feeInterest,
            "First lender balance"
        );
        /*  assertEq(
            balanceBeforeExtend + feeExtend + feeInterest,
            balanceAfterExtend,
            "Fee Debita Extend"
        ); */

        assertEq(
            balanceWETHBeforeExtend +
                feeExtendWETH +
                feeInterestSecondOffer +
                feeInterestThirdOffer,
            balanceWETHAfterExtend,
            "Fee Debita Extend WETH"
        );
    }

    function testClaimDebtAfterExtend() public {
        matchOffers();
        uint[] memory indexes = allDynamicData.getDynamicUintArray(3);
        indexes[0] = 0;
        indexes[1] = 1;
        indexes[2] = 2;
        vm.startPrank(borrower);
        deal(wETH, borrower, 100e18, false);
        AEROContract.approve(address(DebitaV3LoanContract), 100e18);
        wETHContract.approve(address(DebitaV3LoanContract), 100e18);
        vm.warp(block.timestamp + 86401);
        uint interestFirstOffer = calculateInterest(0);
        uint interestSecondOffer = calculateInterest(1);
        uint interestThirdOffer = calculateInterest(2);

        uint fee = (interestFirstOffer * 1500) / 10000;
        uint feeSecondOffer = (interestSecondOffer * 1500) / 10000;
        uint feeThirdOffer = (interestThirdOffer * 1500) / 10000;

        DebitaV3LoanContract.extendLoan();
        vm.stopPrank();

        vm.startPrank(firstLender);
        uint balanceBeforeFirstLender = AEROContract.balanceOf(firstLender);
        DebitaV3LoanContract.claimDebt(0);
        vm.expectRevert();
        DebitaV3LoanContract.claimDebt(0);
        vm.expectRevert();
        DebitaV3LoanContract.claimDebt(1);
        uint balanceAfterFirstLender = AEROContract.balanceOf(firstLender);

        vm.stopPrank();

        vm.startPrank(secondLender);
        uint balanceBeforeSecondLender = wETHContract.balanceOf(secondLender);
        DebitaV3LoanContract.claimDebt(1);
        vm.expectRevert();
        DebitaV3LoanContract.claimDebt(1);

        uint balanceAfterSecondLender = wETHContract.balanceOf(secondLender);
        vm.stopPrank();

        vm.startPrank(thirdLender);
        uint balanceBeforeThirdLender = wETHContract.balanceOf(thirdLender);
        DebitaV3LoanContract.claimDebt(2);
        vm.expectRevert();
        DebitaV3LoanContract.claimDebt(2);
        uint balanceAfterThirdLender = wETHContract.balanceOf(thirdLender);
        vm.stopPrank();

        vm.startPrank(borrower);
        DebitaV3LoanContract.payDebt(indexes);
        vm.stopPrank();

        // claim debt again
        vm.startPrank(firstLender);
        uint balanceBefore_AfterPayment = AEROContract.balanceOf(firstLender);
        uint interestFirstOffer_AfterPayment = calculateInterest(0);
        uint fee_AfterPayment = (interestFirstOffer_AfterPayment * 1500) /
            10000;
        DebitaV3LoanContract.claimDebt(0);
        uint balanceAfter_AfterPayment = AEROContract.balanceOf(firstLender);
        vm.stopPrank();

        vm.startPrank(secondLender);
        uint balanceBeforeSecond_AfterPayment = wETHContract.balanceOf(
            secondLender
        );
        DebitaV3LoanContract.claimDebt(1);
        uint balanceAfterSecond_AfterPayment = wETHContract.balanceOf(
            secondLender
        );
        vm.stopPrank();

        vm.startPrank(thirdLender);
        uint balanceBeforeThird_AfterPayment = wETHContract.balanceOf(
            thirdLender
        );
        DebitaV3LoanContract.claimDebt(2);
        uint balanceAfterThird_AfterPayment = wETHContract.balanceOf(
            thirdLender
        );
        vm.stopPrank();

        assertEq(
            balanceAfterFirstLender,
            balanceBeforeFirstLender + interestFirstOffer - fee,
            "First lender balance"
        );

        assertEq(
            balanceAfterSecondLender,
            balanceBeforeSecondLender + interestSecondOffer - feeSecondOffer,
            "Second lender balance"
        );

        assertEq(
            balanceAfterThirdLender,
            balanceBeforeThirdLender + interestThirdOffer - feeThirdOffer,
            "Third lender balance"
        );

        assertEq(
            balanceAfter_AfterPayment,
            balanceBefore_AfterPayment +
                interestFirstOffer_AfterPayment -
                fee_AfterPayment +
                25e17,
            "First lender balance after payment"
        );

        //   assertEq(interestFirstOffer_AfterPayment - fee_AfterPayment , 0);
    }

    function matchOffers() public {
        vm.startPrank(connector);
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
        vm.stopPrank();
    }

    function setOracles() internal {
        DebitaChainlink oracle = new DebitaChainlink(
            0xBCF85224fc0756B9Fa45aA7892530B47e10b6433,
            address(this)
        );
        DebitaPyth oracle2 = new DebitaPyth(address(0x0), address(0x0));
        DebitaV3AggregatorContract.setOracleEnabled(address(oracle), true);
        DebitaV3AggregatorContract.setOracleEnabled(address(oracle2), true);

        oracle.setPriceFeeds(AERO, AEROFEED);
        oracle.setPriceFeeds(USDC, USDCFEED);
        oracle.setPriceFeeds(wETH, WETHFEED);

        DebitaChainlinkOracle = address(oracle);
        DebitaPythOracle = address(oracle2);
    }
}
