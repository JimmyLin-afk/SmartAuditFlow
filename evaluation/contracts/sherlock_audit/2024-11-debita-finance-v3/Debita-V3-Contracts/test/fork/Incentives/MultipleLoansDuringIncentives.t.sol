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
import {DebitaChainlink} from "@contracts/oracles/DebitaChainlink.sol";
import {DebitaPyth} from "@contracts/oracles/DebitaPyth.sol";

contract testIncentivesAmongMultipleLoans is Test {
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
    address secondBorrower = address(0x03);
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
        setOracles();
        incentivesContract.whitelListCollateral(AERO, AERO, true);
        incentivesContract.whitelListCollateral(AERO, USDC, true);
    }

    function testIncentivesMultipleLoans() public {
        incentivize(AERO, AERO, USDC, true, 1e18, 2);
        vm.warp(block.timestamp + 15 days);
        createLoan(borrower, secondLender, AERO, AERO);
        createLoan(borrower, thirdLender, AERO, AERO);
        createLoan(borrower, firstLender, AERO, AERO);
        vm.warp(block.timestamp + 30 days);
        uint balanceBefore = IERC20(USDC).balanceOf(secondLender);
        // principles, tokenIncentives, epoch with dynamic Data
        address[] memory principles = allDynamicData.getDynamicAddressArray(1);
        address[] memory tokenUsedIncentive = allDynamicData
            .getDynamicAddressArray(1);
        address[][] memory tokenIncentives = new address[][](
            tokenUsedIncentive.length
        );
        principles[0] = AERO;
        tokenUsedIncentive[0] = USDC;
        tokenIncentives[0] = tokenUsedIncentive;

        vm.startPrank(secondLender);
        incentivesContract.claimIncentives(principles, tokenIncentives, 2);
        vm.expectRevert();
        incentivesContract.claimIncentives(principles, tokenIncentives, 2);
        uint balanceAfter = IERC20(USDC).balanceOf(secondLender);
        vm.stopPrank();

        vm.startPrank(thirdLender);
        uint balanceBefore_Third = IERC20(USDC).balanceOf(thirdLender);
        incentivesContract.claimIncentives(principles, tokenIncentives, 2);
        uint balanceAfter_Third = IERC20(USDC).balanceOf(thirdLender);
        vm.stopPrank();

        vm.startPrank(firstLender);
        uint balanceBefore_First = IERC20(USDC).balanceOf(firstLender);
        incentivesContract.claimIncentives(principles, tokenIncentives, 2);
        uint balanceAfter_First = IERC20(USDC).balanceOf(firstLender);
        vm.expectRevert();
        incentivesContract.claimIncentives(principles, tokenIncentives, 2);
        vm.expectRevert();
        incentivesContract.claimIncentives(principles, tokenIncentives, 3);

        vm.stopPrank();

        uint amount = (1e18 * 3333) / 10000;

        assertEq(balanceBefore_Third + amount, balanceAfter_Third);
        assertEq(balanceBefore_First + amount, balanceAfter_First);
        assertEq(balanceBefore + amount, balanceAfter);
    }

    function testIncentivesBorrowerMultipleLoans() public {
        incentivize(AERO, AERO, USDC, false, 1e18, 2);
        vm.warp(block.timestamp + 15 days);
        createLoan(borrower, secondLender, AERO, AERO);
        createLoan(secondBorrower, thirdLender, AERO, USDC);
        createLoan(secondBorrower, firstLender, AERO, USDC);

        vm.warp(block.timestamp + 30 days);
        uint balanceBefore = IERC20(USDC).balanceOf(borrower);
        // principles, tokenIncentives, epoch with dynamic Data

        address[] memory principles = allDynamicData.getDynamicAddressArray(1);
        address[] memory tokenUsedIncentive = allDynamicData
            .getDynamicAddressArray(1);
        address[][] memory tokenIncentives = new address[][](
            tokenUsedIncentive.length
        );
        principles[0] = AERO;
        tokenUsedIncentive[0] = USDC;
        tokenIncentives[0] = tokenUsedIncentive;
        vm.startPrank(borrower);
        incentivesContract.claimIncentives(principles, tokenIncentives, 2);
        vm.expectRevert();
        incentivesContract.claimIncentives(principles, tokenIncentives, 2);
        uint balanceAfter = IERC20(USDC).balanceOf(borrower);
        vm.stopPrank();

        vm.startPrank(secondBorrower);
        uint balanceBefore_Second = IERC20(USDC).balanceOf(secondBorrower);
        incentivesContract.claimIncentives(principles, tokenIncentives, 2);
        uint balanceAfter_Second = IERC20(USDC).balanceOf(secondBorrower);
        vm.stopPrank();

        vm.startPrank(firstLender);
        uint balanceBefore_First = IERC20(USDC).balanceOf(firstLender);
        vm.expectRevert();
        incentivesContract.claimIncentives(principles, tokenIncentives, 2);
        uint balanceAfter_First = IERC20(USDC).balanceOf(firstLender);
        vm.stopPrank();

        uint amount = (1e18 * 3333) / 10000;

        assertEq(balanceBefore_Second + amount + amount, balanceAfter_Second);
        assertEq(balanceBefore + amount, balanceAfter);
        assertEq(balanceBefore_First, balanceAfter_First);
    }

    function incentivize(
        address _principle,
        address _collateral,
        address _incentiveToken,
        bool _isLend,
        uint _amount,
        uint epoch
    ) internal {
        address[] memory principles = allDynamicData.getDynamicAddressArray(1);
        address[] memory collateral = allDynamicData.getDynamicAddressArray(1);
        address[] memory incentiveToken = allDynamicData.getDynamicAddressArray(
            1
        );

        bool[] memory isLend = allDynamicData.getDynamicBoolArray(1);
        uint[] memory amount = allDynamicData.getDynamicUintArray(1);
        uint[] memory epochs = allDynamicData.getDynamicUintArray(1);

        principles[0] = _principle;
        collateral[0] = _collateral;
        incentiveToken[0] = _incentiveToken;
        isLend[0] = _isLend;
        amount[0] = _amount;
        epochs[0] = epoch;

        IERC20(_incentiveToken).approve(address(incentivesContract), 1000e18);
        deal(_incentiveToken, address(this), _amount, false);
        incentivesContract.incentivizePair(
            principles,
            incentiveToken,
            isLend,
            amount,
            epochs
        );
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
    function createLoan(
        address _borrower,
        address lender,
        address principle,
        address collateral
    ) internal returns (address) {
        vm.startPrank(_borrower);
        deal(principle, lender, 1000e18, false);
        deal(collateral, _borrower, 1000e18, false);
        IERC20(collateral).approve(address(DBOFactoryContract), 100e18);
        bool[] memory oraclesActivated = allDynamicData.getDynamicBoolArray(1);
        uint[] memory ltvs = allDynamicData.getDynamicUintArray(1);
        uint[] memory ratio = allDynamicData.getDynamicUintArray(1);
        uint[] memory ratioLenders = allDynamicData.getDynamicUintArray(1);
        uint[] memory ltvsLenders = allDynamicData.getDynamicUintArray(1);
        bool[] memory oraclesActivatedLenders = allDynamicData
            .getDynamicBoolArray(1);

        address[] memory acceptedPrinciples = allDynamicData
            .getDynamicAddressArray(1);
        address[] memory acceptedCollaterals = allDynamicData
            .getDynamicAddressArray(1);
        address[] memory oraclesCollateral = allDynamicData
            .getDynamicAddressArray(1);
        address[] memory oraclesPrinciples = allDynamicData
            .getDynamicAddressArray(1);

        // set the values for the loan
        ltvs[0] = 5000;
        acceptedPrinciples[0] = principle;
        acceptedCollaterals[0] = collateral;
        oraclesActivated[0] = true;

        oraclesPrinciples[0] = DebitaChainlinkOracle;
        oraclesCollateral[0] = DebitaChainlinkOracle;

        address borrowOrderAddress = DBOFactoryContract.createBorrowOrder(
            oraclesActivated,
            ltvs,
            1400,
            864000,
            acceptedPrinciples,
            collateral,
            false,
            0,
            oraclesPrinciples,
            ratio,
            DebitaChainlinkOracle,
            40e18
        );

        vm.stopPrank();

        vm.startPrank(lender);
        IERC20(principle).approve(address(DLOFactoryContract), 100e18);
        ltvsLenders[0] = 5000;
        ratioLenders[0] = 5e17;
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
            principle,
            oraclesCollateral,
            ratioLenders,
            DebitaChainlinkOracle,
            5e18
        );
        vm.stopPrank();
        vm.startPrank(connector);

        address[] memory lendOrders = allDynamicData.getDynamicAddressArray(1);
        uint[] memory lendAmountPerOrder = allDynamicData.getDynamicUintArray(
            1
        );
        uint[] memory porcentageOfRatioPerLendOrder = allDynamicData
            .getDynamicUintArray(1);
        address[] memory principles = allDynamicData.getDynamicAddressArray(1);
        uint[] memory indexForPrinciple_BorrowOrder = allDynamicData
            .getDynamicUintArray(1);
        uint[] memory indexForCollateral_LendOrder = allDynamicData
            .getDynamicUintArray(1);
        uint[] memory indexPrinciple_LendOrder = allDynamicData
            .getDynamicUintArray(1);

        lendOrders[0] = lendOrderAddress;
        lendAmountPerOrder[0] = 5e18;

        porcentageOfRatioPerLendOrder[0] = 10000;
        principles[0] = principle;

        // match
        address loan = DebitaV3AggregatorContract.matchOffersV3(
            lendOrders,
            lendAmountPerOrder,
            porcentageOfRatioPerLendOrder,
            borrowOrderAddress,
            principles,
            indexForPrinciple_BorrowOrder,
            indexForCollateral_LendOrder,
            indexPrinciple_LendOrder
        );

        DebitaV3LoanContract = DebitaV3Loan(loan);
        vm.stopPrank();
    }
}
