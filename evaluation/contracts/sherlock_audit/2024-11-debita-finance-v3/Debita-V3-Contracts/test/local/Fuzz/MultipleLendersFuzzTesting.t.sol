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

contract FuzzTestingOrders is Test {
    DBOFactory public DBOFactoryContract;
    DLOFactory public DLOFactoryContract;
    Ownerships public ownershipsContract;
    DebitaIncentives public incentivesContract;
    DebitaV3Aggregator public DebitaV3AggregatorContract;
    auctionFactoryDebita public auctionFactoryDebitaContract;
    DynamicData public allDynamicData;
    DebitaV3Loan public DebitaV3LoanContract;

    DLOImplementation public LendOrder;
    DBOImplementation public BorrowOrder;
    ERC20Mock public AEROContract;
    address AERO;

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
        AERO = address(AEROContract);
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
    }

    function testFuzzWithTwoOrders(
        uint _lendAmountPerOrder,
        uint _secondLendAmountPerOrder,
        uint porcentageOfRatio,
        uint secondPorcentageOfRatio
    ) public {
        vm.assume(
            porcentageOfRatio <= 10000 &&
                porcentageOfRatio > 0 &&
                secondPorcentageOfRatio <= 10000 &&
                secondPorcentageOfRatio > 0
        );
        vm.assume(_lendAmountPerOrder > 0 && _secondLendAmountPerOrder > 0);
        vm.assume(
            _lendAmountPerOrder < 5e17 && _secondLendAmountPerOrder < 1e18
        );

        address principle = AERO;
        address collateral = AERO;
        address lender = address(0x02);
        address secondLender = address(0x01);
        address borrower = address(0x03);

        uint lendAmountPerOrder = _lendAmountPerOrder;
        uint secondLendAmountPerOrder = _secondLendAmountPerOrder;

        uint porcentageOfRatioPerLendOrder = porcentageOfRatio;
        uint secondPorcentageOfRatioPerLendOrder = secondPorcentageOfRatio;

        address[] memory lendOrders = allDynamicData.getDynamicAddressArray(2);

        createBorrowOrder(75e17, 1000, 86400, 100e18, AERO, AERO, borrower);

        lendOrders[0] = createLendOrder(
            5e17,
            1000,
            86400,
            864000,
            100e18,
            principle,
            collateral,
            secondLender
        );

        lendOrders[1] = createLendOrder(
            1e18,
            1000,
            86400,
            864000,
            100e18,
            principle,
            collateral,
            secondLender
        );

        MatchOffers(
            lendAmountPerOrder,
            secondLendAmountPerOrder,
            10000,
            10000,
            lendOrders,
            principle
        );
    }

    function testFuzzCreateLendOrdersWithHigherInterest(
        uint _lendAmountPerOrder,
        uint _secondLendAmountPerOrder,
        uint porcentageOfRatio,
        uint secondPorcentageOfRatio
    ) public {
        vm.assume(
            porcentageOfRatio <= 10000 &&
                porcentageOfRatio > 0 &&
                secondPorcentageOfRatio <= 10000 &&
                secondPorcentageOfRatio > 0
        );
        vm.assume(_lendAmountPerOrder > 0 && _secondLendAmountPerOrder > 0);
        vm.assume(
            _lendAmountPerOrder < 5e17 && _secondLendAmountPerOrder < 1e18
        );

        address principle = AERO;
        address collateral = AERO;
        address lender = address(0x02);
        address secondLender = address(0x01);
        address borrower = address(0x03);

        uint lendAmountPerOrder = _lendAmountPerOrder;
        uint secondLendAmountPerOrder = _secondLendAmountPerOrder;

        address[] memory lendOrders = allDynamicData.getDynamicAddressArray(2);

        createBorrowOrder(75e17, 1000, 86400, 100e18, AERO, AERO, borrower);

        lendOrders[0] = createLendOrder(
            75e17,
            2000,
            86400,
            864000,
            100e18,
            principle,
            collateral,
            secondLender
        );

        lendOrders[1] = createLendOrder(
            75e17,
            2000,
            86400,
            864000,
            100e18,
            principle,
            collateral,
            secondLender
        );

        MatchOffers(
            lendAmountPerOrder,
            secondLendAmountPerOrder,
            10000,
            10000,
            lendOrders,
            principle
        );
    }

    function createLendOrder(
        uint _ratio,
        uint maxInterest,
        uint minTime,
        uint maxTime,
        uint amountPrinciple,
        address principle,
        address collateral,
        address lender
    ) internal returns (address) {
        deal(principle, lender, amountPrinciple, false);
        IERC20(principle).approve(address(DLOFactoryContract), 1000e18);
        bool[] memory oraclesActivated = allDynamicData.getDynamicBoolArray(1);
        uint[] memory ltvs = allDynamicData.getDynamicUintArray(1);
        uint[] memory ratio = allDynamicData.getDynamicUintArray(1);

        address[] memory acceptedCollaterals = allDynamicData
            .getDynamicAddressArray(1);
        address[] memory oraclesPrinciples = allDynamicData
            .getDynamicAddressArray(1);

        ratio[0] = _ratio;
        oraclesPrinciples[0] = address(0x0);
        acceptedCollaterals[0] = collateral;
        oraclesActivated[0] = false;
        ltvs[0] = 0;

        address lendOrderAddress = DLOFactoryContract.createLendOrder(
            false,
            oraclesActivated,
            false,
            ltvs,
            maxInterest,
            maxTime,
            minTime,
            acceptedCollaterals,
            collateral,
            oraclesPrinciples,
            ratio,
            address(0x0),
            amountPrinciple
        );
        return lendOrderAddress;
    }

    function createBorrowOrder(
        uint _ratio,
        uint maxInterest,
        uint time,
        uint amountCollateral,
        address principle,
        address collateral,
        address borrower
    ) internal {
        deal(collateral, borrower, amountCollateral, false);
        IERC20(collateral).approve(address(DBOFactoryContract), 1000e18);
        bool[] memory oraclesActivated = allDynamicData.getDynamicBoolArray(1);
        uint[] memory ltvs = allDynamicData.getDynamicUintArray(1);
        uint[] memory ratio = allDynamicData.getDynamicUintArray(1);

        address[] memory acceptedPrinciples = allDynamicData
            .getDynamicAddressArray(1);
        address[] memory oraclesPrinciples = allDynamicData
            .getDynamicAddressArray(1);

        ratio[0] = _ratio;
        oraclesPrinciples[0] = address(0x0);
        acceptedPrinciples[0] = principle;
        oraclesActivated[0] = false;
        ltvs[0] = 0;

        address borrowOrderAddress = DBOFactoryContract.createBorrowOrder(
            oraclesActivated,
            ltvs,
            maxInterest,
            time,
            acceptedPrinciples,
            collateral,
            false,
            0,
            oraclesPrinciples,
            ratio,
            address(0x0),
            amountCollateral
        );
        BorrowOrder = DBOImplementation(borrowOrderAddress);
    }

    function MatchOffers(
        uint _lendAmountPerOrder,
        uint _secondAmountPerOrder,
        uint _porcentageOfRatioPerLendOrder,
        uint _porcentageOfRatioPerSecondLendOrder,
        address[] memory lendOrders,
        address principle
    ) internal {
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

        lendAmountPerOrder[0] = _lendAmountPerOrder;
        lendAmountPerOrder[1] = _secondAmountPerOrder;
        porcentageOfRatioPerLendOrder[0] = _porcentageOfRatioPerLendOrder;
        porcentageOfRatioPerLendOrder[1] = _porcentageOfRatioPerSecondLendOrder;
        principles[0] = principle;

        vm.expectRevert();
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
    }
}
