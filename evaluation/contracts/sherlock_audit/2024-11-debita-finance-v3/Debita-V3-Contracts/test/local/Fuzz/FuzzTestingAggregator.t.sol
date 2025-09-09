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
    function testFuzz_(
        uint _lendAmountPerOrder,
        uint _porcentageOfRatioPerLendOrder
    ) public {
        vm.assume(
            _porcentageOfRatioPerLendOrder <= 10000 &&
                _porcentageOfRatioPerLendOrder > 0
        );

        createBorrowOrder(
            2e18,
            1000,
            86400,
            1000e18,
            AERO,
            AERO,
            address(this)
        );

        createLendOrder(
            1e18,
            1000,
            86400,
            864000,
            1000e18,
            AERO,
            AERO,
            address(this)
        );
        MatchOffers(1e18, _porcentageOfRatioPerLendOrder);
    }

    function testFuzzSpecificRatio(
        uint _lendAmountPerOrder,
        uint porcentageOfRatio
    ) public {
        vm.assume(porcentageOfRatio <= 10000 && porcentageOfRatio > 0);
        vm.assume(porcentageOfRatio < 4900 || porcentageOfRatio > 5100);
        createBorrowOrder(
            2e18,
            1000,
            86400,
            1000e18,
            AERO,
            AERO,
            address(this)
        );

        createLendOrder(
            4e18,
            1000,
            86400,
            864000,
            1000e18,
            AERO,
            AERO,
            address(this)
        );
        MatchOffers(1e18, porcentageOfRatio);
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

        address[] memory acceptedCollaterals = allDynamicData
            .getDynamicAddressArray(1);
        address[] memory oraclesPrinciples = allDynamicData
            .getDynamicAddressArray(1);

        ratio[0] = _ratio;
        oraclesPrinciples[0] = address(0x0);
        acceptedCollaterals[0] = collateral;
        oraclesActivated[0] = false;
        ltvs[0] = 0;

        address borrowOrderAddress = DBOFactoryContract.createBorrowOrder(
            oraclesActivated,
            ltvs,
            maxInterest,
            time,
            acceptedCollaterals,
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

    function createLendOrder(
        uint _ratio,
        uint maxInterest,
        uint minTime,
        uint maxTime,
        uint amountPrinciple,
        address principle,
        address collateral,
        address lender
    ) internal {
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
            principle,
            oraclesPrinciples,
            ratio,
            address(0x0),
            amountPrinciple
        );
        LendOrder = DLOImplementation(lendOrderAddress);
    }

    function MatchOffers(
        uint _lendAmountPerOrder,
        uint _porcentageOfRatioPerLendOrder
    ) internal {
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
        indexForPrinciple_BorrowOrder[0] = 0;
        indexForCollateral_LendOrder[0] = 0;
        indexPrinciple_LendOrder[0] = 0;

        lendOrders[0] = address(LendOrder);
        lendAmountPerOrder[0] = _lendAmountPerOrder;
        porcentageOfRatioPerLendOrder[0] = _porcentageOfRatioPerLendOrder;
        principles[0] = AERO;

        vm.expectRevert("Invalid ratio");
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
