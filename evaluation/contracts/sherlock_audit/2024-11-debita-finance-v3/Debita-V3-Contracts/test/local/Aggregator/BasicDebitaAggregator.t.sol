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

contract DebitaAggregatorTest is Test, DynamicData {
    DBOFactory public DBOFactoryContract;
    DLOFactory public DLOFactoryContract;
    Ownerships public ownershipsContract;
    DebitaIncentives public incentivesContract;
    DebitaV3Aggregator public DebitaV3AggregatorContract;
    auctionFactoryDebita public auctionFactoryDebitaContract;
    DynamicData public allDynamicData;

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

        deal(AERO, address(this), 1000e18, false);
        IERC20(AERO).approve(address(DBOFactoryContract), 1000e18);
        IERC20(AERO).approve(address(DLOFactoryContract), 1000e18);

        bool[] memory oraclesActivated = allDynamicData.getDynamicBoolArray(1);
        uint[] memory ltvs = allDynamicData.getDynamicUintArray(1);
        uint[] memory ratio = allDynamicData.getDynamicUintArray(1);

        address[] memory acceptedPrinciples = allDynamicData
            .getDynamicAddressArray(1);
        address[] memory oraclesPrinciples = allDynamicData
            .getDynamicAddressArray(1);

        ratio[0] = 1e18;
        oraclesPrinciples[0] = address(0x0);
        acceptedPrinciples[0] = AERO;
        oraclesActivated[0] = false;
        ltvs[0] = 0;

        address borrowOrderAddress = DBOFactoryContract.createBorrowOrder(
            oraclesActivated,
            ltvs,
            1000,
            864000,
            acceptedPrinciples,
            AERO,
            false,
            0,
            oraclesPrinciples,
            ratio,
            address(0x0),
            10e18
        );

        address lendOrderAddress = DLOFactoryContract.createLendOrder(
            false,
            oraclesActivated,
            false,
            ltvs,
            1000,
            8640000,
            86400,
            acceptedPrinciples,
            AERO,
            oraclesPrinciples,
            ratio,
            address(0x0),
            5e18
        );

        LendOrder = DLOImplementation(lendOrderAddress);
        BorrowOrder = DBOImplementation(borrowOrderAddress);
    }

    function testMatchOffers() public {
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
        lendAmountPerOrder[0] = 3e18;
        porcentageOfRatioPerLendOrder[0] = 10000;
        principles[0] = AERO;

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
    function testMatchOffersAndCheckParams() public {
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
        lendAmountPerOrder[0] = 3e18;
        porcentageOfRatioPerLendOrder[0] = 10000;
        principles[0] = AERO;

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

        DebitaV3Loan loanContract = DebitaV3Loan(loan);
        assertEq(
            loanContract.getLoanData()._acceptedOffers[0].principleAmount,
            3e18
        );
        assertEq(loanContract.getLoanData()._acceptedOffers[0].apr, 1000);

        assertEq(
            loanContract.getLoanData()._acceptedOffers[0].maxDeadline,
            8640000 + loanContract.getLoanData().startedAt
        );
        assertEq(loanContract.getLoanData().initialDuration, 864000);
    }
    function testMatchOffersAndPayBack() public {
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
        lendAmountPerOrder[0] = 3e18;
        porcentageOfRatioPerLendOrder[0] = 10000;
        principles[0] = AERO;

        uint balanceBefore = IERC20(AERO).balanceOf(address(this));
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
        uint balanceAfter = IERC20(AERO).balanceOf(address(this));
        assertEq(balanceAfter, balanceBefore + 3e18);
        DebitaV3Loan loanContract = DebitaV3Loan(loan);
        uint[] memory indexes = allDynamicData.getDynamicUintArray(1);
        indexes[0] = 0;

        IERC20(AERO).approve(loan, 4e18);
        loanContract.payDebt(indexes);
        uint balanceBeforeClaim = IERC20(AERO).balanceOf(address(this));
        loanContract.claimDebt(0);
        uint balanceAfterClaim = IERC20(AERO).balanceOf(address(this));
        uint principleAmount = loanContract
            .getLoanData()
            ._acceptedOffers[0]
            .principleAmount;
        uint anualInterest = (principleAmount *
            loanContract.getLoanData()._acceptedOffers[0].apr) / 10000;
        uint activeTime = (loanContract.getLoanData().initialDuration * 1000) /
            10000;
        uint interest = (anualInterest * activeTime) / 31536000;
        uint feeInterest = (interest * 1500) / 10000;

        assertEq(
            balanceAfterClaim,
            balanceBeforeClaim + interest + principleAmount - feeInterest
        );

        uint balanceBeforeClaimCollateral = IERC20(AERO).balanceOf(
            address(this)
        );
        loanContract.claimCollateralAsBorrower(indexes);
        uint balanceAfterClaimCollateral = IERC20(AERO).balanceOf(
            address(this)
        );
        assertEq(
            balanceAfterClaimCollateral,
            balanceBeforeClaimCollateral + 3e18
        );
    }
}
