pragma solidity ^0.8.0;

import {Test, console} from "forge-std/Test.sol";
import {veNFTAerodrome} from "@contracts/Non-Fungible-Receipts/veNFTS/Aerodrome/Receipt-veNFT.sol";

import {veNFTVault} from "@contracts/Non-Fungible-Receipts/veNFTS/Aerodrome/veNFTAerodrome.sol";
import {DBOFactory} from "@contracts/DebitaBorrowOffer-Factory.sol";
import {DBOImplementation} from "@contracts/DebitaBorrowOffer-Implementation.sol";
import {DLOFactory} from "@contracts/DebitaLendOfferFactory.sol";
import {DLOImplementation} from "@contracts/DebitaLendOffer-Implementation.sol";
import {DebitaV3Aggregator} from "@contracts/DebitaV3Aggregator.sol";
import {Ownerships} from "@contracts/DebitaLoanOwnerships.sol";
import {auctionFactoryDebita} from "@contracts/auctions/AuctionFactory.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {DynamicData} from "../../../interfaces/getDynamicData.sol";
import "@openzeppelin/contracts/token/ERC721/IERC721.sol";

// import ERC20
import {DebitaV3Loan} from "@contracts/DebitaV3Loan.sol";
import {DebitaIncentives} from "@contracts/DebitaIncentives.sol";
import {DebitaV3Loan} from "@contracts/DebitaV3Loan.sol";
import {DebitaChainlink} from "@contracts/oracles/DebitaChainlink.sol";
import {DebitaPyth} from "@contracts/oracles/DebitaPyth.sol";
import {TaxTokensReceipts} from "@contracts/Non-Fungible-Receipts/TaxTokensReceipts/TaxTokensReceipt.sol";
import {ERC20Mock} from "@openzeppelin/contracts/mocks/token/ERC20Mock.sol";

contract TestTaxTokensReceipts is Test {
    DBOFactory public dbFactory;
    DLOFactory public dlFactory;
    Ownerships public ownershipsContract;
    DebitaIncentives public incentivesContract;
    DebitaV3Aggregator public aggregator;
    auctionFactoryDebita public auctionFactoryDebitaContract;
    DynamicData public allDynamicData;
    DebitaV3Loan public DebitaV3LoanContract;
    DLOImplementation public LendOrder;
    DBOImplementation public BorrowOrder;
    // simulating a taxable token --> fBomb
    // taxable token has to extent TaxTokensReceipts interface

    address fBomb = 0x940181a94A35A4569E4529A3CDfB74e38FD98631;
    address buyer = 0x548D484F5d768a497A1919a57f643AEF403FE3BE;
    ERC20Mock token;

    TaxTokensReceipts public receiptContract;
    function setUp() public {
        allDynamicData = new DynamicData();
        ownershipsContract = new Ownerships();
        incentivesContract = new DebitaIncentives();
        DBOImplementation borrowOrderImplementation = new DBOImplementation();
        dbFactory = new DBOFactory(address(borrowOrderImplementation));
        DLOImplementation proxyImplementation = new DLOImplementation();
        dlFactory = new DLOFactory(address(proxyImplementation));
        auctionFactoryDebitaContract = new auctionFactoryDebita();

        token = ERC20Mock(fBomb);
        DebitaV3Loan loanInstance = new DebitaV3Loan();

        aggregator = new DebitaV3Aggregator(
            address(dbFactory),
            address(dlFactory),
            address(incentivesContract),
            address(ownershipsContract),
            address(auctionFactoryDebitaContract),
            address(loanInstance)
        );
        receiptContract = new TaxTokensReceipts(
            fBomb,
            address(dbFactory),
            address(dlFactory),
            address(aggregator)
        );
        deal(fBomb, buyer, 100e18, true);
    }

    function testDepositAndWithdraw() public {
        vm.startPrank(buyer);
        token.approve(address(receiptContract), 1000e18);
        uint tokenID = receiptContract.deposit(100e18);
        assertEq(token.balanceOf(address(receiptContract)), 100e18);
        assertEq(token.balanceOf(address(buyer)), 0);

        assertEq(receiptContract.balanceOf(buyer), 1);
        receiptContract.withdraw(tokenID);
        assertEq(token.balanceOf(address(receiptContract)), 0);
        assertEq(token.balanceOf(address(buyer)), 100e18);
        assertEq(receiptContract.balanceOf(buyer), 0);
        vm.stopPrank();
    }

    function testTransferReceipts() public {
        vm.startPrank(buyer);
        token.approve(address(receiptContract), 1000e18);
        uint tokenID = receiptContract.deposit(100e18);
        assertEq(receiptContract.balanceOf(buyer), 1);

        address newOwner = 0x5F35576Ae82553209224d85Bbe9657565ab16a4f;
        vm.expectRevert("TaxTokensReceipts: Debita not involved");
        receiptContract.transferFrom(buyer, newOwner, tokenID);

        // execute transfer function
        vm.expectRevert("TaxTokensReceipts: Debita not involved");
        receiptContract.safeTransferFrom(buyer, newOwner, tokenID);

        vm.stopPrank();
    }

    function testCreateOrders() public {
        vm.startPrank(buyer);
        token.approve(address(receiptContract), 1000e18);
        uint tokenID = receiptContract.deposit(100e18);
        assertEq(receiptContract.balanceOf(buyer), 1);
        createBorrowOrder(
            5e17,
            4000,
            tokenID,
            864000,
            1,
            fBomb,
            address(receiptContract),
            buyer
        );

        createLendOrder(
            5e17,
            4000,
            864000,
            864000,
            100e18,
            fBomb,
            address(receiptContract),
            buyer
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
        IERC20(principle).approve(address(dlFactory), 1000e18);
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

        address lendOrderAddress = dlFactory.createLendOrder(
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
        return lendOrderAddress;
    }
    function createBorrowOrder(
        uint _ratio,
        uint maxInterest,
        uint tokenId,
        uint time,
        uint amountCollateral,
        address principle,
        address collateral,
        address borrower
    ) internal {
        IERC721(collateral).approve(address(dbFactory), tokenId);
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

        address borrowOrderAddress = dbFactory.createBorrowOrder(
            oraclesActivated,
            ltvs,
            maxInterest,
            time,
            acceptedPrinciples,
            collateral,
            true,
            tokenId,
            oraclesPrinciples,
            ratio,
            address(0x0),
            amountCollateral
        );
        BorrowOrder = DBOImplementation(borrowOrderAddress);
    }

    function matchOffers() public {
        address[] memory lendOrders = allDynamicData.getDynamicAddressArray(1);
        uint[] memory lendAmountPerOrder = allDynamicData.getDynamicUintArray(
            1
        );
        uint[] memory porcentageOfRatioPerLendOrder = allDynamicData
            .getDynamicUintArray(1);
        address[] memory principles = allDynamicData.getDynamicAddressArray(2);
        uint[] memory indexForPrinciple_BorrowOrder = allDynamicData
            .getDynamicUintArray(1);
        uint[] memory indexForCollateral_LendOrder = allDynamicData
            .getDynamicUintArray(1);
        uint[] memory indexPrinciple_LendOrder = allDynamicData
            .getDynamicUintArray(1);

        lendOrders[0] = address(LendOrder);
        lendAmountPerOrder[0] = 25e17;
        porcentageOfRatioPerLendOrder[0] = 10000;
        principles[0] = fBomb;

        // 0.1e18 --> 1e18 collateral

        address loan = aggregator.matchOffersV3(
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
