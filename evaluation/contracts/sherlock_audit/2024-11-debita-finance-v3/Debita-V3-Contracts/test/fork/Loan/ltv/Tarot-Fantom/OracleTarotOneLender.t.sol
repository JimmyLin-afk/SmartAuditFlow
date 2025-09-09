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
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

import {DynamicData} from "../../../../interfaces/getDynamicData.sol";
// import ERC20
import {ERC20Mock} from "@openzeppelin/contracts/mocks/token/ERC20Mock.sol";
import {DebitaV3Loan} from "@contracts/DebitaV3Loan.sol";
import {DebitaIncentives} from "@contracts/DebitaIncentives.sol";
import {VotingEscrow} from "@aerodrome/VotingEscrow.sol";
import {DutchAuction_veNFT} from "@contracts/auctions/Auction.sol";
import {DebitaChainlink} from "@contracts/oracles/DebitaChainlink.sol";
import {DebitaPyth} from "@contracts/oracles/DebitaPyth.sol";
import {MixOracle} from "@contracts/oracles/MixOracle/MixOracle.sol";
import {TarotPriceOracle} from "@contracts/oracles/MixOracle/TarotOracle/TarotPriceOracle.sol";

contract DebitaAggregatorTest is Test, DynamicData {
    VotingEscrow public ABIERC721Contract;
    veNFTAerodrome public receiptContract;
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
    DBOImplementation public BorrowOrder;
    MixOracle public DebitaMixOracle;

    address DebitaPythOracle;

    address veEQUAL = 0x8313f3551C4D3984FfbaDFb42f780D0c8763Ce94;
    address EQUAL = 0x3Fd3A0c85B70754eFc07aC9Ac0cbBDCe664865A6;
    address USDC = 0x2F733095B80A04b38b0D10cC884524a3d09b836a;
    address AEROFEED = 0x4EC5970fC728C5f65ba413992CD5fF6FD70fcfF0;
    address EQUALPAIR = 0x3d6c56f6855b7Cc746fb80848755B0a9c3770122;
    address wFTM = 0x21be370D5312f44cB42ce377BC9b8a0cEF1A4C83;
    address borrower = address(0x02);
    address lender = address(this);

    uint receiptID;

    function setUp() public {
        receiptContract = new veNFTAerodrome(veEQUAL, EQUAL);
        ABIERC721Contract = VotingEscrow(veEQUAL);
        allDynamicData = new DynamicData();
        ownershipsContract = new Ownerships();
        incentivesContract = new DebitaIncentives();
        DBOImplementation borrowOrderImplementation = new DBOImplementation();
        DBOFactoryContract = new DBOFactory(address(borrowOrderImplementation));
        DLOImplementation proxyImplementation = new DLOImplementation();
        DLOFactoryContract = new DLOFactory(address(proxyImplementation));
        auctionFactoryDebitaContract = new auctionFactoryDebita();
        AEROContract = ERC20Mock(EQUAL);
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
        deal(EQUAL, lender, 1000e18, false);
        deal(EQUAL, borrower, 1000e18, false);

        setOracles();

        vm.startPrank(borrower);

        IERC20(EQUAL).approve(address(DBOFactoryContract), 100e18);

        bool[] memory oraclesActivated = allDynamicData.getDynamicBoolArray(1);
        uint[] memory ltvs = allDynamicData.getDynamicUintArray(1);
        uint[] memory ratio = allDynamicData.getDynamicUintArray(1);

        address[] memory acceptedPrinciples = allDynamicData
            .getDynamicAddressArray(1);
        address[] memory acceptedCollaterals = allDynamicData
            .getDynamicAddressArray(1);
        address[] memory oraclesPrinciples = allDynamicData
            .getDynamicAddressArray(1);

        oraclesPrinciples[0] = address(DebitaMixOracle);
        acceptedPrinciples[0] = EQUAL;
        acceptedCollaterals[0] = address(EQUAL);
        oraclesActivated[0] = true;
        ltvs[0] = 5000;
        address borrowOrderAddress = DBOFactoryContract.createBorrowOrder(
            oraclesActivated,
            ltvs,
            1400,
            864000,
            acceptedPrinciples,
            EQUAL,
            false,
            receiptID,
            oraclesPrinciples,
            ratio,
            address(DebitaMixOracle),
            100e18
        );
        vm.stopPrank();

        AEROContract.approve(address(DLOFactoryContract), 5e18);
        address lendOrderAddress = DLOFactoryContract.createLendOrder(
            false,
            oraclesActivated,
            false,
            ltvs,
            1000,
            8640000,
            86400,
            acceptedCollaterals,
            EQUAL,
            oraclesPrinciples,
            ratio,
            address(DebitaMixOracle),
            5e18
        );

        LendOrder = DLOImplementation(lendOrderAddress);
        BorrowOrder = DBOImplementation(borrowOrderAddress);
    }

    function testTarotOracle() public {
        DebitaMixOracle.setAttachedTarotPriceOracle(EQUALPAIR);
        vm.warp(block.timestamp + 1201);
        DebitaMixOracle.getThePrice(EQUAL);
    }

    function testDesactivateOracle() public {
        DebitaMixOracle.setAttachedTarotPriceOracle(EQUALPAIR);
        vm.warp(block.timestamp + 1201);

        DebitaMixOracle.pauseStatusPriceId(EQUALPAIR);
        vm.expectRevert();
        DebitaMixOracle.getThePrice(EQUAL);

        DebitaMixOracle.reactivateStatusPriceId(EQUALPAIR);
        DebitaMixOracle.getThePrice(EQUAL);
    }

    function testMatch() public {
        DebitaMixOracle.setAttachedTarotPriceOracle(EQUALPAIR);
        vm.warp(block.timestamp + 1201);
        DebitaMixOracle.getThePrice(EQUAL);
        MatchOffers();
        DebitaV3Loan.LoanData memory _loanData = DebitaV3LoanContract
            .getLoanData();

        assertEq(_loanData._acceptedOffers[0].ratio, 5e17);
    }

    function MatchOffers() internal {
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
        lendAmountPerOrder[0] = 5e18;
        porcentageOfRatioPerLendOrder[0] = 10000;
        principles[0] = EQUAL;

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
        TarotPriceOracle tarotORacle = new TarotPriceOracle();

        DebitaPyth oracle2 = new DebitaPyth(
            0xff1a0f4744e8582DF1aE09D5611b887B6a12925C,
            address(this)
        );

        oracle2.setPriceFeeds(
            0x21be370D5312f44cB42ce377BC9b8a0cEF1A4C83,
            0x5c6c0d2386e3352356c3ab84434fafb5ea067ac2678a38a338c4a69ddc4bdb0c
        );

        DebitaMixOracle = new MixOracle(address(tarotORacle), address(oracle2));
        DebitaV3AggregatorContract.setOracleEnabled(
            address(DebitaMixOracle),
            true
        );
        DebitaV3AggregatorContract.setOracleEnabled(address(oracle2), true);
        DebitaPythOracle = address(oracle2);
    }
    function calculateInterest(uint index) internal returns (uint) {
        return DebitaV3LoanContract.calculateInterestToPay(index);
    }
}
