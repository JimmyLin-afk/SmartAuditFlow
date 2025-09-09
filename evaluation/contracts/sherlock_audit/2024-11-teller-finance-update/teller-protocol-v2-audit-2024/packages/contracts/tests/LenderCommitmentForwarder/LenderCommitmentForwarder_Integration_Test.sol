import { Testable } from "../Testable.sol";

import "../../contracts/interfaces/ICommitmentRolloverLoan.sol";
import "../../contracts/interfaces/ILenderCommitmentForwarder.sol";
import "../../contracts/interfaces/ITellerV2Context.sol";
import "../../contracts/interfaces/ICollateralManager.sol";

import "../integration/IntegrationTestHelpers.sol";

import { WethMock } from "../../contracts/mock/WethMock.sol";
import { TestERC721Token } from "../tokens/TestERC721Token.sol";

import { TellerV2SolMock } from "../../contracts/mock/TellerV2SolMock.sol";
//import { LenderCommitmentForwarder } from "../../contracts/LenderCommitmentForwarder/LenderCommitmentForwarder.sol";
import { LenderCommitmentForwarder_G2 } from "../../contracts/LenderCommitmentForwarder/LenderCommitmentForwarder_G2.sol";
import { LenderCommitmentForwarder_G3 } from "../../contracts/LenderCommitmentForwarder/LenderCommitmentForwarder_G3.sol";
import { MarketRegistryMock } from "../../contracts/mock/MarketRegistryMock.sol";

import { PaymentType, PaymentCycleType } from "../../contracts/libraries/V2Calculations.sol";

import { Collateral, CollateralType } from "../../contracts/interfaces/escrow/ICollateralEscrowV1.sol";

import "lib/forge-std/src/console.sol";

/*
contract CommitmentRolloverLoanMock is CommitmentRolloverLoan {
    constructor(address _tellerV2, address _lenderCommitmentForwarder)
        CommitmentRolloverLoan(_tellerV2, _lenderCommitmentForwarder)
    {}

    function acceptCommitment(AcceptCommitmentArgs calldata _commitmentArgs)
        public
        returns (uint256 bidId_)
    {
        bidId_ = super._acceptCommitment(_commitmentArgs);
    }
}*/

contract LenderCommitmentForwarder_Integration_Test is Testable {
    constructor() {}

    User private borrower;
    User private lender;
    User private marketOwner;

    ILenderCommitmentForwarder lenderCommitmentForwarder;
    TellerV2 tellerV2;

    IMarketRegistry marketRegistry;

    WethMock wethMock;
    TestERC721Token erc721Token;

    function setUp() public {
        borrower = new User();
        lender = new User();
        marketOwner = new User();

        tellerV2 = IntegrationTestHelpers.deployIntegrationSuite();

        marketRegistry = IMarketRegistry(tellerV2.marketRegistry());

        lenderCommitmentForwarder = ILenderCommitmentForwarder(
            tellerV2.lenderCommitmentForwarder()
        );

        wethMock = new WethMock();
        erc721Token = new TestERC721Token("squig", "squig");

        erc721Token.mint(address(borrower));

        uint32 _paymentCycleDuration = uint32(1 days);
        uint32 _paymentDefaultDuration = uint32(5 days);
        uint32 _bidExpirationTime = uint32(7 days);
        uint16 _feePercent = 900;
        PaymentType _paymentType = PaymentType.EMI;
        PaymentCycleType _paymentCycleType = PaymentCycleType.Seconds;

        vm.prank(address(marketOwner));
        uint256 marketId = marketRegistry.createMarket(
            address(marketOwner),
            _paymentCycleDuration,
            _paymentDefaultDuration,
            _bidExpirationTime,
            _feePercent,
            false,
            false,
            _paymentType,
            _paymentCycleType,
            "uri"
        );

        wethMock.deposit{ value: 100e18 }();
        wethMock.transfer(address(lender), 5e18);
        wethMock.transfer(address(borrower), 5e18);
    }

    function test_accept_commitment_with_collateral() public {
        address lendingToken = address(wethMock);

        //initial loan - need to pay back 1 weth + 0.1 weth (interest) to the lender
        uint256 marketId = 1;

        /*   {
       
        uint256 principalAmount = 1e18;
        uint32 duration = 365 days;
        uint16 interestRate = 1000;

       // wethMock.transfer(address(commitmentRolloverLoan), 100);

       

        address collateralManager = address(tellerV2.collateralManager());
       // erc721Token.setApprovalForAll(address(collateralManager), true);
       
    
        Collateral[] memory collateral = new Collateral[](1);

        collateral[0]._collateralType = CollateralType.ERC721; //ERC721
        collateral[0]._tokenId = 0;
        collateral[0]._amount = 1;
        collateral[0]._collateralAddress = address(erc721Token);

   
        vm.prank(address(borrower));
        uint256 loanId = tellerV2.submitBid(
            lendingToken,
            marketId,
            principalAmount,
            duration,
            interestRate,
            "",
            address(borrower),
            collateral
        );

 
        vm.prank(address(lender));
        (
            uint256 amountToProtocol,
            uint256 amountToMarketplace,
            uint256 amountToBorrower
        ) = tellerV2.lenderAcceptBid(loanId);
       }*/

        address collateralManager = address(tellerV2.collateralManager());

        vm.prank(address(borrower));
        erc721Token.approve(address(collateralManager), 0);

        vm.prank(address(lender));
        wethMock.approve(address(tellerV2), 2e18);

        vm.warp(365 days + 1);

        uint256 commitmentPrincipalAmount = 50 * 1e16; //0.50 weth

        ILenderCommitmentForwarder.Commitment
            memory commitment = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: commitmentPrincipalAmount,
                expiration: uint32(block.timestamp + 1 days),
                maxDuration: 365 days,
                minInterestRate: 1000,
                collateralTokenAddress: address(erc721Token),
                collateralTokenId: 0,
                maxPrincipalPerCollateralAmount: 1e20,
                collateralTokenType: ILenderCommitmentForwarder
                    .CommitmentCollateralType
                    .ERC721,
                lender: address(lender),
                marketId: marketId,
                principalTokenAddress: lendingToken
            });

        address[] memory _borrowerAddressList;

        vm.prank(address(lender));
        uint256 commitmentId = lenderCommitmentForwarder.createCommitment(
            commitment,
            _borrowerAddressList
        );

        uint256 principalAmount = 100;
        address recipient = address(borrower);
        uint32 loanDuration = 1 days;

        {
            vm.prank(address(marketOwner));
            ITellerV2Context(address(tellerV2)).setTrustedMarketForwarder(
                marketId,
                address(lenderCommitmentForwarder)
            );

            //borrower AND lender  approves the lenderCommitmentForwarder as trusted

            vm.prank(address(borrower));
            ITellerV2Context(address(tellerV2)).approveMarketForwarder(
                marketId,
                address(lenderCommitmentForwarder)
            );

            vm.prank(address(lender));
            ITellerV2Context(address(tellerV2)).approveMarketForwarder(
                marketId,
                address(lenderCommitmentForwarder)
            );
        }

        vm.prank(address(borrower));
        //accept commitment and make sure the collateral is moved
        uint256 bidId = LenderCommitmentForwarder_G2(
            address(lenderCommitmentForwarder)
        ).acceptCommitment(
                commitmentId,
                principalAmount,
                1, //collateral amount
                0, //collateral token id
                address(erc721Token), //collateral address
                //   recipient,
                1000, //interest rate
                loanDuration
            );

        address ownerOfNft = erc721Token.ownerOf(0);

        address escrowForLoan = ICollateralManager(collateralManager).getEscrow(
            bidId
        );

        assertEq(
            ownerOfNft,
            address(escrowForLoan),
            "Nft not moved to collateral escrow"
        );
    }
}

contract User {}
