import { Testable } from "../../../Testable.sol";

import { FlashRolloverLoan } from "../../../../contracts/LenderCommitmentForwarder/extensions/FlashRolloverLoan.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

import "../../../../contracts/interfaces/IFlashRolloverLoan.sol";
import "../../../../contracts/interfaces/ILenderCommitmentForwarder.sol";
import "../../../../contracts/interfaces/ITellerV2Context.sol";
import "../../../../contracts/interfaces/IExtensionsContext.sol";

import "../../../integration/IntegrationTestHelpers.sol";

import "../../../../contracts/LenderCommitmentForwarder/extensions/ExtensionsContextUpgradeable.sol";

import { WethMock } from "../../../../contracts/mock/WethMock.sol";
import { TestERC721Token } from "../../../tokens/TestERC721Token.sol";

import { TellerV2SolMock } from "../../../../contracts/mock/TellerV2SolMock.sol";
import { LenderCommitmentForwarderMock } from "../../../../contracts/mock/LenderCommitmentForwarderMock.sol";
import { MarketRegistryMock } from "../../../../contracts/mock/MarketRegistryMock.sol";

import { AavePoolAddressProviderMock } from "../../../../contracts/mock/aave/AavePoolAddressProviderMock.sol";
import { AavePoolMock } from "../../../../contracts/mock/aave/AavePoolMock.sol";

import { LenderCommitmentForwarder_G2 } from "../../../../contracts/LenderCommitmentForwarder/LenderCommitmentForwarder_G2.sol";
import { LenderCommitmentForwarder_G3 } from "../../../../contracts/LenderCommitmentForwarder/LenderCommitmentForwarder_G3.sol";

import { PaymentType, PaymentCycleType } from "../../../../contracts/libraries/V2Calculations.sol";

import { FlashRolloverLoan_G4 } from "../../../../contracts/LenderCommitmentForwarder/extensions/FlashRolloverLoan_G4.sol";

import "lib/forge-std/src/console.sol";

contract FlashRolloverLoan_G3_Integration_Test is Testable {
    constructor() {}

    User private borrower;
    User private lender;
    User private marketOwner;

    AavePoolMock aavePoolMock;
    AavePoolAddressProviderMock aavePoolAddressProvider;
    FlashRolloverLoan_G4 flashRolloverLoan;
    TellerV2 tellerV2;
    WethMock wethMock;
    ILenderCommitmentForwarder lenderCommitmentForwarder;
    IMarketRegistry marketRegistry;
    TestERC721Token testNft;

    event RolloverLoanComplete(
        address borrower,
        uint256 originalLoanId,
        uint256 newLoanId,
        uint256 fundsRemaining
    );

    function setUp() public {
        borrower = new User();
        lender = new User();
        marketOwner = new User();

        tellerV2 = IntegrationTestHelpers.deployIntegrationSuite();

        console.logAddress(address(tellerV2));

        marketRegistry = IMarketRegistry(tellerV2.marketRegistry());

        lenderCommitmentForwarder = ILenderCommitmentForwarder(
            tellerV2.lenderCommitmentForwarder()
        );

        aavePoolAddressProvider = new AavePoolAddressProviderMock(
            "marketId",
            address(this)
        );

        aavePoolMock = new AavePoolMock();

        bytes32 POOL = "POOL";
        aavePoolAddressProvider.setAddress(POOL, address(aavePoolMock));

        wethMock = new WethMock();

        testNft = new TestERC721Token("NFT", "NFT");
        testNft.mint(address(borrower));
        testNft.mint(address(borrower));

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

        wethMock.transfer(address(aavePoolMock), 5e18);

        //  wethMock.transfer(address(flashLoanVault), 5e18);

        flashRolloverLoan = new FlashRolloverLoan_G4(
            address(tellerV2), 
            address(aavePoolAddressProvider)
        );
    }

    function test_flashRollover() public {
        address lendingToken = address(wethMock);

        //initial loan - need to pay back 1 weth + 0.1 weth (interest) to the lender

        uint256 principalAmount = 1e18;
        uint32 duration = 365 days;
        uint16 interestRate = 1000;

        //wethMock.transfer(address(flashRolloverLoan), 100);

        vm.prank(address(borrower));
        uint256 loanId = tellerV2.submitBid(
            lendingToken,
            1, //market id
            principalAmount,
            duration,
            interestRate,
            "",
            address(borrower)
        );

        vm.prank(address(lender));
        wethMock.approve(address(tellerV2), 5e18);

        vm.prank(address(lender));
        tellerV2.lenderAcceptBid(loanId);

        vm.warp(365 days + 1);

        uint256 commitmentPrincipalAmount = 150 * 1e16; //1.50 weth

        uint256 tokenIdLeafA = 1;
        uint256 tokenIdLeafB = 3;

        bytes32 merkleLeafA = keccak256(abi.encodePacked(tokenIdLeafA)); //  0xc89efdaa54c0f20c7adf612882df0950f5a951637e0307cdcb4c672f298b8bc6;
        bytes32 merkleLeafB = keccak256(abi.encodePacked(tokenIdLeafB));

        //a merkle root is simply the hash of the hashes of the leaves in the layer above, where the leaves are always sorted alphanumerically.
        //it so happens that the hash of (1) is less than the hash of (3) so we can compute the merkle root manually like this without a sorting function:
        bytes32 merkleRoot = keccak256(
            abi.encodePacked(merkleLeafA, merkleLeafB)
        );

        // should be an nft
        address collateralToken = address(testNft);

        ILenderCommitmentForwarder.Commitment
            memory commitment = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: commitmentPrincipalAmount,
                expiration: uint32(block.timestamp + 1 days),
                maxDuration: duration,
                minInterestRate: interestRate,
                collateralTokenAddress: address(collateralToken),
                collateralTokenId: uint256(merkleRoot),
                maxPrincipalPerCollateralAmount: commitmentPrincipalAmount *
                    1e18,
                collateralTokenType: ILenderCommitmentForwarder
                    .CommitmentCollateralType
                    .ERC721_MERKLE_PROOF,
                lender: address(lender),
                marketId: 1,
                principalTokenAddress: address(lendingToken)
            });

        address[] memory _borrowerAddressList;

        vm.prank(address(lender));
        uint256 commitmentId = lenderCommitmentForwarder.createCommitment(
            commitment,
            _borrowerAddressList
        );

        bytes32[] memory merkleProof = new bytes32[](1);
        merkleProof[0] = merkleLeafB;

        FlashRolloverLoan_G4.AcceptCommitmentArgs
            memory _acceptCommitmentArgs = FlashRolloverLoan_G4
                .AcceptCommitmentArgs({
                    commitmentId: commitmentId,
                    principalAmount: commitmentPrincipalAmount,
                    collateralAmount: 1,
                    collateralTokenId: tokenIdLeafA,
                    collateralTokenAddress: address(collateralToken),
                    interestRate: 1000,
                    loanDuration: 365 days,
                    merkleProof: merkleProof
                });

        {
            vm.prank(address(marketOwner));
            ITellerV2Context(address(tellerV2)).setTrustedMarketForwarder(
                1,
                address(lenderCommitmentForwarder)
            );

            //borrower AND lender  approves the lenderCommitmentForwarder as trusted

            vm.prank(address(lender));
            ITellerV2Context(address(tellerV2)).approveMarketForwarder(
                1,
                address(lenderCommitmentForwarder)
            );

            vm.prank(address(borrower));
            ITellerV2Context(address(tellerV2)).approveMarketForwarder(
                1,
                address(lenderCommitmentForwarder)
            );

            //borrower must approve the extension
            vm.prank(address(borrower));
            IExtensionsContext(address(lenderCommitmentForwarder)).addExtension(
                    address(flashRolloverLoan)
                );

            address collateralManager = address(tellerV2.collateralManager());

            vm.prank(address(borrower));
            testNft.setApprovalForAll(address(collateralManager), true);
        }
        //how do we calc how much to flash ??
      //  uint256 flashLoanAmount = 110 * 1e16;

        // uint256 borrowerAmount = 0;

        vm.expectEmit(true, false, false, false);
        emit RolloverLoanComplete(address(borrower), 0, 0, 0);

        vm.prank(address(borrower));
        flashRolloverLoan.rolloverLoanWithFlash(
             address(lenderCommitmentForwarder),
            loanId,
            110 * 1e16, // flash loan amt 
            0,
            _acceptCommitmentArgs
        );
    }
}

contract User {}
