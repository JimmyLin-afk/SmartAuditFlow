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

import { TellerV2SolMock } from "../../../../contracts/mock/TellerV2SolMock.sol";
import { LenderCommitmentForwarderMock } from "../../../../contracts/mock/LenderCommitmentForwarderMock.sol";
import { MarketRegistryMock } from "../../../../contracts/mock/MarketRegistryMock.sol";

import { AavePoolAddressProviderMock } from "../../../../contracts/mock/aave/AavePoolAddressProviderMock.sol";
import { AavePoolMock } from "../../../../contracts/mock/aave/AavePoolMock.sol";

import { LenderCommitmentForwarder_G2 } from "../../../../contracts/LenderCommitmentForwarder/LenderCommitmentForwarder_G2.sol";
import { LenderCommitmentForwarder_G3 } from "../../../../contracts/LenderCommitmentForwarder/LenderCommitmentForwarder_G3.sol";

import { PaymentType, PaymentCycleType } from "../../../../contracts/libraries/V2Calculations.sol";

import { FlashRolloverLoan_G1 } from "../../../../contracts/LenderCommitmentForwarder/extensions/FlashRolloverLoan_G1.sol";
import { FlashRolloverLoan_G2 } from "../../../../contracts/LenderCommitmentForwarder/extensions/FlashRolloverLoan_G2.sol";

import "lib/forge-std/src/console.sol";

contract FlashRolloverLoan_Integration_Test is Testable {
    constructor() {}

    User private borrower;
    User private lender;
    User private marketOwner;

    AavePoolMock aavePoolMock;
    AavePoolAddressProviderMock aavePoolAddressProvider;
    FlashRolloverLoan_G2 flashRolloverLoan;
    TellerV2 tellerV2;
    WethMock wethMock;
    ILenderCommitmentForwarder lenderCommitmentForwarder;
    IMarketRegistry marketRegistry;

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

        flashRolloverLoan = new FlashRolloverLoan_G2(
            address(tellerV2),
            address(lenderCommitmentForwarder),
            address(aavePoolAddressProvider)
        );
    }

    function test_flashRollover() public {
        address lendingToken = address(wethMock);

        //initial loan - need to pay back 1 weth + 0.1 weth (interest) to the lender
        uint256 marketId = 1;
        uint256 principalAmount = 1e18;
        uint32 duration = 365 days;
        uint16 interestRate = 1000;

        //wethMock.transfer(address(flashRolloverLoan), 100);

        vm.prank(address(borrower));
        uint256 loanId = tellerV2.submitBid(
            lendingToken,
            marketId,
            principalAmount,
            duration,
            interestRate,
            "",
            address(borrower)
        );

        vm.prank(address(lender));
        wethMock.approve(address(tellerV2), 5e18);

        vm.prank(address(lender));
        (
            uint256 amountToProtocol,
            uint256 amountToMarketplace,
            uint256 amountToBorrower
        ) = tellerV2.lenderAcceptBid(loanId);

        vm.warp(365 days + 1);

        uint256 commitmentPrincipalAmount = 150 * 1e16; //1.50 weth

        ILenderCommitmentForwarder.Commitment
            memory commitment = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: commitmentPrincipalAmount,
                expiration: uint32(block.timestamp + 1 days),
                maxDuration: duration,
                minInterestRate: interestRate,
                collateralTokenAddress: address(0),
                collateralTokenId: 0,
                maxPrincipalPerCollateralAmount: 0,
                collateralTokenType: ILenderCommitmentForwarder
                    .CommitmentCollateralType
                    .NONE,
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

        //should get 0.45  weth   from accepting this commitment  during the rollover process

        FlashRolloverLoan_G1.AcceptCommitmentArgs memory _acceptCommitmentArgs = FlashRolloverLoan_G1
            .AcceptCommitmentArgs({
                commitmentId: commitmentId,
                principalAmount: commitmentPrincipalAmount,
                collateralAmount: 0,
                collateralTokenId: 0,
                collateralTokenAddress: address(0),
                interestRate: interestRate,
                loanDuration: duration
                // merkleProof: new bytes32[](0)
            });

        ///approve forwarders

        vm.prank(address(marketOwner));
        ITellerV2Context(address(tellerV2)).setTrustedMarketForwarder(
            marketId,
            address(lenderCommitmentForwarder)
        );

        //borrower AND lender  approves the lenderCommitmentForwarder as trusted

        vm.prank(address(lender));
        ITellerV2Context(address(tellerV2)).approveMarketForwarder(
            marketId,
            address(lenderCommitmentForwarder)
        );

        vm.prank(address(borrower));
        ITellerV2Context(address(tellerV2)).approveMarketForwarder(
            marketId,
            address(lenderCommitmentForwarder)
        );

        //borrower must approve the extension
        vm.prank(address(borrower));
        IExtensionsContext(address(lenderCommitmentForwarder)).addExtension(
            address(flashRolloverLoan)
        );

        //how do we calc how much to flash ??
        uint256 flashLoanAmount = 110 * 1e16;

        uint256 borrowerAmount = 0;

        vm.expectEmit(true, false, false, false);
        emit RolloverLoanComplete(address(borrower), 0, 0, 0);

        vm.prank(address(borrower));
        flashRolloverLoan.rolloverLoanWithFlash(
            loanId,
            flashLoanAmount,
            borrowerAmount,
            _acceptCommitmentArgs
        );
    }
}

contract User {}
