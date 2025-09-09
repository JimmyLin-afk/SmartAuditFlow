pragma solidity ^0.8.0;

import { Testable } from "../../../Testable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

import { FlashRolloverLoan } from "../../../../contracts/LenderCommitmentForwarder/extensions/FlashRolloverLoan.sol";

import "../../../../contracts/interfaces/ILenderCommitmentForwarder.sol";
import "../../../../contracts/interfaces/IFlashRolloverLoan_G4.sol";

import "../../../integration/IntegrationTestHelpers.sol";

import { WethMock } from "../../../../contracts/mock/WethMock.sol";

import { TellerV2SolMock } from "../../../../contracts/mock/TellerV2SolMock.sol";
import { LenderCommitmentForwarderMock } from "../../../../contracts/mock/LenderCommitmentForwarderMock.sol";
import { MarketRegistryMock } from "../../../../contracts/mock/MarketRegistryMock.sol";

import { AavePoolAddressProviderMock } from "../../../../contracts/mock/aave/AavePoolAddressProviderMock.sol";
import { AavePoolMock } from "../../../../contracts/mock/aave/AavePoolMock.sol";

import { FlashRolloverLoan_G4 } from "../../../../contracts/LenderCommitmentForwarder/extensions/FlashRolloverLoan_G4.sol";

contract FlashRolloverLoanOverride is FlashRolloverLoan_G4 {
    constructor(
        address _tellerV2,
        address _lenderCommitmentForwarder,
        address _aaveAddressProvider
    )
        FlashRolloverLoan_G4(
            _tellerV2, 
            _aaveAddressProvider
        )
    {}

    function acceptCommitment(
        address _lenderCommitmentForwarder,
        address borrower,
        address principalToken,
        FlashRolloverLoan_G4.AcceptCommitmentArgs calldata _commitmentArgs
    ) public returns (uint256 bidId_, uint256 acceptCommitmentAmount_) {
        return
            super._acceptCommitment(_lenderCommitmentForwarder,borrower, principalToken, _commitmentArgs);
    }
}

contract FlashRolloverLoan_G4_Unit_Test is Testable {
    constructor() {}

    User private borrower;
    User private lender;

    AavePoolMock aavePoolMock;
    AavePoolAddressProviderMock aavePoolAddressProvider;
    FlashRolloverLoanOverride flashRolloverLoan;
    TellerV2SolMock tellerV2;
    WethMock wethMock;
    LenderCommitmentForwarderMock lenderCommitmentForwarder;
    MarketRegistryMock marketRegistryMock;

    function setUp() public {
        borrower = new User();
        lender = new User();

        tellerV2 = new TellerV2SolMock();
        wethMock = new WethMock();

        marketRegistryMock = new MarketRegistryMock();

        tellerV2.setMarketRegistry(address(marketRegistryMock));

        lenderCommitmentForwarder = new LenderCommitmentForwarderMock();

        aavePoolAddressProvider = new AavePoolAddressProviderMock(
            "marketId",
            address(this)
        );

        aavePoolMock = new AavePoolMock();

        bytes32 POOL = "POOL";
        aavePoolAddressProvider.setAddress(POOL, address(aavePoolMock));

        wethMock.deposit{ value: 100e18 }();
        wethMock.transfer(address(lender), 5e18);
        wethMock.transfer(address(borrower), 5e18);
        wethMock.transfer(address(lenderCommitmentForwarder), 5e18);

        wethMock.transfer(address(aavePoolMock), 5e18);

        //marketRegistryMock = new MarketRegistryMock();

        flashRolloverLoan = new FlashRolloverLoanOverride(
            address(tellerV2),
            address(lenderCommitmentForwarder),
            address(aavePoolAddressProvider)
        );

        IntegrationTestHelpers.deployIntegrationSuite();
    }

    function test_rolloverLoanWithFlash() public {
        address lendingToken = address(wethMock);
        uint256 marketId = 0;
        uint256 principalAmount = 500;
        uint32 duration = 10 days;
        uint16 interestRate = 100;

        ILenderCommitmentForwarder.Commitment
            memory commitment = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: principalAmount,
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

        lenderCommitmentForwarder.setCommitment(0, commitment);

        FlashRolloverLoan_G4.AcceptCommitmentArgs
            memory commitmentArgs = FlashRolloverLoan_G4.AcceptCommitmentArgs({
                commitmentId: 0,
                principalAmount: principalAmount,
                collateralAmount: 100,
                collateralTokenId: 0,
                collateralTokenAddress: address(0),
                interestRate: interestRate,
                loanDuration: duration,
                merkleProof: new bytes32[](0)
            });

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

        uint256 flashAmount = 500;
        uint256 borrowerAmount = 5; //have to pay the aave fee..

        vm.prank(address(borrower));
        IERC20(lendingToken).approve(address(flashRolloverLoan), 1e18);

        vm.prank(address(borrower));

        flashRolloverLoan.rolloverLoanWithFlash(
            address(lenderCommitmentForwarder),
            loanId,
            flashAmount,
            borrowerAmount,
            commitmentArgs
        );

        bool flashLoanSimpleWasCalled = aavePoolMock.flashLoanSimpleWasCalled();
        assertTrue(
            flashLoanSimpleWasCalled,
            "flashLoanSimpleWasCalled not called"
        );
    }

    function test_rolloverLoanWithFlashAndProof() public {
        address lendingToken = address(wethMock);
        uint256 marketId = 0;
        uint256 principalAmount = 500;
        uint32 duration = 10 days;
        uint16 interestRate = 100;

        ILenderCommitmentForwarder.Commitment
            memory commitment = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: principalAmount,
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

        lenderCommitmentForwarder.setCommitment(0, commitment);

        bytes32[] memory merkleProof = new bytes32[](1);

        FlashRolloverLoan_G4.AcceptCommitmentArgs
            memory commitmentArgs = FlashRolloverLoan_G4.AcceptCommitmentArgs({
                commitmentId: 0,
                principalAmount: principalAmount,
                collateralAmount: 100,
                collateralTokenId: 0,
                collateralTokenAddress: address(0),
                interestRate: interestRate,
                loanDuration: duration,
                merkleProof: merkleProof
            });

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

        uint256 flashAmount = 500;
        uint256 borrowerAmount = 5; //have to pay the aave fee..

        vm.prank(address(borrower));
        IERC20(lendingToken).approve(address(flashRolloverLoan), 1e18);

        vm.prank(address(borrower));

        flashRolloverLoan.rolloverLoanWithFlash(
              address(lenderCommitmentForwarder),
            loanId,
            flashAmount,
            borrowerAmount,
            commitmentArgs
        );

        bool flashLoanSimpleWasCalled = aavePoolMock.flashLoanSimpleWasCalled();
        assertTrue(
            flashLoanSimpleWasCalled,
            "flashLoanSimpleWasCalled not called"
        );
    }

    function test_executeOperation() public {
        address lendingToken = address(wethMock);
        uint256 marketId = 0;
        uint256 principalAmount = 500;
        uint32 duration = 10 days;
        uint16 interestRate = 100;

        wethMock.transfer(address(flashRolloverLoan), 5e18);

        ILenderCommitmentForwarder.Commitment
            memory commitment = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: principalAmount,
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

        lenderCommitmentForwarder.setCommitment(0, commitment);

        FlashRolloverLoan_G4.AcceptCommitmentArgs
            memory commitmentArgs = FlashRolloverLoan_G4.AcceptCommitmentArgs({
                commitmentId: 0,
                principalAmount: principalAmount,
                collateralAmount: 100,
                collateralTokenId: 0,
                collateralTokenAddress: address(0),
                interestRate: interestRate,
                loanDuration: duration,
                merkleProof: new bytes32[](0)
            });

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

        uint256 flashAmount = 500;
        uint256 borrowerAmount = 5; //have to pay the aave fee..

        vm.prank(address(borrower));
        IERC20(lendingToken).approve(address(flashRolloverLoan), 1e18);

        vm.prank(address(aavePoolMock));

        uint256 flashFees = 5;
        address initiator = address(flashRolloverLoan);

        bytes memory flashData = abi.encode(
            IFlashRolloverLoan_G4.RolloverCallbackArgs({
                lenderCommitmentForwarder: address(lenderCommitmentForwarder),
                loanId: loanId,
                borrower: address(borrower),
                borrowerAmount: borrowerAmount,
                acceptCommitmentArgs: abi.encode(commitmentArgs)
            })
        );

        flashRolloverLoan.executeOperation(
            lendingToken,
            flashAmount,
            flashFees,
            initiator,
            flashData
        );

        bool acceptCommitmentWithRecipientWasCalled = lenderCommitmentForwarder
            .acceptCommitmentWithRecipientWasCalled();
        assertTrue(
            acceptCommitmentWithRecipientWasCalled,
            "acceptCommitmentWithRecipient not called"
        );
    }

    function test_executeOperationWithProof() public {
        address lendingToken = address(wethMock);
        uint256 marketId = 0;
        uint256 principalAmount = 500;
        uint32 duration = 10 days;
        uint16 interestRate = 100;

        wethMock.transfer(address(flashRolloverLoan), 5e18);

        ILenderCommitmentForwarder.Commitment
            memory commitment = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: principalAmount,
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

        lenderCommitmentForwarder.setCommitment(0, commitment);

        bytes32[] memory merkleProof = new bytes32[](1);

        FlashRolloverLoan_G4.AcceptCommitmentArgs
            memory commitmentArgs = FlashRolloverLoan_G4.AcceptCommitmentArgs({
                commitmentId: 0,
                principalAmount: principalAmount,
                collateralAmount: 100,
                collateralTokenId: 0,
                collateralTokenAddress: address(0),
                interestRate: interestRate,
                loanDuration: duration,
                merkleProof: merkleProof
            });

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

        uint256 flashAmount = 500;
        uint256 borrowerAmount = 5; //have to pay the aave fee..

        vm.prank(address(borrower));
        IERC20(lendingToken).approve(address(flashRolloverLoan), 1e18);

        vm.prank(address(aavePoolMock));

        uint256 flashFees = 5;
        address initiator = address(flashRolloverLoan);

        bytes memory flashData = abi.encode(
            IFlashRolloverLoan_G4.RolloverCallbackArgs({
                    lenderCommitmentForwarder: address(lenderCommitmentForwarder),
                loanId: loanId,
                borrower: address(borrower),
                borrowerAmount: borrowerAmount,
                acceptCommitmentArgs: abi.encode(commitmentArgs)
            })
        );

        flashRolloverLoan.executeOperation(
            lendingToken,
            flashAmount,
            flashFees,
            initiator,
            flashData
        );

        bool acceptCommitmentWithRecipientAndProofWasCalled = lenderCommitmentForwarder
                .acceptCommitmentWithRecipientAndProofWasCalled();
        assertTrue(
            acceptCommitmentWithRecipientAndProofWasCalled,
            "acceptCommitmentWithRecipientAndProof not called"
        );
    }

    function test_executeOperation_invalid_sender() public {
        address lendingToken = address(wethMock);
        uint256 marketId = 0;
        uint256 principalAmount = 500;
        uint32 duration = 10 days;
        uint16 interestRate = 100;

        ILenderCommitmentForwarder.Commitment
            memory commitment = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: principalAmount,
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

        lenderCommitmentForwarder.setCommitment(0, commitment);

        FlashRolloverLoan_G4.AcceptCommitmentArgs
            memory commitmentArgs = FlashRolloverLoan_G4.AcceptCommitmentArgs({
                commitmentId: 0,
                principalAmount: principalAmount,
                collateralAmount: 100,
                collateralTokenId: 0,
                collateralTokenAddress: address(0),
                interestRate: interestRate,
                loanDuration: duration,
                merkleProof: new bytes32[](0)
            });

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

        uint256 flashAmount = 500;
        uint256 borrowerAmount = 5; //have to pay the aave fee..

        vm.prank(address(borrower));
        IERC20(lendingToken).approve(address(flashRolloverLoan), 1e18);

        vm.prank(address(this));

        uint256 flashFees = 10;
        address initiator = address(this);

        bytes memory flashData = abi.encode(
            IFlashRolloverLoan_G4.RolloverCallbackArgs({
                    lenderCommitmentForwarder: address(lenderCommitmentForwarder),
                loanId: loanId,
                borrower: address(borrower),
                borrowerAmount: borrowerAmount,
                acceptCommitmentArgs: abi.encode(commitmentArgs)
            })
        );

        vm.expectRevert("FlashRolloverLoan: Must be called by FlashLoanPool");

        flashRolloverLoan.executeOperation(
            lendingToken,
            flashAmount,
            flashFees,
            initiator,
            flashData
        );
    }

    function test_executeOperation_invalid_initiator() public {
        address lendingToken = address(wethMock);
        uint256 marketId = 0;
        uint256 principalAmount = 500;
        uint32 duration = 10 days;
        uint16 interestRate = 100;

        ILenderCommitmentForwarder.Commitment
            memory commitment = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: principalAmount,
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

        lenderCommitmentForwarder.setCommitment(0, commitment);

        FlashRolloverLoan_G4.AcceptCommitmentArgs
            memory commitmentArgs = FlashRolloverLoan_G4.AcceptCommitmentArgs({
                commitmentId: 0,
                principalAmount: principalAmount,
                collateralAmount: 100,
                collateralTokenId: 0,
                collateralTokenAddress: address(0),
                interestRate: interestRate,
                loanDuration: duration,
                merkleProof: new bytes32[](0)
            });

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

        uint256 flashAmount = 500;
        uint256 borrowerAmount = 5; //have to pay the aave fee..

        vm.prank(address(borrower));
        IERC20(lendingToken).approve(address(flashRolloverLoan), 1e18);

        vm.prank(address(aavePoolMock));

        uint256 flashFees = 10;
        address initiator = address(this);

        bytes memory flashData = abi.encode(
            IFlashRolloverLoan_G4.RolloverCallbackArgs({
                    lenderCommitmentForwarder: address(lenderCommitmentForwarder),
                loanId: loanId,
                borrower: address(borrower),
                borrowerAmount: borrowerAmount,
                acceptCommitmentArgs: abi.encode(commitmentArgs)
            })
        );

        vm.expectRevert("This contract must be the initiator");

        flashRolloverLoan.executeOperation(
            lendingToken,
            flashAmount,
            flashFees,
            initiator,
            flashData
        );
    }

    function test_rolloverLoan_invalid_caller() public {
        address lendingToken = address(wethMock);
        uint256 marketId = 0;
        uint256 principalAmount = 500;
        uint32 duration = 10 days;
        uint16 interestRate = 100;

        ILenderCommitmentForwarder.Commitment
            memory commitment = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: principalAmount,
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

        lenderCommitmentForwarder.setCommitment(0, commitment);

        FlashRolloverLoan_G4.AcceptCommitmentArgs
            memory commitmentArgs = FlashRolloverLoan_G4.AcceptCommitmentArgs({
                commitmentId: 0,
                principalAmount: principalAmount,
                collateralAmount: 100,
                collateralTokenId: 0,
                collateralTokenAddress: address(0),
                interestRate: interestRate,
                loanDuration: duration,
                merkleProof: new bytes32[](0)
            });

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

        uint256 flashAmount = 0;
        uint256 borrowerAmount = 0;

        vm.prank(address(lender));

        vm.expectRevert("CommitmentRolloverLoan: not borrower");

        flashRolloverLoan.rolloverLoanWithFlash(
              address(lenderCommitmentForwarder),
            loanId,
            flashAmount,
            borrowerAmount,
            commitmentArgs
        );
    }

    function test_calculate_rollover_amount() public {
        address lendingToken = address(wethMock);

        //initial loan - need to pay back 1 weth + 0.1 weth (interest) to the lender
        uint256 marketId = 1;
        uint256 principalAmount = 50000;
        uint32 duration = 365 days;
        uint16 interestRate = 1000;

        //this is the old loan
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

        //why approve so much ?
        vm.prank(address(lender));
        wethMock.approve(address(tellerV2), 5e18);

        vm.prank(address(lender));
        (
            uint256 amountToProtocol,
            uint256 amountToMarketplace,
            uint256 amountToBorrower
        ) = tellerV2.lenderAcceptBid(loanId);

        vm.warp(365 days + 1);

        uint256 newLoanPrincipalAmount = 50000;

        //this is prepping the new loan
        ILenderCommitmentForwarder.Commitment
            memory commitment = ILenderCommitmentForwarder.Commitment({
                maxPrincipal: newLoanPrincipalAmount,
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

        lenderCommitmentForwarder.setCommitment(0, commitment);

        FlashRolloverLoan_G4.AcceptCommitmentArgs
            memory commitmentArgs = FlashRolloverLoan_G4.AcceptCommitmentArgs({
                commitmentId: 0,
                principalAmount: newLoanPrincipalAmount,
                collateralAmount: 0,
                collateralTokenId: 0,
                collateralTokenAddress: address(0),
                interestRate: interestRate,
                loanDuration: duration,
                merkleProof: new bytes32[](0)
            });

        vm.prank(address(borrower));
        IERC20(lendingToken).approve(address(flashRolloverLoan), 1e18);

        //making sure the flashloan premium logic works
        (uint256 flashAmount, int256 borrowerAmount) = flashRolloverLoan
            .calculateRolloverAmount(
                  address(lenderCommitmentForwarder),
                loanId,
                commitmentArgs,
                9,
                block.timestamp
            );

        assertEq(flashAmount, 55000, "invalid flashAmount");
        assertEq(borrowerAmount, -10549, "invalid borrowerAmount");

        (flashAmount, borrowerAmount) = flashRolloverLoan
            .calculateRolloverAmount(
                  address(lenderCommitmentForwarder),
                loanId,
                commitmentArgs,
                0,
                block.timestamp
            );

        assertEq(flashAmount, 55000, "invalid flashAmount");
        assertEq(borrowerAmount, -10500, "invalid borrowerAmount");
    }
}

contract User {}
