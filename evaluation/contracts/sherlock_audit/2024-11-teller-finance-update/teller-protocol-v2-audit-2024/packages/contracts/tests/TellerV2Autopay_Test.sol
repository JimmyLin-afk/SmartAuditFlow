// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { Testable } from "./Testable.sol";

import { TellerV2Autopay } from "../contracts/TellerV2Autopay.sol";
import { MarketRegistry } from "../contracts/MarketRegistry.sol";
import { ReputationManager } from "../contracts/ReputationManager.sol";

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

import "../contracts/TellerV2Storage.sol";

import "../contracts/interfaces/IMarketRegistry.sol";
import "../contracts/interfaces/IReputationManager.sol";

import "../contracts/EAS/TellerAS.sol";

import "../contracts/mock/WethMock.sol";

import "../contracts/mock/TellerV2SolMock.sol";
import "../contracts/mock/MarketRegistryMock.sol";
import "../contracts/interfaces/IWETH.sol";
import "../contracts/interfaces/ITellerV2Autopay.sol";

import { PaymentType } from "../contracts/libraries/V2Calculations.sol";

contract TellerV2Autopay_Test is Testable, TellerV2Autopay {
    User private marketOwner;
    User private borrower;
    User private lender;
    User private contractOwner;

    WethMock wethMock;

    address marketRegistry;

    constructor() TellerV2Autopay(address(new TellerV2SolMock())) {
        marketRegistry = address(new MarketRegistryMock());
        TellerV2SolMock(address(tellerV2)).setMarketRegistry(marketRegistry);
    }

    function setUp() public {
        wethMock = new WethMock();

        marketOwner = new User(
            address(this),
            address(tellerV2),
            address(wethMock)
        );
        borrower = new User(
            address(this),
            address(tellerV2),
            address(wethMock)
        );
        lender = new User(address(this), address(tellerV2), address(wethMock));

        contractOwner = new User(
            address(this),
            address(tellerV2),
            address(wethMock)
        );

        contractOwner.initialize(5, address(contractOwner));
    }

    function setAutoPayEnabled_before() public {
        uint256 marketplaceId = 1;
        marketOwner.createMarketWithinRegistry(
            address(marketRegistry),
            8000,
            7000,
            5000,
            500,
            false,
            false,
            PaymentType.EMI,
            "uri://"
        );

        uint256 bidId = borrower.submitBid(
            address(wethMock),
            marketplaceId,
            100,
            4000,
            300,
            "ipfs://",
            address(borrower)
        );
    }

    function test_setAutoPayEnabled() public {
        setAutoPayEnabled_before();

        uint256 bidId = 0;

        borrower.enableAutoPay(bidId, true);

        assertEq(
            loanAutoPayEnabled[bidId],
            true,
            "Autopay not enabled after setAutoPayEnabled"
        );
    }

    function test_setAutopayFee() public {
        _setAutopayFee(4);
        assertEq(4, getAutopayFee(), "Auto pay fee not set");
    }

    function test_setAutopayFeeOnlyOwner() public {
        User user = new User(
            address(this),
            address(tellerV2),
            address(wethMock)
        );
        try user.setAutopayFee(4) {
            fail("Auto pay fee set by non owner");
        } catch Error(string memory reason) {
            assertEq(
                reason,
                "Ownable: caller is not the owner",
                "Should not be able to set autopay fee"
            );
        } catch {
            fail("Unknown error");
        }
    }

    function autoPayLoanMinimum_before() public {
        uint256 marketplaceId = 1;

        uint256 lenderNewBalance = 500000;

        payable(address(lender)).transfer(lenderNewBalance);

        //lender approves for acceptBid
        lender.depositToWeth(lenderNewBalance);
        lender.approveWeth(address(tellerV2), lenderNewBalance);

        uint256 bidId = borrower.submitBid(
            address(wethMock),
            marketplaceId,
            1000,
            4000,
            300,
            "ipfs://",
            address(borrower)
        );

        borrower.enableAutoPay(bidId, true);

        uint256 lenderBalance = ERC20(address(wethMock)).balanceOf(
            address(lender)
        );

        lender.acceptBid(bidId);

        uint256 borrowerNewBalance = 500000;

        payable(address(borrower)).transfer(borrowerNewBalance);

        //borrower approve to do repay
        borrower.depositToWeth(borrowerNewBalance);
        borrower.approveWeth(address(this), borrowerNewBalance);
    }

    function test_autoPayLoanMinimum() public {
        autoPayLoanMinimum_before();

        uint256 bidId = 0;

        uint256 lenderBalanceBefore = ERC20(address(wethMock)).balanceOf(
            address(lender)
        );
        uint256 borrowerBalanceBefore = ERC20(address(wethMock)).balanceOf(
            address(borrower)
        );

        lender.autoPayLoanMinimum(bidId);

        uint256 lenderBalanceAfter = ERC20(address(wethMock)).balanceOf(
            address(lender)
        );
        uint256 borrowerBalanceAfter = ERC20(address(wethMock)).balanceOf(
            address(borrower)
        );

        uint256 lenderBalanceDelta = lenderBalanceAfter - lenderBalanceBefore;

        assertEq(
            lenderBalanceDelta,
            2,
            "lender did not receive the auto pay charge"
        );

        uint256 borrowerBalanceDelta = borrowerBalanceBefore -
            borrowerBalanceAfter;

        assertEq(borrowerBalanceDelta, 4002, "borrower did not autopay");
    }

    function getEstimatedMinimumPayment(uint256 _bidId, uint256 _timestamp)
        public
        override
        returns (uint256 _amount)
    {
        return 4000; //stub this for this test since there is not a good way to fast forward timestamp
    }
}

contract User {
    address public immutable tellerV2;
    address public immutable wethMock;
    address public immutable tellerV2Autopay;

    constructor(address _tellerV2Autopay, address _tellerV2, address _wethMock)
    {
        tellerV2Autopay = _tellerV2Autopay;
        tellerV2 = _tellerV2;
        wethMock = _wethMock;
    }

    function enableAutoPay(uint256 bidId, bool enabled) public {
        ITellerV2Autopay(tellerV2Autopay).setAutoPayEnabled(bidId, enabled);
    }

    function createMarketWithinRegistry(
        address marketRegistry,
        uint32 _paymentCycleDuration,
        uint32 _paymentDefaultDuration,
        uint32 _bidExpirationTime,
        uint16 _feePercent,
        bool _requireLenderAttestation,
        bool _requireBorrowerAttestation,
        PaymentType _paymentType,
        string calldata _uri
    ) public {
        IMarketRegistry(marketRegistry).createMarket(
            address(this),
            _paymentCycleDuration,
            _paymentDefaultDuration,
            _bidExpirationTime,
            _feePercent,
            _requireLenderAttestation,
            _requireBorrowerAttestation,
            _paymentType,
            PaymentCycleType.Seconds,
            _uri
        );
    }

    function autoPayLoanMinimum(uint256 bidId) public {
        ITellerV2Autopay(tellerV2Autopay).autoPayLoanMinimum(bidId);
    }

    function submitBid(
        address _lendingToken,
        uint256 _marketplaceId,
        uint256 _principal,
        uint32 _duration,
        uint16 _APR,
        string calldata _metadataURI,
        address _receiver
    ) public returns (uint256) {
        return
            ITellerV2(tellerV2).submitBid(
                _lendingToken,
                _marketplaceId,
                _principal,
                _duration,
                _APR,
                _metadataURI,
                _receiver
            );
    }

    function initialize(uint16 _newFee, address _newOwner) public {
        ITellerV2Autopay(tellerV2Autopay).initialize(_newFee, _newOwner);
    }

    function setAutopayFee(uint16 _newFee) public {
        ITellerV2Autopay(tellerV2Autopay).setAutopayFee(_newFee);
    }

    function acceptBid(uint256 _bidId) public {
        ITellerV2(tellerV2).lenderAcceptBid(_bidId);
    }

    function depositToWeth(uint256 amount) public {
        IWETH(wethMock).deposit{ value: amount }();
    }

    function approveWeth(address to, uint256 amount) public {
        ERC20(wethMock).approve(to, amount);
    }

    receive() external payable {}
}
