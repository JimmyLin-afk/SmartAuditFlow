pragma solidity >=0.8.0 <0.9.0;
// SPDX-License-Identifier: MIT

import "./interfaces/ITellerV2.sol";
import "./interfaces/ITellerV2Autopay.sol";

import "./libraries/NumbersLib.sol";

import "@openzeppelin/contracts-upgradeable/utils/AddressUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/token/ERC20/IERC20Upgradeable.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
import { Payment } from "./TellerV2Storage.sol";

/**
 * @dev Helper contract to autopay loans
 */
contract TellerV2Autopay is OwnableUpgradeable, ITellerV2Autopay {
    using SafeERC20 for ERC20;
    using NumbersLib for uint256;

    ITellerV2 public immutable tellerV2;

    //bidId => enabled
    mapping(uint256 => bool) public loanAutoPayEnabled;

    // Autopay fee set for automatic loan payments
    uint16 private _autopayFee;

    /**
     * @notice This event is emitted when a loan is autopaid.
     * @param bidId The id of the bid/loan which was repaid.
     * @param msgsender The account that called the method
     */
    event AutoPaidLoanMinimum(uint256 indexed bidId, address indexed msgsender);

    /**
     * @notice This event is emitted when loan autopayments are enabled or disabled.
     * @param bidId The id of the bid/loan.
     * @param enabled Whether the autopayments are enabled or disabled
     */
    event AutoPayEnabled(uint256 indexed bidId, bool enabled);

    /**
     * @notice This event is emitted when the autopay fee has been updated.
     * @param newFee The new autopay fee set.
     * @param oldFee The previously set autopay fee.
     */
    event AutopayFeeSet(uint16 newFee, uint16 oldFee);

    constructor(address _protocolAddress) {
        tellerV2 = ITellerV2(_protocolAddress);
    }

    /**
     * @notice Initialized the proxy.
     * @param _fee The fee collected for automatic payment processing.
     * @param _owner The address of the ownership to be transferred to.
     */
    function initialize(uint16 _fee, address _owner) external initializer {
        _transferOwnership(_owner);
        _setAutopayFee(_fee);
    }

    /**
     * @notice Let the owner of the contract set a new autopay fee.
     * @param _newFee The new autopay fee to set.
     */
    function setAutopayFee(uint16 _newFee) public virtual onlyOwner {
        _setAutopayFee(_newFee);
    }

    function _setAutopayFee(uint16 _newFee) internal {
        // Skip if the fee is the same
        if (_newFee == _autopayFee) return;
        uint16 oldFee = _autopayFee;
        _autopayFee = _newFee;
        emit AutopayFeeSet(_newFee, oldFee);
    }

    /**
     * @notice Returns the current autopay fee.
     */
    function getAutopayFee() public view virtual returns (uint16) {
        return _autopayFee;
    }

    /**
     * @notice Function for a borrower to enable or disable autopayments
     * @param _bidId The id of the bid to cancel.
     * @param _autoPayEnabled boolean for allowing autopay on a loan
     */
    function setAutoPayEnabled(uint256 _bidId, bool _autoPayEnabled) external {
        require(
            _msgSender() == tellerV2.getLoanBorrower(_bidId),
            "Only the borrower can set autopay"
        );

        loanAutoPayEnabled[_bidId] = _autoPayEnabled;

        emit AutoPayEnabled(_bidId, _autoPayEnabled);
    }

    /**
     * @notice Function for a minimum autopayment to be performed on a loan
     * @param _bidId The id of the bid to repay.
     */
    function autoPayLoanMinimum(uint256 _bidId) external {
        require(
            loanAutoPayEnabled[_bidId],
            "Autopay is not enabled for that loan"
        );

        address lendingToken = ITellerV2(tellerV2).getLoanLendingToken(_bidId);
        address borrower = ITellerV2(tellerV2).getLoanBorrower(_bidId);

        uint256 amountToRepayMinimum = getEstimatedMinimumPayment(
            _bidId,
            block.timestamp
        );
        uint256 autopayFeeAmount = amountToRepayMinimum.percent(
            getAutopayFee()
        );

        // Pull lendingToken in from the borrower to this smart contract
        ERC20(lendingToken).safeTransferFrom(
            borrower,
            address(this),
            amountToRepayMinimum + autopayFeeAmount
        );

        // Transfer fee to msg sender
        ERC20(lendingToken).safeTransfer(_msgSender(), autopayFeeAmount);

        // Approve the lendingToken to tellerV2
        ERC20(lendingToken).approve(address(tellerV2), amountToRepayMinimum);

        // Use that lendingToken to repay the loan
        tellerV2.repayLoan(_bidId, amountToRepayMinimum);

        emit AutoPaidLoanMinimum(_bidId, msg.sender);
    }

    function getEstimatedMinimumPayment(uint256 _bidId, uint256 _timestamp)
        public
        virtual
        returns (uint256 _amount)
    {
        Payment memory estimatedPayment = tellerV2.calculateAmountDue(
            _bidId,
            _timestamp
        );

        _amount = estimatedPayment.principal + estimatedPayment.interest;
    }
}
