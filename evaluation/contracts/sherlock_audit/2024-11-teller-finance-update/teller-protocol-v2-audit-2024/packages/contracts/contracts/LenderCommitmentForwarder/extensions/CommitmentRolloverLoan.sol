// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Contracts
import "@openzeppelin/contracts-upgradeable/utils/AddressUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/token/ERC20/IERC20Upgradeable.sol";

// Interfaces
import "../../interfaces/ITellerV2.sol";
import "../../interfaces/IProtocolFee.sol";
import "../../interfaces/ITellerV2Storage.sol";
import "../../interfaces/IMarketRegistry.sol";
import "../../interfaces/ILenderCommitmentForwarder.sol";
import "../../interfaces/ICommitmentRolloverLoan.sol";
import "../../libraries/NumbersLib.sol";

contract CommitmentRolloverLoan is ICommitmentRolloverLoan {
    using AddressUpgradeable for address;
    using NumbersLib for uint256;

    /// @custom:oz-upgrades-unsafe-allow state-variable-immutable
    ITellerV2 public immutable TELLER_V2;
    /// @custom:oz-upgrades-unsafe-allow state-variable-immutable
    ILenderCommitmentForwarder public immutable LENDER_COMMITMENT_FORWARDER;

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor(address _tellerV2, address _lenderCommitmentForwarder) {
        TELLER_V2 = ITellerV2(_tellerV2);
        LENDER_COMMITMENT_FORWARDER = ILenderCommitmentForwarder(
            _lenderCommitmentForwarder
        );
    }

    /**
     * @notice Allows a borrower to rollover a loan to a new commitment.
     * @param _loanId The ID of the existing loan.
     * @param _rolloverAmount The amount to rollover.
     * @param _commitmentArgs Arguments for the commitment to accept.
     * @return newLoanId_ The ID of the new loan created by accepting the commitment.
     */
    function rolloverLoan(
        uint256 _loanId,
        uint256 _rolloverAmount,
        AcceptCommitmentArgs calldata _commitmentArgs
    ) external returns (uint256 newLoanId_) {
        address borrower = TELLER_V2.getLoanBorrower(_loanId);
        require(borrower == msg.sender, "CommitmentRolloverLoan: not borrower");

        // Get lending token and balance before
        IERC20Upgradeable lendingToken = IERC20Upgradeable(
            TELLER_V2.getLoanLendingToken(_loanId)
        );
        uint256 balanceBefore = lendingToken.balanceOf(address(this));

        if (_rolloverAmount > 0) {
            //accept funds from the borrower to this contract
            lendingToken.transferFrom(borrower, address(this), _rolloverAmount);
        }

        // Accept commitment and receive funds to this contract
        newLoanId_ = _acceptCommitment(_commitmentArgs);

        // Calculate funds received
        uint256 fundsReceived = lendingToken.balanceOf(address(this)) -
            balanceBefore;

        // Approve TellerV2 to spend funds and repay loan
        lendingToken.approve(address(TELLER_V2), fundsReceived);
        TELLER_V2.repayLoanFull(_loanId);

        uint256 fundsRemaining = lendingToken.balanceOf(address(this)) -
            balanceBefore;

        if (fundsRemaining > 0) {
            lendingToken.transfer(borrower, fundsRemaining);
        }
    }

    /**
     * @notice Calculates the amount for loan rollover, determining if the borrower owes or receives funds.
     * @param _loanId The ID of the loan to calculate the rollover amount for.
     * @param _commitmentArgs Arguments for the commitment.
     * @param _timestamp The timestamp for when the calculation is executed.
     * @return _amount The calculated amount, positive if borrower needs to send funds and negative if they will receive funds.
     */
    function calculateRolloverAmount(
        uint256 _loanId,
        AcceptCommitmentArgs calldata _commitmentArgs,
        uint256 _timestamp
    ) external view returns (int256 _amount) {
        Payment memory repayAmountOwed = TELLER_V2.calculateAmountOwed(
            _loanId,
            _timestamp
        );

        _amount +=
            int256(repayAmountOwed.principal) +
            int256(repayAmountOwed.interest);

        uint256 _marketId = _getMarketIdForCommitment(
            _commitmentArgs.commitmentId
        );
        uint16 marketFeePct = _getMarketFeePct(_marketId);
        uint16 protocolFeePct = _getProtocolFeePct();

        uint256 commitmentPrincipalRequested = _commitmentArgs.principalAmount;
        uint256 amountToMarketplace = commitmentPrincipalRequested.percent(
            marketFeePct
        );
        uint256 amountToProtocol = commitmentPrincipalRequested.percent(
            protocolFeePct
        );

        uint256 amountToBorrower = commitmentPrincipalRequested -
            amountToProtocol -
            amountToMarketplace;

        _amount -= int256(amountToBorrower);
    }

    /**
     * @notice Internally accepts a commitment via the `LENDER_COMMITMENT_FORWARDER`.
     * @param _commitmentArgs Arguments required to accept a commitment.
     * @return bidId_ The ID of the bid associated with the accepted commitment.
     */
    function _acceptCommitment(AcceptCommitmentArgs calldata _commitmentArgs)
        internal
        returns (uint256 bidId_)
    {
        bytes memory responseData = address(LENDER_COMMITMENT_FORWARDER)
            .functionCall(
                abi.encodePacked(
                    abi.encodeWithSelector(
                        ILenderCommitmentForwarder
                            .acceptCommitmentWithRecipient
                            .selector,
                        _commitmentArgs.commitmentId,
                        _commitmentArgs.principalAmount,
                        _commitmentArgs.collateralAmount,
                        _commitmentArgs.collateralTokenId,
                        _commitmentArgs.collateralTokenAddress,
                        address(this),
                        _commitmentArgs.interestRate,
                        _commitmentArgs.loanDuration
                    ),
                    msg.sender
                )
            );

        (bidId_) = abi.decode(responseData, (uint256));
    }

    /**
     * @notice Retrieves the market ID associated with a given commitment.
     * @param _commitmentId The ID of the commitment for which to fetch the market ID.
     * @return The ID of the market associated with the provided commitment.
     */
    function _getMarketIdForCommitment(uint256 _commitmentId)
        internal
        view
        returns (uint256)
    {
        return LENDER_COMMITMENT_FORWARDER.getCommitmentMarketId(_commitmentId);
    }

    /**
     * @notice Fetches the marketplace fee percentage for a given market ID.
     * @param _marketId The ID of the market for which to fetch the fee percentage.
     * @return The marketplace fee percentage for the provided market ID.
     */
    function _getMarketFeePct(uint256 _marketId)
        internal
        view
        returns (uint16)
    {
        address _marketRegistryAddress = ITellerV2Storage(address(TELLER_V2))
            .marketRegistry();

        return
            IMarketRegistry(_marketRegistryAddress).getMarketplaceFee(
                _marketId
            );
    }

    /**
     * @notice Fetches the protocol fee percentage from the Teller V2 protocol.
     * @return The protocol fee percentage as defined in the Teller V2 protocol.
     */
    function _getProtocolFeePct() internal view returns (uint16) {
        return IProtocolFee(address(TELLER_V2)).protocolFee();
    }
}
