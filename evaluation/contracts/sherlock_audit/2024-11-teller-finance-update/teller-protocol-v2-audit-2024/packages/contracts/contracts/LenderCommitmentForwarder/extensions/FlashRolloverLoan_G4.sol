// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Contracts
import "@openzeppelin/contracts-upgradeable/utils/AddressUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/token/ERC20/IERC20Upgradeable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

// Interfaces
import "../../interfaces/ITellerV2.sol";
import "../../interfaces/IProtocolFee.sol";
import "../../interfaces/ITellerV2Storage.sol";
import "../../interfaces/IMarketRegistry.sol";
import "../../interfaces/ILenderCommitmentForwarder.sol";
import "../../interfaces/IFlashRolloverLoan_G4.sol";
import "../../libraries/NumbersLib.sol";

import { IPool } from "../../interfaces/aave/IPool.sol";
import { IFlashLoanSimpleReceiver } from "../../interfaces/aave/IFlashLoanSimpleReceiver.sol";
import { IPoolAddressesProvider } from "../../interfaces/aave/IPoolAddressesProvider.sol";

contract FlashRolloverLoan_G4 is IFlashLoanSimpleReceiver, IFlashRolloverLoan_G4 {
    using AddressUpgradeable for address;
    using NumbersLib for uint256;

    /// @custom:oz-upgrades-unsafe-allow state-variable-immutable
    ITellerV2 public immutable TELLER_V2;
    /// @custom:oz-upgrades-unsafe-allow state-variable-immutable
    

    address public immutable POOL_ADDRESSES_PROVIDER;

    event RolloverLoanComplete(
        address borrower,
        uint256 originalLoanId,
        uint256 newLoanId,
        uint256 fundsRemaining
    );

    struct AcceptCommitmentArgs {
        uint256 commitmentId;
        uint256 principalAmount;
        uint256 collateralAmount;
        uint256 collateralTokenId;
        address collateralTokenAddress;
        uint16 interestRate;
        uint32 loanDuration;
        bytes32[] merkleProof; //empty array if not used
    }

    /**
     *
     * @notice Initializes the FlashRolloverLoan with necessary contract addresses.
     *
     * @dev Using a custom OpenZeppelin upgrades tag. Ensure the constructor logic is safe for upgrades.
     *
     * @param _tellerV2 The address of the TellerV2 contract.
     * @param _lenderCommitmentForwarder The address of the LenderCommitmentForwarder contract.
     * @param _poolAddressesProvider The address of the PoolAddressesProvider.
     */
    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor(
        address _tellerV2,
        address _poolAddressesProvider
    ) {
        TELLER_V2 = ITellerV2(_tellerV2);
        POOL_ADDRESSES_PROVIDER = _poolAddressesProvider;
    }

    modifier onlyFlashLoanPool() {
        require(
            msg.sender == address(POOL()),
            "FlashRolloverLoan: Must be called by FlashLoanPool"
        );

        _;
    }

    /**
     *
     * @notice Allows the borrower to rollover their existing loan using a flash loan mechanism.
     *         The borrower might also provide an additional amount during the rollover.
     *
     * @dev The function first verifies that the caller is the borrower of the loan.
     *      It then optionally transfers the additional amount specified by the borrower.
     *      A flash loan is then taken from the pool to facilitate the rollover and
     *      a callback is executed for further operations.
     *
     * @param _loanId Identifier of the existing loan to be rolled over.
     * @param _flashLoanAmount Amount of flash loan to be borrowed for the rollover.
     * @param _borrowerAmount Additional amount that the borrower may want to add during rollover.
     * @param _acceptCommitmentArgs Commitment arguments that might be necessary for internal operations.
     *
     * @return newLoanId_ Identifier of the new loan post rollover.
     */
    function rolloverLoanWithFlash(
        address _lenderCommitmentForwarder,
        uint256 _loanId,
        uint256 _flashLoanAmount,
        uint256 _borrowerAmount, //an additional amount borrower may have to add
        AcceptCommitmentArgs calldata _acceptCommitmentArgs
    ) external returns (uint256 newLoanId_) {
        address borrower = TELLER_V2.getLoanBorrower(_loanId);
        require(borrower == msg.sender, "CommitmentRolloverLoan: not borrower");

        // Get lending token and balance before
        address lendingToken = TELLER_V2.getLoanLendingToken(_loanId);

        if (_borrowerAmount > 0) {
            IERC20(lendingToken).transferFrom(
                borrower,
                address(this),
                _borrowerAmount
            );
        }

        // Call 'Flash' on the vault to borrow funds and call tellerV2FlashCallback
        // This ultimately calls executeOperation
        IPool(POOL()).flashLoanSimple(
            address(this),
            lendingToken,
            _flashLoanAmount,
            abi.encode(
                RolloverCallbackArgs({
                    lenderCommitmentForwarder :_lenderCommitmentForwarder,
                    loanId: _loanId,
                    borrower: borrower,
                    borrowerAmount: _borrowerAmount,
                    acceptCommitmentArgs: abi.encode(_acceptCommitmentArgs)
                })
            ),
            0 //referral code
        );
    }

    /**
     *
     * @notice Callback function that is triggered by Aave during the flash loan process.
     *         This function handles the logic to use the borrowed funds to rollover the loan,
     *         make necessary repayments, and manage the loan commitments.
     *
     * @dev The function ensures the initiator is this contract, decodes the data provided by
     *      the flash loan call, repays the original loan in full, accepts new loan commitments,
     *      approves the repayment for the flash loan and then handles any remaining funds.
     *      This function should only be called by the FlashLoanPool as ensured by the `onlyFlashLoanPool` modifier.
     *
     * @param _flashToken The token in which the flash loan is borrowed.
     * @param _flashAmount The amount of tokens borrowed via the flash loan.
     * @param _flashFees The fees associated with the flash loan to be repaid to Aave.
     * @param _initiator The address initiating the flash loan (must be this contract).
     * @param _data Encoded data containing necessary information for loan rollover.
     *
     * @return Returns true if the operation was successful.
     */
    function executeOperation(
        address _flashToken,
        uint256 _flashAmount,
        uint256 _flashFees,
        address _initiator,
        bytes calldata _data
    ) external virtual onlyFlashLoanPool returns (bool) {
        require(
            _initiator == address(this),
            "This contract must be the initiator"
        );

        RolloverCallbackArgs memory _rolloverArgs = abi.decode(
            _data,
            (RolloverCallbackArgs)
        );

        uint256 repaymentAmount = _repayLoanFull(
            _rolloverArgs.loanId,
            _flashToken,
            _flashAmount
        );

        AcceptCommitmentArgs memory acceptCommitmentArgs = abi.decode(
            _rolloverArgs.acceptCommitmentArgs,
            (AcceptCommitmentArgs)
        );

        // Accept commitment and receive funds to this contract

        (uint256 newLoanId, uint256 acceptCommitmentAmount) = _acceptCommitment(
            _rolloverArgs.lenderCommitmentForwarder,
            _rolloverArgs.borrower,
            _flashToken,
            acceptCommitmentArgs
        );

        //approve the repayment for the flash loan
        IERC20Upgradeable(_flashToken).approve(
            address(POOL()),
            _flashAmount + _flashFees
        );

        uint256 fundsRemaining = acceptCommitmentAmount +
            _rolloverArgs.borrowerAmount -
            repaymentAmount -
            _flashFees;

        if (fundsRemaining > 0) {
            IERC20Upgradeable(_flashToken).transfer(
                _rolloverArgs.borrower,
                fundsRemaining
            );
        }

        emit RolloverLoanComplete(
            _rolloverArgs.borrower,
            _rolloverArgs.loanId,
            newLoanId,
            fundsRemaining
        );

        return true;
    }

    /**
     *
     *
     * @notice Internal function that repays a loan in full on behalf of this contract.
     *
     * @dev The function first calculates the funds held by the contract before repayment, then approves
     *      the repayment amount to the TellerV2 contract and finally repays the loan in full.
     *
     * @param _bidId Identifier of the loan to be repaid.
     * @param _principalToken The token in which the loan was originated.
     * @param _repayAmount The amount to be repaid.
     *
     * @return repayAmount_ The actual amount that was used for repayment.
     */
    function _repayLoanFull(
        uint256 _bidId,
        address _principalToken,
        uint256 _repayAmount
    ) internal returns (uint256 repayAmount_) {
        uint256 fundsBeforeRepayment = IERC20Upgradeable(_principalToken)
            .balanceOf(address(this));

        IERC20Upgradeable(_principalToken).approve(
            address(TELLER_V2),
            _repayAmount
        );
        TELLER_V2.repayLoanFull(_bidId);

        uint256 fundsAfterRepayment = IERC20Upgradeable(_principalToken)
            .balanceOf(address(this));

        repayAmount_ = fundsBeforeRepayment - fundsAfterRepayment;
    }

    /**
     *
     *
     * @notice Accepts a loan commitment using either a Merkle proof or standard method.
     *
     * @dev The function first checks if a Merkle proof is provided, based on which it calls the relevant
     *      `acceptCommitment` function in the LenderCommitmentForwarder contract.
     *
     * @param borrower The address of the borrower for whom the commitment is being accepted.
     * @param principalToken The token in which the loan is being accepted.
     * @param _commitmentArgs The arguments necessary for accepting the commitment.
     *
     * @return bidId_ Identifier of the accepted loan.
     * @return acceptCommitmentAmount_ The amount received from accepting the commitment.
     */
    function _acceptCommitment(
        address lenderCommitmentForwarder,
        address borrower,
        address principalToken,
        AcceptCommitmentArgs memory _commitmentArgs
    )
        internal
        virtual
        returns (uint256 bidId_, uint256 acceptCommitmentAmount_)
    {
        uint256 fundsBeforeAcceptCommitment = IERC20Upgradeable(principalToken)
            .balanceOf(address(this));

        bool usingMerkleProof = _commitmentArgs.merkleProof.length > 0;

        if (usingMerkleProof) {
            bytes memory responseData = address(lenderCommitmentForwarder)
                .functionCall(
                    abi.encodePacked(
                        abi.encodeWithSelector(
                            ILenderCommitmentForwarder
                                .acceptCommitmentWithRecipientAndProof
                                .selector,
                            _commitmentArgs.commitmentId,
                            _commitmentArgs.principalAmount,
                            _commitmentArgs.collateralAmount,
                            _commitmentArgs.collateralTokenId,
                            _commitmentArgs.collateralTokenAddress,
                            address(this),
                            _commitmentArgs.interestRate,
                            _commitmentArgs.loanDuration,
                            _commitmentArgs.merkleProof
                        ),
                        borrower //cant be msg.sender because of the flash flow
                    )
                );

            (bidId_) = abi.decode(responseData, (uint256));
        } else {
            bytes memory responseData = address(lenderCommitmentForwarder)
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
                        borrower //cant be msg.sender because of the flash flow
                    )
                );

            (bidId_) = abi.decode(responseData, (uint256));
        }

        uint256 fundsAfterAcceptCommitment = IERC20Upgradeable(principalToken)
            .balanceOf(address(this));
        acceptCommitmentAmount_ =
            fundsAfterAcceptCommitment -
            fundsBeforeAcceptCommitment;
    }

    function ADDRESSES_PROVIDER() public view returns (IPoolAddressesProvider) {
        return IPoolAddressesProvider(POOL_ADDRESSES_PROVIDER);
    }

    function POOL() public view returns (IPool) {
        return IPool(ADDRESSES_PROVIDER().getPool());
    }

    /**
     * @notice Calculates the amount for loan rollover, determining if the borrower owes or receives funds.
     * @param _loanId The ID of the loan to calculate the rollover amount for.
     * @param _commitmentArgs Arguments for the commitment.
     * @param _timestamp The timestamp for when the calculation is executed.
    
     */
    function calculateRolloverAmount(
        address _lenderCommitmentForwarder,
        uint256 _loanId,
        AcceptCommitmentArgs calldata _commitmentArgs,
        uint16 _flashloanPremiumPct,
        uint256 _timestamp
    ) external view returns (uint256 _flashAmount, int256 _borrowerAmount) {
        Payment memory repayAmountOwed = TELLER_V2.calculateAmountOwed(
            _loanId,
            _timestamp
        );

        uint256 _marketId = _getMarketIdForCommitment(_lenderCommitmentForwarder,
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

        uint256 commitmentPrincipalReceived = commitmentPrincipalRequested -
            amountToMarketplace -
            amountToProtocol;

        // by default, we will flash exactly what we need to do relayLoanFull
        uint256 repayFullAmount = repayAmountOwed.principal +
            repayAmountOwed.interest;

        _flashAmount = repayFullAmount;
        uint256 _flashLoanFee = _flashAmount.percent(_flashloanPremiumPct);

        _borrowerAmount =
            int256(commitmentPrincipalReceived) -
            int256(repayFullAmount) -
            int256(_flashLoanFee);
    }

    /**
     * @notice Retrieves the market ID associated with a given commitment.
     * @param _commitmentId The ID of the commitment for which to fetch the market ID.
     * @return The ID of the market associated with the provided commitment.
     */
    function _getMarketIdForCommitment(address _lenderCommitmentForwarder, uint256 _commitmentId)
        internal
        view
        returns (uint256)
    {
        return ILenderCommitmentForwarder(_lenderCommitmentForwarder).getCommitmentMarketId(_commitmentId);
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
 
