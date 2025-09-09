// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Contracts
import "@openzeppelin/contracts-upgradeable/utils/AddressUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/token/ERC20/IERC20Upgradeable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "../../libraries/NumbersLib.sol";

// Interfaces
import "./FlashRolloverLoan_G1.sol";

contract FlashRolloverLoan_G2 is FlashRolloverLoan_G1 {
    using AddressUpgradeable for address;
    using NumbersLib for uint256;

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor(
        address _tellerV2,
        address _lenderCommitmentForwarder,
        address _poolAddressesProvider
    )
        FlashRolloverLoan_G1(
            _tellerV2,
            _lenderCommitmentForwarder,
            _poolAddressesProvider
        )
    {}

    /*

        This assumes that the flash amount will be the repayLoanFull amount !!

    */
    /**
     * @notice Calculates the amount for loan rollover, determining if the borrower owes or receives funds.
     * @param _loanId The ID of the loan to calculate the rollover amount for.
     * @param _commitmentArgs Arguments for the commitment.
     * @param _timestamp The timestamp for when the calculation is executed.
    
     */
    function calculateRolloverAmount(
        uint256 _loanId,
        AcceptCommitmentArgs calldata _commitmentArgs,
        uint16 _flashloanPremiumPct,
        uint256 _timestamp
    ) external view returns (uint256 _flashAmount, int256 _borrowerAmount) {
        Payment memory repayAmountOwed = TELLER_V2.calculateAmountOwed(
            _loanId,
            _timestamp
        );

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
