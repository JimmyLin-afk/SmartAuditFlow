pragma solidity >=0.8.0 <0.9.0;
// SPDX-License-Identifier: MIT

// Contracts
import "./ProtocolFee.sol";
import "./TellerV2Storage.sol";
import "./TellerV2Context.sol";
import "./pausing/HasProtocolPausingManager.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol"; 
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/StringsUpgradeable.sol";

// Interfaces
import "./interfaces/IMarketRegistry.sol";
import "./interfaces/IReputationManager.sol";
import "./interfaces/ITellerV2.sol";
import { Collateral } from "./interfaces/escrow/ICollateralEscrowV1.sol";
import "./interfaces/IEscrowVault.sol";

import { ILoanRepaymentCallbacks } from "./interfaces/ILoanRepaymentCallbacks.sol";
import "./interfaces/ILoanRepaymentListener.sol";

// Libraries
import "@openzeppelin/contracts/utils/Address.sol";
import {SafeERC20} from "./openzeppelin/SafeERC20.sol";

import "./libraries/NumbersLib.sol";
import "./libraries/ExcessivelySafeCall.sol";

import { V2Calculations, PaymentCycleType } from "./libraries/V2Calculations.sol";

/* Errors */
/**
 * @notice This error is reverted when the action isn't allowed
 * @param bidId The id of the bid.
 * @param action The action string (i.e: 'repayLoan', 'cancelBid', 'etc)
 * @param message The message string to return to the user explaining why the tx was reverted
 */
error ActionNotAllowed(uint256 bidId, string action, string message);

/**
 * @notice This error is reverted when repayment amount is less than the required minimum
 * @param bidId The id of the bid the borrower is attempting to repay.
 * @param payment The payment made by the borrower
 * @param minimumOwed The minimum owed value
 */
error PaymentNotMinimum(uint256 bidId, uint256 payment, uint256 minimumOwed);

contract TellerV2 is
    ITellerV2,
    ILoanRepaymentCallbacks,
    OwnableUpgradeable,
    ProtocolFee,
    HasProtocolPausingManager,
    TellerV2Storage,
    TellerV2Context
{
    using Address for address;
    using SafeERC20 for IERC20;
    using NumbersLib for uint256;
    using EnumerableSet for EnumerableSet.AddressSet;
    using EnumerableSet for EnumerableSet.UintSet;

    //the first 20 bytes of keccak256("lender manager")
    address constant USING_LENDER_MANAGER =
        0x84D409EeD89F6558fE3646397146232665788bF8;

    /** Events */

    /**
     * @notice This event is emitted when a new bid is submitted.
     * @param bidId The id of the bid submitted.
     * @param borrower The address of the bid borrower.
     * @param metadataURI URI for additional bid information as part of loan bid.
     */
    event SubmittedBid(
        uint256 indexed bidId,
        address indexed borrower,
        address receiver,
        bytes32 indexed metadataURI
    );

    /**
     * @notice This event is emitted when a bid has been accepted by a lender.
     * @param bidId The id of the bid accepted.
     * @param lender The address of the accepted bid lender.
     */
    event AcceptedBid(uint256 indexed bidId, address indexed lender);

    /**
     * @notice This event is emitted when a previously submitted bid has been cancelled.
     * @param bidId The id of the cancelled bid.
     */
    event CancelledBid(uint256 indexed bidId);

    /**
     * @notice This event is emitted when market owner has cancelled a pending bid in their market.
     * @param bidId The id of the bid funded.
     *
     * Note: The `CancelledBid` event will also be emitted.
     */
    event MarketOwnerCancelledBid(uint256 indexed bidId);

    /**
     * @notice This event is emitted when a payment is made towards an active loan.
     * @param bidId The id of the bid/loan to which the payment was made.
     */
    event LoanRepayment(uint256 indexed bidId);

    /**
     * @notice This event is emitted when a loan has been fully repaid.
     * @param bidId The id of the bid/loan which was repaid.
     */
    event LoanRepaid(uint256 indexed bidId);

    /**
     * @notice This event is emitted when a loan has been closed by a lender to claim collateral.
     * @param bidId The id of the bid accepted.
     */
    event LoanClosed(uint256 indexed bidId);

    /**
     * @notice This event is emitted when a loan has been fully repaid.
     * @param bidId The id of the bid/loan which was repaid.
     */
    event LoanLiquidated(uint256 indexed bidId, address indexed liquidator);

    /**
     * @notice This event is emitted when a fee has been paid related to a bid.
     * @param bidId The id of the bid.
     * @param feeType The name of the fee being paid.
     * @param amount The amount of the fee being paid.
     */
    event FeePaid(
        uint256 indexed bidId,
        string indexed feeType,
        uint256 indexed amount
    );

    /** Modifiers */

    /**
     * @notice This modifier is used to check if the state of a bid is pending, before running an action.
     * @param _bidId The id of the bid to check the state for.
     * @param _action The desired action to run on the bid.
     */
    modifier pendingBid(uint256 _bidId, string memory _action) {
        if (bids[_bidId].state != BidState.PENDING) {
            revert ActionNotAllowed(_bidId, _action, "Bid not pending");
        }

        _;
    }

    /**
     * @notice This modifier is used to check if the state of a loan has been accepted, before running an action.
     * @param _bidId The id of the bid to check the state for.
     * @param _action The desired action to run on the bid.
     */
    modifier acceptedLoan(uint256 _bidId, string memory _action) {
        if (bids[_bidId].state != BidState.ACCEPTED) {
            revert ActionNotAllowed(_bidId, _action, "Loan not accepted");
        }

        _;
    }


    /** Constant Variables **/

    uint8 public constant CURRENT_CODE_VERSION = 10;

    uint32 public constant LIQUIDATION_DELAY = 86400; //ONE DAY IN SECONDS

    /** Constructor **/

    constructor(address trustedForwarder) TellerV2Context(trustedForwarder) {}

    /** External Functions **/

    /**
     * @notice Initializes the proxy.
     * @param _protocolFee The fee collected by the protocol for loan processing.
     * @param _marketRegistry The address of the market registry contract for the protocol.
     * @param _reputationManager The address of the reputation manager contract.
     * @param _lenderCommitmentForwarder The address of the lender commitment forwarder contract.
     * @param _collateralManager The address of the collateral manager contracts.
     * @param _lenderManager The address of the lender manager contract for loans on the protocol.
     * @param _protocolPausingManager The address of the pausing manager contract for the protocol.
     */
    function initialize(
        uint16 _protocolFee,
        address _marketRegistry,
        address _reputationManager,
        address _lenderCommitmentForwarder,
        address _collateralManager,
        address _lenderManager,
        address _escrowVault,
        address _protocolPausingManager
    ) external initializer {
        __ProtocolFee_init(_protocolFee);

        //__Pausable_init();

        require(
            _lenderCommitmentForwarder.isContract(),
            "LCF_ic"
        );
        lenderCommitmentForwarder = _lenderCommitmentForwarder;

        require(
            _marketRegistry.isContract(),
            "MR_ic"
        );
        marketRegistry = IMarketRegistry(_marketRegistry);

        require(
            _reputationManager.isContract(),
            "RM_ic"
        );
        reputationManager = IReputationManager(_reputationManager);

        require(
            _collateralManager.isContract(),
            "CM_ic"
        );
        collateralManager = ICollateralManager(_collateralManager);

       
       
        require(
            _lenderManager.isContract(),
            "LM_ic"
        );
        lenderManager = ILenderManager(_lenderManager);


         

         require(_escrowVault.isContract(), "EV_ic");
        escrowVault = IEscrowVault(_escrowVault);




        _setProtocolPausingManager(_protocolPausingManager);
    }


    /* function setEscrowVault(address _escrowVault) external reinitializer(9) {
        _setEscrowVault(_escrowVault);
    }
    */

    
     function setProtocolPausingManager(         
        address _protocolPausingManager
    ) external reinitializer(10) {

         _setProtocolPausingManager(_protocolPausingManager);

    }
 
    
    /**
     * @notice Function for a borrower to create a bid for a loan without Collateral.
     * @param _lendingToken The lending token asset requested to be borrowed.
     * @param _marketplaceId The unique id of the marketplace for the bid.
     * @param _principal The principal amount of the loan bid.
     * @param _duration The recurrent length of time before which a payment is due.
     * @param _APR The proposed interest rate for the loan bid.
     * @param _metadataURI The URI for additional borrower loan information as part of loan bid.
     * @param _receiver The address where the loan amount will be sent to.
     */
    function submitBid(
        address _lendingToken,
        uint256 _marketplaceId,
        uint256 _principal,
        uint32 _duration,
        uint16 _APR,
        string calldata _metadataURI,
        address _receiver
    ) public override whenProtocolNotPaused returns (uint256 bidId_) {
        bidId_ = _submitBid(
            _lendingToken,
            _marketplaceId,
            _principal,
            _duration,
            _APR,
            _metadataURI,
            _receiver
        );
    }

    /**
     * @notice Function for a borrower to create a bid for a loan with Collateral.
     * @param _lendingToken The lending token asset requested to be borrowed.
     * @param _marketplaceId The unique id of the marketplace for the bid.
     * @param _principal The principal amount of the loan bid.
     * @param _duration The recurrent length of time before which a payment is due.
     * @param _APR The proposed interest rate for the loan bid.
     * @param _metadataURI The URI for additional borrower loan information as part of loan bid.
     * @param _receiver The address where the loan amount will be sent to.
     * @param _collateralInfo Additional information about the collateral asset.
     */
    function submitBid(
        address _lendingToken,
        uint256 _marketplaceId,
        uint256 _principal,
        uint32 _duration,
        uint16 _APR,
        string calldata _metadataURI,
        address _receiver,
        Collateral[] calldata _collateralInfo
    ) public override whenProtocolNotPaused returns (uint256 bidId_) {
        bidId_ = _submitBid(
            _lendingToken,
            _marketplaceId,
            _principal,
            _duration,
            _APR,
            _metadataURI,
            _receiver
        );

        bool validation = collateralManager.commitCollateral(
            bidId_,
            _collateralInfo
        );

        require(
            validation == true,
            "C bal NV"
        );
    }

    function _submitBid(
        address _lendingToken,
        uint256 _marketplaceId,
        uint256 _principal,
        uint32 _duration,
        uint16 _APR,
        string calldata _metadataURI,
        address _receiver
    ) internal virtual returns (uint256 bidId_) {
        address sender = _msgSenderForMarket(_marketplaceId);

        (bool isVerified, ) = marketRegistry.isVerifiedBorrower(
            _marketplaceId,
            sender
        );

        require(isVerified, "Borrower NV");

        require(
            marketRegistry.isMarketOpen(_marketplaceId),
            "Mkt C"
        );

        // Set response bid ID.
        bidId_ = bidId;

        // Create and store our bid into the mapping
        Bid storage bid = bids[bidId];
        bid.borrower = sender;
        bid.receiver = _receiver != address(0) ? _receiver : bid.borrower;
        bid.marketplaceId = _marketplaceId;
        bid.loanDetails.lendingToken = IERC20(_lendingToken);
        bid.loanDetails.principal = _principal;
        bid.loanDetails.loanDuration = _duration;
        bid.loanDetails.timestamp = uint32(block.timestamp);

        // Set payment cycle type based on market setting (custom or monthly)
        (bid.terms.paymentCycle, bidPaymentCycleType[bidId]) = marketRegistry
            .getPaymentCycle(_marketplaceId);

        bid.terms.APR = _APR;

        bidDefaultDuration[bidId] = marketRegistry.getPaymentDefaultDuration(
            _marketplaceId
        );

        bidExpirationTime[bidId] = marketRegistry.getBidExpirationTime(
            _marketplaceId
        );

        bid.paymentType = marketRegistry.getPaymentType(_marketplaceId);

        bid.terms.paymentCycleAmount = V2Calculations
            .calculatePaymentCycleAmount(
                bid.paymentType,
                bidPaymentCycleType[bidId],
                _principal,
                _duration,
                bid.terms.paymentCycle,
                _APR
            );

        uris[bidId] = _metadataURI;
        bid.state = BidState.PENDING;

        emit SubmittedBid(
            bidId,
            bid.borrower,
            bid.receiver,
            keccak256(abi.encodePacked(_metadataURI))
        );

        // Store bid inside borrower bids mapping
        borrowerBids[bid.borrower].push(bidId);

        // Increment bid id counter
        bidId++;
    }

    /**
     * @notice Function for a borrower to cancel their pending bid.
     * @param _bidId The id of the bid to cancel.
     */
    function cancelBid(uint256 _bidId) external {
        if (
            _msgSenderForMarket(bids[_bidId].marketplaceId) !=
            bids[_bidId].borrower
        ) {
            revert ActionNotAllowed({
                bidId: _bidId,
                action: "CB",
                message: "Not bid owner"  //this is a TON of storage space
            });
        }
        _cancelBid(_bidId);
    }

    /**
     * @notice Function for a market owner to cancel a bid in the market.
     * @param _bidId The id of the bid to cancel.
     */
    function marketOwnerCancelBid(uint256 _bidId) external {
        if (
            _msgSender() !=
            marketRegistry.getMarketOwner(bids[_bidId].marketplaceId)
        ) {
            revert ActionNotAllowed({
                bidId: _bidId,
                action: "MOCB",
                message: "Not market owner" //this is a TON of storage space 
            });
        }
        _cancelBid(_bidId);
        emit MarketOwnerCancelledBid(_bidId);
    }

    /**
     * @notice Function for users to cancel a bid.
     * @param _bidId The id of the bid to be cancelled.
     */
    function _cancelBid(uint256 _bidId)
        internal
        virtual
        pendingBid(_bidId, "cb")
    {
        // Set the bid state to CANCELLED
        bids[_bidId].state = BidState.CANCELLED;

        // Emit CancelledBid event
        emit CancelledBid(_bidId);
    }

    /**
     * @notice Function for a lender to accept a proposed loan bid.
     * @param _bidId The id of the loan bid to accept.
     */
    function lenderAcceptBid(uint256 _bidId)
        external
        override
        pendingBid(_bidId, "lab")
        whenProtocolNotPaused
        returns (
            uint256 amountToProtocol,
            uint256 amountToMarketplace,
            uint256 amountToBorrower
        )
    {
        // Retrieve bid
        Bid storage bid = bids[_bidId];

        address sender = _msgSenderForMarket(bid.marketplaceId);

        (bool isVerified, ) = marketRegistry.isVerifiedLender(
            bid.marketplaceId,
            sender
        );
        require(isVerified, "NV");

        require(
            !marketRegistry.isMarketClosed(bid.marketplaceId),
            "Market is closed"
        );

        require(!isLoanExpired(_bidId), "BE");

        // Set timestamp
        bid.loanDetails.acceptedTimestamp = uint32(block.timestamp);
        bid.loanDetails.lastRepaidTimestamp = uint32(block.timestamp);

        // Mark borrower's request as accepted
        bid.state = BidState.ACCEPTED;

        // Declare the bid acceptor as the lender of the bid
        bid.lender = sender;

        // Tell the collateral manager to deploy the escrow and pull funds from the borrower if applicable
        collateralManager.deployAndDeposit(_bidId);

        // Transfer funds to borrower from the lender
        amountToProtocol = bid.loanDetails.principal.percent(protocolFee());
        amountToMarketplace = bid.loanDetails.principal.percent(
            marketRegistry.getMarketplaceFee(bid.marketplaceId)
        );
        amountToBorrower =
            bid.loanDetails.principal -
            amountToProtocol -
            amountToMarketplace;

        //transfer fee to protocol
        if (amountToProtocol > 0) {
            bid.loanDetails.lendingToken.safeTransferFrom(
                sender,
                _getProtocolFeeRecipient(),
                amountToProtocol
            );
        }

        //transfer fee to marketplace
        if (amountToMarketplace > 0) {
            bid.loanDetails.lendingToken.safeTransferFrom(
                sender,
                marketRegistry.getMarketFeeRecipient(bid.marketplaceId),
                amountToMarketplace
            );
        }

//local stack scope
{
        uint256 balanceBefore = bid.loanDetails.lendingToken.balanceOf(
            address(bid.receiver)
        );  

        //transfer funds to borrower
        if (amountToBorrower > 0) {
            bid.loanDetails.lendingToken.safeTransferFrom(
                sender,
                bid.receiver,
                amountToBorrower
            );
        }

         uint256 balanceAfter = bid.loanDetails.lendingToken.balanceOf(
                address(bid.receiver)
         );

        //used to revert for fee-on-transfer tokens as principal            
         uint256 paymentAmountReceived = balanceAfter - balanceBefore;
         require(amountToBorrower == paymentAmountReceived, "UT"); 
}


        // Record volume filled by lenders
        lenderVolumeFilled[address(bid.loanDetails.lendingToken)][sender] += bid
            .loanDetails
            .principal;
        totalVolumeFilled[address(bid.loanDetails.lendingToken)] += bid
            .loanDetails
            .principal;

        // Add borrower's active bid
        _borrowerBidsActive[bid.borrower].add(_bidId);

        // Emit AcceptedBid
        emit AcceptedBid(_bidId, sender);

        emit FeePaid(_bidId, "protocol", amountToProtocol);
        emit FeePaid(_bidId, "marketplace", amountToMarketplace);
    }

    function claimLoanNFT(uint256 _bidId)
        external
        acceptedLoan(_bidId, "cln")
        whenProtocolNotPaused
    {
        // Retrieve bid
        Bid storage bid = bids[_bidId];

        address sender = _msgSenderForMarket(bid.marketplaceId);
        require(sender == bid.lender, "NV Lender");

        // set lender address to the lender manager so we know to check the owner of the NFT for the true lender
        bid.lender = address(USING_LENDER_MANAGER);

        // mint an NFT with the lender manager
        lenderManager.registerLoan(_bidId, sender);
    }

    /**
     * @notice Function for users to make the minimum amount due for an active loan.
     * @param _bidId The id of the loan to make the payment towards.
     */
    function repayLoanMinimum(uint256 _bidId)
        external
        acceptedLoan(_bidId, "rl")
    {
        (
            uint256 owedPrincipal,
            uint256 duePrincipal,
            uint256 interest
        ) = V2Calculations.calculateAmountOwed(
                bids[_bidId],
                block.timestamp,
                _getBidPaymentCycleType(_bidId),
                _getBidPaymentCycleDuration(_bidId)
            );
        _repayLoan(
            _bidId,
            Payment({ principal: duePrincipal, interest: interest }),
            owedPrincipal + interest,
            true
        );
    }

    /**
     * @notice Function for users to repay an active loan in full.
     * @param _bidId The id of the loan to make the payment towards.
     */
    function repayLoanFull(uint256 _bidId)
        external
        acceptedLoan(_bidId, "rl")
    {
        _repayLoanFull(_bidId, true);
    }

    // function that the borrower (ideally) sends to repay the loan
    /**
     * @notice Function for users to make a payment towards an active loan.
     * @param _bidId The id of the loan to make the payment towards.
     * @param _amount The amount of the payment.
     */
    function repayLoan(uint256 _bidId, uint256 _amount)
        external
        acceptedLoan(_bidId, "rl")
    {
        _repayLoanAtleastMinimum(_bidId, _amount, true);
    }

    /**
     * @notice Function for users to repay an active loan in full.
     * @param _bidId The id of the loan to make the payment towards.
     */
    function repayLoanFullWithoutCollateralWithdraw(uint256 _bidId)
        external
        acceptedLoan(_bidId, "rl")
    {
        _repayLoanFull(_bidId, false);
    }

    function repayLoanWithoutCollateralWithdraw(uint256 _bidId, uint256 _amount)
        external
        acceptedLoan(_bidId, "rl")
    {
        _repayLoanAtleastMinimum(_bidId, _amount, false);
    }

    function _repayLoanFull(uint256 _bidId, bool withdrawCollateral) internal {
        (uint256 owedPrincipal, , uint256 interest) = V2Calculations
            .calculateAmountOwed(
                bids[_bidId],
                block.timestamp,
                _getBidPaymentCycleType(_bidId),
                _getBidPaymentCycleDuration(_bidId)
            );
        _repayLoan(
            _bidId,
            Payment({ principal: owedPrincipal, interest: interest }),
            owedPrincipal + interest,
            withdrawCollateral
        );
    }

    function _repayLoanAtleastMinimum(
        uint256 _bidId,
        uint256 _amount,
        bool withdrawCollateral
    ) internal {
        (
            uint256 owedPrincipal,
            uint256 duePrincipal,
            uint256 interest
        ) = V2Calculations.calculateAmountOwed(
                bids[_bidId],
                block.timestamp,
                _getBidPaymentCycleType(_bidId),
                _getBidPaymentCycleDuration(_bidId)
            );
        uint256 minimumOwed = duePrincipal + interest;

        // If amount is less than minimumOwed, we revert
        if (_amount < minimumOwed) {
            revert PaymentNotMinimum(_bidId, _amount, minimumOwed);
        }

        _repayLoan(
            _bidId,
            Payment({ principal: _amount - interest, interest: interest }),
            owedPrincipal + interest,
            withdrawCollateral
        );
    }

 


    function lenderCloseLoan(uint256 _bidId)
        external whenProtocolNotPaused whenLiquidationsNotPaused
        acceptedLoan(_bidId, "lcc")
    {
        Bid storage bid = bids[_bidId];
        address _collateralRecipient = getLoanLender(_bidId);

        _lenderCloseLoanWithRecipient(_bidId, _collateralRecipient);
    }

    /**
     * @notice Function for lender to claim collateral for a defaulted loan. The only purpose of a CLOSED loan is to make collateral claimable by lender.
     * @param _bidId The id of the loan to set to CLOSED status.
     */
    function lenderCloseLoanWithRecipient(
        uint256 _bidId,
        address _collateralRecipient
    ) external whenProtocolNotPaused whenLiquidationsNotPaused {
        _lenderCloseLoanWithRecipient(_bidId, _collateralRecipient);
    }

    function _lenderCloseLoanWithRecipient(
        uint256 _bidId,
        address _collateralRecipient
    ) internal acceptedLoan(_bidId, "lcc") {
        require(isLoanDefaulted(_bidId), "ND");

        Bid storage bid = bids[_bidId];
        bid.state = BidState.CLOSED;

        address sender = _msgSenderForMarket(bid.marketplaceId);
        require(sender == getLoanLender(_bidId), "NLL");

      
        collateralManager.lenderClaimCollateralWithRecipient(_bidId, _collateralRecipient);

        emit LoanClosed(_bidId);
    }

    /**
     * @notice Function for users to liquidate a defaulted loan.
     * @param _bidId The id of the loan to make the payment towards.
     */
    function liquidateLoanFull(uint256 _bidId)
        external whenProtocolNotPaused whenLiquidationsNotPaused
        acceptedLoan(_bidId, "ll")
    {
        Bid storage bid = bids[_bidId];

        // If loan is backed by collateral, withdraw and send to the liquidator
        address recipient = _msgSenderForMarket(bid.marketplaceId);

        _liquidateLoanFull(_bidId, recipient);
    }

    function liquidateLoanFullWithRecipient(uint256 _bidId, address _recipient)
        external whenProtocolNotPaused whenLiquidationsNotPaused
        acceptedLoan(_bidId, "ll")
    {
        _liquidateLoanFull(_bidId, _recipient);
    }

    /**
     * @notice Function for users to liquidate a defaulted loan.
     * @param _bidId The id of the loan to make the payment towards.
     */
    function _liquidateLoanFull(uint256 _bidId, address _recipient)
        internal
        acceptedLoan(_bidId, "ll")
    {
        require(isLoanLiquidateable(_bidId), "NL");

        Bid storage bid = bids[_bidId];

        // change state here to prevent re-entrancy
        bid.state = BidState.LIQUIDATED;

        (uint256 owedPrincipal, , uint256 interest) = V2Calculations
            .calculateAmountOwed(
                bid,
                block.timestamp,
                _getBidPaymentCycleType(_bidId),
                _getBidPaymentCycleDuration(_bidId)
            );

        //this sets the state to 'repaid'
        _repayLoan(
            _bidId,
            Payment({ principal: owedPrincipal, interest: interest }),
            owedPrincipal + interest,
            false
        );

        collateralManager.liquidateCollateral(_bidId, _recipient);

        address liquidator = _msgSenderForMarket(bid.marketplaceId);

        emit LoanLiquidated(_bidId, liquidator);
    }

    /**
     * @notice Internal function to make a loan payment.
     * @dev Updates the bid's `status` to `PAID` only if it is not already marked as `LIQUIDATED`
     * @param _bidId The id of the loan to make the payment towards.
     * @param _payment The Payment struct with payments amounts towards principal and interest respectively.
     * @param _owedAmount The total amount owed on the loan.
     */
    function _repayLoan(
        uint256 _bidId,
        Payment memory _payment,
        uint256 _owedAmount,
        bool _shouldWithdrawCollateral
    ) internal virtual {
        Bid storage bid = bids[_bidId];
        uint256 paymentAmount = _payment.principal + _payment.interest;

        RepMark mark = reputationManager.updateAccountReputation(
            bid.borrower,
            _bidId
        );

        // Check if we are sending a payment or amount remaining
        if (paymentAmount >= _owedAmount) {
            paymentAmount = _owedAmount;

            if (bid.state != BidState.LIQUIDATED) {
                bid.state = BidState.PAID;
            }

            // Remove borrower's active bid
            _borrowerBidsActive[bid.borrower].remove(_bidId);

            // If loan is is being liquidated and backed by collateral, withdraw and send to borrower
            if (_shouldWithdrawCollateral) { 
               
                //   _getCollateralManagerForBid(_bidId).withdraw(_bidId);
                collateralManager.withdraw(_bidId);
            }

            emit LoanRepaid(_bidId);
        } else {
            emit LoanRepayment(_bidId);
        }

        _sendOrEscrowFunds(_bidId, _payment); //send or escrow the funds

        // update our mappings
        bid.loanDetails.totalRepaid.principal += _payment.principal;
        bid.loanDetails.totalRepaid.interest += _payment.interest;
        bid.loanDetails.lastRepaidTimestamp = uint32(block.timestamp);

        // If the loan is paid in full and has a mark, we should update the current reputation
        if (mark != RepMark.Good) {
            reputationManager.updateAccountReputation(bid.borrower, _bidId);
        }
    }


    /*
    If for some reason the lender cannot receive funds, should put those funds into the escrow 
    so the loan can always be repaid and the borrower can get collateral out 
 

    */
    function _sendOrEscrowFunds(uint256 _bidId, Payment memory _payment)
        internal virtual 
    {
        Bid storage bid = bids[_bidId];
        address lender = getLoanLender(_bidId);

        uint256 _paymentAmount = _payment.principal + _payment.interest;

            //USER STORY:  Should function properly with USDT and USDC and WETH for sure 

            //USER STORY  :  if the lender cannot receive funds for some reason (denylisted) 
            //then we will try to send the funds to the EscrowContract bc we want the borrower to be able to get back their collateral ! 
            // i.e.  lender not being able to recieve funds should STILL allow repayment to succeed ! 

          
              bool transferSuccess = safeTransferFromERC20Custom( 
                    address(bid.loanDetails.lendingToken),
                   _msgSenderForMarket(bid.marketplaceId) , //from
                   lender, //to
                    _paymentAmount // amount                 
             );
                  
            if  (!transferSuccess) {  
                //could not send funds due to an issue with lender (denylisted?) so we are going to try and send the funds to the
                // escrow wallet FOR the lender to be able to retrieve at a later time when they are no longer denylisted by the token  
            
                address sender = _msgSenderForMarket(bid.marketplaceId);

                // fee on transfer tokens are not supported in the lenderAcceptBid step

                //if unable, pay to escrow
                bid.loanDetails.lendingToken.safeTransferFrom(
                    sender,
                    address(this),
                    _paymentAmount
                ); 

                bid.loanDetails.lendingToken.forceApprove(
                    address(escrowVault),
                    _paymentAmount
                );

                IEscrowVault(escrowVault).deposit(
                    lender,
                    address(bid.loanDetails.lendingToken),
                    _paymentAmount
                );


            }

        address loanRepaymentListener = repaymentListenerForBid[_bidId];

        if (loanRepaymentListener != address(0)) {
            require(gasleft() >= 80000, "NR gas");  //fixes the 63/64 remaining issue
            try
                ILoanRepaymentListener(loanRepaymentListener).repayLoanCallback{
                    gas: 80000
                }( //limit gas costs to prevent lender preventing repayments
                    _bidId,
                    _msgSenderForMarket(bid.marketplaceId),
                    _payment.principal,
                    _payment.interest
                )
            {} catch {}
        }
    }

    /*
      A try/catch pattern for safeTransferERC20 that helps support standard ERC20 tokens and non-standard ones like USDT 

      @notice  If the token address is an EOA, callSuccess will always be true so token address should always be a contract. 
    */
    function safeTransferFromERC20Custom(

        address _token,
        address _from,
        address _to,
        uint256 _amount 

    ) internal virtual returns (bool success) {

        //https://github.com/nomad-xyz/ExcessivelySafeCall
        //this works similarly to a try catch -- an inner revert doesnt revert us but will make callSuccess be false. 
         ( bool callSuccess, bytes memory callReturnData ) = ExcessivelySafeCall.excessivelySafeCall(
                address(_token),
                100000, //max gas 
                0,  //value (eth) to send in call
                1000, //max return data size  
                abi.encodeWithSelector(
                    IERC20
                        .transferFrom
                        .selector,
                    _from,
                    _to,
                    _amount
                )  
           );
    

             //If the token returns data, make sure it returns true. This helps us with USDT which may revert but never returns a bool.
            bool dataIsSuccess = true;
            if (callReturnData.length >= 32) {
                assembly {
                    // Load the first 32 bytes of the return data (assuming it's a bool)
                    let result := mload(add(callReturnData, 0x20))
                    // Check if the result equals `true` (1)
                    dataIsSuccess := eq(result, 1)
                }
            }

           // ensures that both callSuccess (the low-level call didn't fail) and dataIsSuccess (the function returned true if it returned something).
            return callSuccess && dataIsSuccess; 


    }


    /**
     * @notice Calculates the total amount owed for a loan bid at a specific timestamp.
     * @param _bidId The id of the loan bid to calculate the owed amount for.
     * @param _timestamp The timestamp at which to calculate the loan owed amount at.
     */
    function calculateAmountOwed(uint256 _bidId, uint256 _timestamp)
        public
        view
        returns (Payment memory owed)
    {
        Bid storage bid = bids[_bidId];
        if (
            bid.state != BidState.ACCEPTED ||
            bid.loanDetails.acceptedTimestamp >= _timestamp
        ) return owed;

        (uint256 owedPrincipal, , uint256 interest) = V2Calculations
            .calculateAmountOwed(
                bid,
                _timestamp,
                _getBidPaymentCycleType(_bidId),
                _getBidPaymentCycleDuration(_bidId)
            );

        owed.principal = owedPrincipal;
        owed.interest = interest;
    }

    /**
     * @notice Calculates the minimum payment amount due for a loan at a specific timestamp.
     * @param _bidId The id of the loan bid to get the payment amount for.
     * @param _timestamp The timestamp at which to get the due payment at.
     */
    function calculateAmountDue(uint256 _bidId, uint256 _timestamp)
        public
        view
        returns (Payment memory due)
    {
        Bid storage bid = bids[_bidId];
        if (
            bids[_bidId].state != BidState.ACCEPTED ||
            bid.loanDetails.acceptedTimestamp >= _timestamp
        ) return due;

        (, uint256 duePrincipal, uint256 interest) = V2Calculations
            .calculateAmountOwed(
                bid,
                _timestamp,
                _getBidPaymentCycleType(_bidId),
                _getBidPaymentCycleDuration(_bidId)
            );
        due.principal = duePrincipal;
        due.interest = interest;
    }

    /**
     * @notice Returns the next due date for a loan payment.
     * @param _bidId The id of the loan bid.
     */
    function calculateNextDueDate(uint256 _bidId)
        public
        view
        returns (uint32 dueDate_)
    {
        Bid storage bid = bids[_bidId];
        if (bids[_bidId].state != BidState.ACCEPTED) return dueDate_;

        return
            V2Calculations.calculateNextDueDate(
                bid.loanDetails.acceptedTimestamp,
                bid.terms.paymentCycle,
                bid.loanDetails.loanDuration,
                lastRepaidTimestamp(_bidId),
                bidPaymentCycleType[_bidId]
            );
    }

    /**
     * @notice Checks to see if a borrower is delinquent.
     * @param _bidId The id of the loan bid to check for.
     */
    function isPaymentLate(uint256 _bidId) public view override returns (bool) {
        if (bids[_bidId].state != BidState.ACCEPTED) return false;
        return uint32(block.timestamp) > calculateNextDueDate(_bidId);
    }

    /**
     * @notice Checks to see if a borrower is delinquent.
     * @param _bidId The id of the loan bid to check for.
     * @return bool True if the loan is defaulted.
     */
    function isLoanDefaulted(uint256 _bidId)
        public
        view
        override
        returns (bool)
    {
        return _isLoanDefaulted(_bidId, 0);
    }

    /**
     * @notice Checks to see if a loan was delinquent for longer than liquidation delay.
     * @param _bidId The id of the loan bid to check for.
     * @return bool True if the loan is liquidateable.
     */
    function isLoanLiquidateable(uint256 _bidId)
        public
        view
        override
        returns (bool)
    {
        return _isLoanDefaulted(_bidId, LIQUIDATION_DELAY);
    }

    /**
     * @notice Checks to see if a borrower is delinquent.
     * @param _bidId The id of the loan bid to check for.
     * @param _additionalDelay Amount of additional seconds after a loan defaulted to allow a liquidation.
     * @return bool True if the loan is liquidateable.
     */
    function _isLoanDefaulted(uint256 _bidId, uint32 _additionalDelay)
        internal
        view
        returns (bool)
    {
        Bid storage bid = bids[_bidId];

        // Make sure loan cannot be liquidated if it is not active
        if (bid.state != BidState.ACCEPTED) return false;

        uint32 defaultDuration = bidDefaultDuration[_bidId];

        uint32 dueDate = calculateNextDueDate(_bidId);

        return
            uint32(block.timestamp) >
            dueDate + defaultDuration + _additionalDelay;
    }

    function getEscrowVault() external view returns(address){
        return address(escrowVault);
    }

    function getBidState(uint256 _bidId)
        external
        view
        override
        returns (BidState)
    {
        return bids[_bidId].state;
    }

    function getBorrowerActiveLoanIds(address _borrower)
        external
        view
        override
        returns (uint256[] memory)
    {
        return _borrowerBidsActive[_borrower].values();
    }

    function getBorrowerLoanIds(address _borrower)
        external
        view
        returns (uint256[] memory)
    {
        return borrowerBids[_borrower];
    }

    /**
     * @notice Checks to see if a pending loan has expired so it is no longer able to be accepted.
     * @param _bidId The id of the loan bid to check for.
     */
    function isLoanExpired(uint256 _bidId) public view returns (bool) {
        Bid storage bid = bids[_bidId];

        if (bid.state != BidState.PENDING) return false;
        if (bidExpirationTime[_bidId] == 0) return false;

        return (uint32(block.timestamp) >
            bid.loanDetails.timestamp + bidExpirationTime[_bidId]);
    }

    /**
     * @notice Returns the last repaid timestamp for a loan.
     * @param _bidId The id of the loan bid to get the timestamp for.
     */
    function lastRepaidTimestamp(uint256 _bidId) public view returns (uint32) {
        return V2Calculations.lastRepaidTimestamp(bids[_bidId]);
    }

    /**
     * @notice Returns the borrower address for a given bid.
     * @param _bidId The id of the bid/loan to get the borrower for.
     * @return borrower_ The address of the borrower associated with the bid.
     */
    function getLoanBorrower(uint256 _bidId)
        public
        view
        returns (address borrower_)
    {
        borrower_ = bids[_bidId].borrower;
    }

    /**
     * @notice Returns the lender address for a given bid. If the stored lender address is the `LenderManager` NFT address, return the `ownerOf` for the bid ID.
     * @param _bidId The id of the bid/loan to get the lender for.
     * @return lender_ The address of the lender associated with the bid.
     */
    function getLoanLender(uint256 _bidId)
        public
        view
        returns (address lender_)
    {
        lender_ = bids[_bidId].lender;

        if (lender_ == address(USING_LENDER_MANAGER)) {
            return lenderManager.ownerOf(_bidId);
        }

        //this is left in for backwards compatibility only
        if (lender_ == address(lenderManager)) {
            return lenderManager.ownerOf(_bidId);
        }
    }

    function getLoanLendingToken(uint256 _bidId)
        external
        view
        returns (address token_)
    {
        token_ = address(bids[_bidId].loanDetails.lendingToken);
    }

    function getLoanMarketId(uint256 _bidId)
        external
        view
        returns (uint256 _marketId)
    {
        _marketId = bids[_bidId].marketplaceId;
    }

    function getLoanSummary(uint256 _bidId)
        external
        view
        returns (
            address borrower,
            address lender,
            uint256 marketId,
            address principalTokenAddress,
            uint256 principalAmount,
            uint32 acceptedTimestamp,
            uint32 lastRepaidTimestamp,
            BidState bidState
        )
    {
        Bid storage bid = bids[_bidId];

        borrower = bid.borrower;
        lender = getLoanLender(_bidId);
        marketId = bid.marketplaceId;
        principalTokenAddress = address(bid.loanDetails.lendingToken);
        principalAmount = bid.loanDetails.principal;
        acceptedTimestamp = bid.loanDetails.acceptedTimestamp;
        lastRepaidTimestamp = V2Calculations.lastRepaidTimestamp(bids[_bidId]);
        bidState = bid.state;
    }

    // Additions for lender groups

    function getLoanDefaultTimestamp(uint256 _bidId)
        public
        view
        returns (uint256)
    {
        Bid storage bid = bids[_bidId];

        uint32 defaultDuration = _getBidDefaultDuration(_bidId);

        uint32 dueDate = calculateNextDueDate(_bidId);

        return dueDate + defaultDuration;
    }
 
    function setRepaymentListenerForBid(uint256 _bidId, address _listener) external {
        uint256 codeSize;
        assembly {
            codeSize := extcodesize(_listener) 
        }
        require(codeSize > 0, "Not a contract");
        address sender = _msgSenderForMarket(bids[_bidId].marketplaceId);

        require(
            sender == getLoanLender(_bidId),
            "Not lender"
        );

        repaymentListenerForBid[_bidId] = _listener;
     }




    function getRepaymentListenerForBid(uint256 _bidId)
        external
        view
        returns (address)
    {
        return repaymentListenerForBid[_bidId];
    }

    // ----------

    function _getBidPaymentCycleType(uint256 _bidId)
        internal
        view
        returns (PaymentCycleType)
    {
         

        return bidPaymentCycleType[_bidId];
    }

    function _getBidPaymentCycleDuration(uint256 _bidId)
        internal
        view
        returns (uint32)
    {
        

        Bid storage bid = bids[_bidId];

        return bid.terms.paymentCycle;
    }

    function _getBidDefaultDuration(uint256 _bidId)
        internal
        view
        returns (uint32)
    {
       

        return bidDefaultDuration[_bidId];
    }



    function _getProtocolFeeRecipient()
    internal view
    returns (address) {

        if (protocolFeeRecipient == address(0x0)){
            return owner();
        }else {
            return protocolFeeRecipient; 
        }

    }

    function getProtocolFeeRecipient()
    external view
    returns (address) {

       return _getProtocolFeeRecipient();

    }

    function setProtocolFeeRecipient(address _recipient)
    external onlyOwner
      {

      protocolFeeRecipient = _recipient;

    }

    // -----

    /** OpenZeppelin Override Functions **/

    function _msgSender()
        internal
        view
        virtual
        override(ERC2771ContextUpgradeable, ContextUpgradeable)
        returns (address sender)
    {
        sender = ERC2771ContextUpgradeable._msgSender();
    }

    function _msgData()
        internal
        view
        virtual
        override(ERC2771ContextUpgradeable, ContextUpgradeable)
        returns (bytes calldata)
    {
        return ERC2771ContextUpgradeable._msgData();
    }
}
