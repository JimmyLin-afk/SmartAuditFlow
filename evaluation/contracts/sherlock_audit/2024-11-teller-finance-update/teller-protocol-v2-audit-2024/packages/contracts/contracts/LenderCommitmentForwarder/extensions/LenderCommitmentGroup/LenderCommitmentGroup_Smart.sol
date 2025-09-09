// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Contracts
import "@openzeppelin/contracts-upgradeable/utils/AddressUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/token/ERC20/IERC20Upgradeable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";


// Interfaces
import "../../../interfaces/ITellerV2Context.sol";
import "../../../interfaces/IProtocolFee.sol";
 
import "../../../interfaces/ITellerV2.sol";

//import "../../../interfaces/IFlashRolloverLoan.sol";
import "../../../libraries/NumbersLib.sol";

import "../../../interfaces/uniswap/IUniswapV3Pool.sol";


import "../../../interfaces/IHasProtocolPausingManager.sol";

import "../../../interfaces/IProtocolPausingManager.sol";



import "../../../interfaces/uniswap/IUniswapV3Factory.sol";
import "../../../interfaces/ISmartCommitmentForwarder.sol";

import "../../../libraries/uniswap/TickMath.sol";
import "../../../libraries/uniswap/FixedPoint96.sol";
import "../../../libraries/uniswap/FullMath.sol";

import {LenderCommitmentGroupShares} from "./LenderCommitmentGroupShares.sol";


import {OracleProtectedChild} from "../../../oracleprotection/OracleProtectedChild.sol";

import { MathUpgradeable } from "@openzeppelin/contracts-upgradeable/utils/math/MathUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/token/ERC20/extensions/IERC20MetadataUpgradeable.sol";

import { CommitmentCollateralType, ISmartCommitment } from "../../../interfaces/ISmartCommitment.sol";
import { ILoanRepaymentListener } from "../../../interfaces/ILoanRepaymentListener.sol";

import { ILoanRepaymentCallbacks } from "../../../interfaces/ILoanRepaymentCallbacks.sol";

import { IEscrowVault } from "../../../interfaces/IEscrowVault.sol";

import { IPausableTimestamp } from "../../../interfaces/IPausableTimestamp.sol";
import { ILenderCommitmentGroup } from "../../../interfaces/ILenderCommitmentGroup.sol";
import { Payment } from "../../../TellerV2Storage.sol";

import {IUniswapPricingLibrary} from "../../../interfaces/IUniswapPricingLibrary.sol";
import {UniswapPricingLibrary} from "../../../libraries/UniswapPricingLibrary.sol";


import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/security/PausableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/security/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
/*
 

 Each LenderCommitmentGroup SmartContract acts as its own Loan Commitment (for the SmartCommitmentForwarder) and acts as its own Lender in the Teller Protocol.

 Lender Users can deposit principal tokens in this contract and this will give them Share Tokens (LP tokens) representing their ownership in the liquidity pool of this contract.

 Borrower Users can borrow principal token funds from this contract (via the SCF contract) by providing collateral tokens in the proper amount as specified by the rules of this smart contract.
 These collateral tokens are then owned by this smart contract and are returned to the borrower via the Teller Protocol rules to the borrower if and only if the borrower repays principal and interest of the loan they took.

 If the borrower defaults on a loan, for 24 hours a liquidation auction is automatically conducted by this smart contract in order to incentivize a liquidator to take the collateral tokens in exchange for principal tokens.

  
 

*/

contract LenderCommitmentGroup_Smart is
    ILenderCommitmentGroup,
    ISmartCommitment,
    ILoanRepaymentListener,
    IPausableTimestamp,
    Initializable,
    OracleProtectedChild,
    OwnableUpgradeable,
    PausableUpgradeable,
    ReentrancyGuardUpgradeable  //adds many storage slots so breaks upgradeability 
{
    using AddressUpgradeable for address;
    using NumbersLib for uint256;

    uint256 public immutable STANDARD_EXPANSION_FACTOR = 1e18;

    uint256 public immutable MIN_TWAP_INTERVAL = 3;

    uint256 public immutable UNISWAP_EXPANSION_FACTOR = 2**96;

    uint256 public immutable EXCHANGE_RATE_EXPANSION_FACTOR = 1e36;  

    using SafeERC20 for IERC20;

    /// @custom:oz-upgrades-unsafe-allow state-variable-immutable
    address public immutable TELLER_V2;
    address public immutable SMART_COMMITMENT_FORWARDER;
    address public immutable UNISWAP_V3_FACTORY;
    //address private __UNISWAP_V3_POOL; //deprecated
 
    LenderCommitmentGroupShares public poolSharesToken;

    IERC20 public principalToken;
    IERC20 public collateralToken;
    //uint24 private __uniswapPoolFee; //deprecated

    uint256 marketId;


    uint256 public totalPrincipalTokensCommitted; 
    uint256 public totalPrincipalTokensWithdrawn;

    uint256 public totalPrincipalTokensLended;
    uint256 public totalPrincipalTokensRepaid; //subtract this and the above to find total principal tokens outstanding for loans

    uint256 public totalInterestCollected;

    uint16 public liquidityThresholdPercent; //max ratio of principal allowed to be borrowed vs escrowed  //  maximum of 10000 (100%)
    uint16 public collateralRatio; //the overcollateralization ratio, typically >100 pct

    //uint32 private __twapInterval; //deprecated
    uint32 public maxLoanDuration;
    uint16 public interestRateLowerBound;
    uint16 public interestRateUpperBound;




    uint256 immutable public DEFAULT_WITHDRAW_DELAY_TIME_SECONDS = 300;
    uint256 immutable public MAX_WITHDRAW_DELAY_TIME = 86400;

    //mapping(address => uint256) public principalTokensCommittedByLender;
    mapping(uint256 => bool) public activeBids;

    //this excludes interest
    // maybe it is possible to get rid of this storage slot and calculate it from totalPrincipalTokensRepaid, totalPrincipalTokensLended
    int256 tokenDifferenceFromLiquidations;

    bool public firstDepositMade;
    uint256 public withdrawDelayTimeSeconds; 

    IUniswapPricingLibrary.PoolRouteConfig[]  public  poolOracleRoutes;

    //configured by the owner. If 0 , not used. 
    uint256 public maxPrincipalPerCollateralAmount; 


    uint256 public lastUnpausedAt;
   

    event PoolInitialized(
        address indexed principalTokenAddress,
        address indexed collateralTokenAddress,
        uint256 marketId,
        uint32 maxLoanDuration,
        uint16 interestRateLowerBound,
        uint16 interestRateUpperBound,
        uint16 liquidityThresholdPercent,
        uint16 loanToValuePercent,
      //  uint24 uniswapPoolFee,
      //  uint32 twapInterval,
        address poolSharesToken
    );

    event LenderAddedPrincipal(
        address indexed lender,
        uint256 amount,
        uint256 sharesAmount,
        address indexed sharesRecipient
    );

    event BorrowerAcceptedFunds(
        address indexed borrower,
        uint256 indexed bidId,
        uint256 principalAmount,
        uint256 collateralAmount,
        uint32 loanDuration,
        uint16 interestRate
    );

    event EarningsWithdrawn(
        address indexed lender,
        uint256 amountPoolSharesTokens,
        uint256 principalTokensWithdrawn,
        address indexed recipient
    );


    event DefaultedLoanLiquidated(
        uint256 indexed bidId,
        address indexed liquidator,
        uint256 amountDue, 
        int256 tokenAmountDifference 
    );


    event LoanRepaid(
        uint256 indexed bidId,
        address indexed repayer,
        uint256 principalAmount,
        uint256 interestAmount,
        uint256 totalPrincipalRepaid,
        uint256 totalInterestCollected
    );


    event WithdrawFromEscrow(
       
        uint256  indexed amount

    );


    modifier onlySmartCommitmentForwarder() {
        require(
            msg.sender == address(SMART_COMMITMENT_FORWARDER),
            "Can only be called by Smart Commitment Forwarder"
        );
        _;
    }

    modifier onlyTellerV2() {
        require(
            msg.sender == address(TELLER_V2),
            "Can only be called by TellerV2"
        );
        _;
    }


    modifier onlyProtocolOwner() {
        require(
            msg.sender == Ownable(address(TELLER_V2)).owner(),
            "Not Protocol Owner"
        );
        _;
    }

    modifier onlyProtocolPauser() {

        address pausingManager = IHasProtocolPausingManager( address(TELLER_V2) ).getProtocolPausingManager();

        require(
           IProtocolPausingManager( pausingManager ).isPauser(msg.sender)  ,
            "Not Owner or Protocol Owner"
        );
        _;
    }

    modifier bidIsActiveForGroup(uint256 _bidId) {
        require(activeBids[_bidId] == true, "Bid is not active for group");

        _;
    }
 

    modifier whenForwarderNotPaused() {
         require( PausableUpgradeable(address(SMART_COMMITMENT_FORWARDER)).paused() == false , "Smart Commitment Forwarder is paused");
        _;
    }



    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor(
        address _tellerV2,
        address _smartCommitmentForwarder,
        address _uniswapV3Factory
    ) OracleProtectedChild(_smartCommitmentForwarder) {
        TELLER_V2 = _tellerV2;
        SMART_COMMITMENT_FORWARDER = _smartCommitmentForwarder;
        UNISWAP_V3_FACTORY = _uniswapV3Factory;
    }

    /**
     * @notice Initializes the LenderCommitmentGroup_Smart contract.
     * @param _commitmentGroupConfig Configuration for the commitment group (lending pool).
     * @param _poolOracleRoutes Route configuration for the principal/collateral oracle.
     * @return poolSharesToken_ Address of the deployed pool shares token.
     */
   function initialize(
       CommitmentGroupConfig calldata _commitmentGroupConfig,
       IUniswapPricingLibrary.PoolRouteConfig[] calldata _poolOracleRoutes
    ) external initializer returns (address poolSharesToken_) {
       
        __Ownable_init();
        __Pausable_init();

        principalToken = IERC20(_commitmentGroupConfig.principalTokenAddress);
        collateralToken = IERC20(_commitmentGroupConfig.collateralTokenAddress);
         
        marketId = _commitmentGroupConfig.marketId;

        withdrawDelayTimeSeconds = DEFAULT_WITHDRAW_DELAY_TIME_SECONDS;

        //in order for this to succeed, first, the SmartCommitmentForwarder needs to be a trusted forwarder for the market         
        ITellerV2Context(TELLER_V2).approveMarketForwarder(
            _commitmentGroupConfig.marketId,
            SMART_COMMITMENT_FORWARDER
        );

        maxLoanDuration = _commitmentGroupConfig.maxLoanDuration;
        interestRateLowerBound = _commitmentGroupConfig.interestRateLowerBound;
        interestRateUpperBound = _commitmentGroupConfig.interestRateUpperBound;

        require(interestRateLowerBound <= interestRateUpperBound, "invalid _interestRateLowerBound");

       
        liquidityThresholdPercent = _commitmentGroupConfig.liquidityThresholdPercent;
        collateralRatio = _commitmentGroupConfig.collateralRatio;
      
        require( liquidityThresholdPercent <= 10000, "invalid _liquidityThresholdPercent"); 

        for (uint256 i = 0; i < _poolOracleRoutes.length; i++) {
            poolOracleRoutes.push(_poolOracleRoutes[i]);
        }


         require(poolOracleRoutes.length >= 1 && poolOracleRoutes.length <= 2, "invalid pool routes length");
        
        poolSharesToken_ = _deployPoolSharesToken();


        emit PoolInitialized(
            _commitmentGroupConfig.principalTokenAddress,
            _commitmentGroupConfig.collateralTokenAddress,
            _commitmentGroupConfig.marketId,
            _commitmentGroupConfig.maxLoanDuration,
            _commitmentGroupConfig.interestRateLowerBound,
            _commitmentGroupConfig.interestRateUpperBound,
            _commitmentGroupConfig.liquidityThresholdPercent,
            _commitmentGroupConfig.collateralRatio,
            //_commitmentGroupConfig.uniswapPoolFee,
            //_commitmentGroupConfig.twapInterval,
            poolSharesToken_
        );
    }


    /**
     * @notice Sets the delay time for withdrawing funds. Only Protocol Owner.
     * @param _seconds Delay time in seconds.
     */
    function setWithdrawDelayTime(uint256 _seconds) 
    external 
    onlyProtocolOwner {
        require( _seconds < MAX_WITHDRAW_DELAY_TIME );

        withdrawDelayTimeSeconds = _seconds;
    }


    /**
     * @notice Sets an optional manual ratio for principal/collateral ratio for borrowers. Only Pool Owner.
     * @param _maxPrincipalPerCollateralAmount Price ratio, expanded to support sub-one ratios.
     */
    function setMaxPrincipalPerCollateralAmount(uint256 _maxPrincipalPerCollateralAmount) 
    external 
    onlyOwner {
       maxPrincipalPerCollateralAmount = _maxPrincipalPerCollateralAmount;
    }

     /**
     * @notice Deploys the pool shares token for the lending pool.
     * @dev This function can only be called during initialization.
     * @return poolSharesToken_ Address of the deployed pool shares token.
     */
    function _deployPoolSharesToken()
        internal
        onlyInitializing
        returns (address poolSharesToken_)
    {      
        require(
            address(poolSharesToken) == address(0),
            "Pool shares already deployed"
        );
 
        poolSharesToken = new LenderCommitmentGroupShares(
            "LenderGroupShares",
            "SHR",
            18  
        );

        return address(poolSharesToken);
    } 


    /**
     * @notice This determines the number of shares you get for depositing principal tokens and the number of principal tokens you receive for burning shares
     * @return rate_ The current exchange rate, scaled by the EXCHANGE_RATE_FACTOR.
     */

    function sharesExchangeRate() public view virtual returns (uint256 rate_) {
        

        uint256 poolTotalEstimatedValue = getPoolTotalEstimatedValue();

        if (poolSharesToken.totalSupply() == 0) {
            return EXCHANGE_RATE_EXPANSION_FACTOR; // 1 to 1 for first swap
        }

        rate_ =
            MathUpgradeable.mulDiv(poolTotalEstimatedValue , 
                EXCHANGE_RATE_EXPANSION_FACTOR ,
                  poolSharesToken.totalSupply() );
    }

    function sharesExchangeRateInverse()
        public
        view
        virtual
        returns (uint256 rate_)
    {
        return
            (EXCHANGE_RATE_EXPANSION_FACTOR * EXCHANGE_RATE_EXPANSION_FACTOR) /
            sharesExchangeRate();
    }

    function getPoolTotalEstimatedValue()
        public
        view
        returns (uint256 poolTotalEstimatedValue_)
    {
       
         int256 poolTotalEstimatedValueSigned = int256(totalPrincipalTokensCommitted) 
         + int256(totalInterestCollected)  + int256(tokenDifferenceFromLiquidations) 
         - int256(totalPrincipalTokensWithdrawn);

        //if the poolTotalEstimatedValue_ is less than 0, we treat it as 0.  
        poolTotalEstimatedValue_ = poolTotalEstimatedValueSigned > int256(0)
            ? uint256(poolTotalEstimatedValueSigned)
            : 0;
    }

    /**
     * @notice Adds principal to the lending pool in exchange for shares.
     * @param _amount Amount of principal tokens to deposit.
     * @param _sharesRecipient Address receiving the shares.
     * @param _minSharesAmountOut Minimum amount of shares expected.
     * @return sharesAmount_ Amount of shares minted.
     */
    function addPrincipalToCommitmentGroup(
        uint256 _amount,
        address _sharesRecipient,
        uint256 _minSharesAmountOut
    ) external whenForwarderNotPaused whenNotPaused nonReentrant onlyOracleApprovedAllowEOA 
    returns (uint256 sharesAmount_) {
       
        uint256 principalTokenBalanceBefore = principalToken.balanceOf(address(this));

        principalToken.safeTransferFrom(msg.sender, address(this), _amount);
 
        uint256 principalTokenBalanceAfter = principalToken.balanceOf(address(this));
 
        require( principalTokenBalanceAfter == principalTokenBalanceBefore + _amount, "Token balance was not added properly" );

        sharesAmount_ = _valueOfUnderlying(_amount, sharesExchangeRate());
 
        totalPrincipalTokensCommitted += _amount;
         
        //mint shares equal to _amount and give them to the shares recipient 
        poolSharesToken.mint(_sharesRecipient, sharesAmount_);
   
        emit LenderAddedPrincipal( 

            msg.sender,
            _amount,
            sharesAmount_,
            _sharesRecipient

         );

        require( sharesAmount_ >= _minSharesAmountOut, "Invalid: Min Shares AmountOut" );
 
         if(!firstDepositMade){
            require(msg.sender == owner(), "Owner must initialize the pool with a deposit first.");
            require( sharesAmount_>= 1e6, "Initial shares amount must be atleast 1e6" );

            firstDepositMade = true;
        }
    }

    function _valueOfUnderlying(uint256 amount, uint256 rate)
        internal
        pure
        returns (uint256 value_)
    {
        if (rate == 0) {
            return 0;
        }

        value_ = MathUpgradeable.mulDiv(amount ,  EXCHANGE_RATE_EXPANSION_FACTOR   ,  rate );
    }

    /**
 * @notice Validates loan parameters and starts the TellerV2 Loan where this contract as the lender.
 * @dev Must be called via the Smart Commitment Forwarder 
 * @param _borrower Address of the borrower accepting the loan.
 * @param _bidId Identifier for the loan bid.
 * @param _principalAmount Amount of principal being lent.
 * @param _collateralAmount Amount of collateral provided by the borrower.
 * @param _collateralTokenAddress Address of the collateral token contract.
 * @param _collateralTokenId Token ID of the collateral (if applicable).
 * @param _loanDuration Duration of the loan in seconds.
 * @param _interestRate Interest rate for the loan, scaled by 100 (e.g., 500 = 5%).
 */
    function acceptFundsForAcceptBid(
        address _borrower,
        uint256 _bidId,
        uint256 _principalAmount,
        uint256 _collateralAmount,
        address _collateralTokenAddress,
        uint256 _collateralTokenId, 
        uint32 _loanDuration,
        uint16 _interestRate
    ) external onlySmartCommitmentForwarder whenForwarderNotPaused whenNotPaused {
        
        require(
            _collateralTokenAddress == address(collateralToken),
            "Mismatching collateral token"
        );
        //the interest rate must be at least as high has the commitment demands. The borrower can use a higher interest rate although that would not be beneficial to the borrower.
        require(_interestRate >= getMinInterestRate(_principalAmount), "Invalid interest rate");
        //the loan duration must be less than the commitment max loan duration. The lender who made the commitment expects the money to be returned before this window.
        require(_loanDuration <= maxLoanDuration, "Invalid loan max duration");

        require(
            getPrincipalAmountAvailableToBorrow() >= _principalAmount,
            "Invalid loan max principal"
        );
 
 
        uint256 requiredCollateral = calculateCollateralRequiredToBorrowPrincipal(
            _principalAmount
        );



        require(    
             _collateralAmount   >=
                requiredCollateral,
            "Insufficient Borrower Collateral"
        );
 
        principalToken.approve(address(TELLER_V2), _principalAmount);

        //do not have to override msg.sender as this contract is the lender !
        _acceptBidWithRepaymentListener(_bidId);

        totalPrincipalTokensLended += _principalAmount;

        activeBids[_bidId] = true; //bool for now
        

        emit BorrowerAcceptedFunds(  
            _borrower,
            _bidId,
            _principalAmount,
            _collateralAmount, 
            _loanDuration,
            _interestRate 
         );
    }

    /**
    * @notice Internal function to accept a loan bid and set a repayment listener.
    * @dev Interacts with the Teller Protocol to accept the bid and register the repayment listener.
    * @param _bidId Identifier for the loan bid being accepted.
    */
    function _acceptBidWithRepaymentListener(uint256 _bidId) internal {
        ITellerV2(TELLER_V2).lenderAcceptBid(_bidId); //this gives out the funds to the borrower

        ILoanRepaymentCallbacks(TELLER_V2).setRepaymentListenerForBid(
            _bidId,
            address(this)
        );

        
    }

    
     function prepareSharesForBurn(
        uint256 _amountPoolSharesTokens 
    ) external whenForwarderNotPaused whenNotPaused nonReentrant
     returns (bool) {
        
        return poolSharesToken.prepareSharesForBurn(msg.sender, _amountPoolSharesTokens); 
    }




   /**
    * @notice Burns shares to withdraw an equivalent amount of principal tokens.
    * @dev Requires shares to have been prepared for withdrawal in advance.
    * @param _amountPoolSharesTokens Amount of pool shares to burn.
    * @param _recipient Address receiving the withdrawn principal tokens.
    * @param _minAmountOut Minimum amount of principal tokens expected to be withdrawn.
    * @return principalTokenValueToWithdraw Amount of principal tokens withdrawn.
    */
    function burnSharesToWithdrawEarnings(
        uint256 _amountPoolSharesTokens,
        address _recipient,
        uint256 _minAmountOut
    ) external whenForwarderNotPaused whenNotPaused  nonReentrant onlyOracleApprovedAllowEOA 
    returns (uint256) {
       
        //require(poolSharesPreparedToWithdrawForLender[msg.sender] >= _amountPoolSharesTokens,"Shares not prepared for withdraw");
       // require(poolSharesPreparedTimestamp[msg.sender] <= block.timestamp - withdrawlDelayTimeSeconds,"Shares not prepared for withdraw");
         
       
        //this should compute BEFORE shares burn 
        uint256 principalTokenValueToWithdraw = _valueOfUnderlying(
            _amountPoolSharesTokens,
            sharesExchangeRateInverse()
        );

        poolSharesToken.burn(msg.sender, _amountPoolSharesTokens, withdrawDelayTimeSeconds);

        totalPrincipalTokensWithdrawn += principalTokenValueToWithdraw;

        principalToken.safeTransfer(_recipient, principalTokenValueToWithdraw);


        emit EarningsWithdrawn(
            msg.sender,
            _amountPoolSharesTokens,
            principalTokenValueToWithdraw,
            _recipient
        );
        
        require( principalTokenValueToWithdraw >=  _minAmountOut ,"Invalid: Min Amount Out");

        return principalTokenValueToWithdraw;
    }

/**
 * @notice Liquidates a defaulted loan using a reverse auction that starts high and falls to zero.
 * @dev The amount of tokens withdrawn from the liquidator is always the sum of _tokenAmountDifference + amountDue .
 * @dev Handles the liquidation process for a defaulted loan bid, ensuring all conditions for liquidation are met.
 * @param _bidId Identifier for the defaulted loan bid.
 * @param _tokenAmountDifference The incentive difference in tokens required for liquidation. Positive values indicate extra tokens to take, and negative values indicate extra tokens to give.
 */


    function liquidateDefaultedLoanWithIncentive(
        uint256 _bidId,
        int256 _tokenAmountDifference
    ) external whenForwarderNotPaused whenNotPaused bidIsActiveForGroup(_bidId) nonReentrant onlyOracleApprovedAllowEOA {
        
        //use original principal amount as amountDue

        uint256 amountDue = _getAmountOwedForBid(_bidId);
        

        uint256 loanDefaultedTimeStamp = ITellerV2(TELLER_V2)
            .getLoanDefaultTimestamp(_bidId);

        uint256 loanDefaultedOrUnpausedAtTimeStamp = Math.max(
            loanDefaultedTimeStamp,
            getLastUnpausedAt()
        );

        int256 minAmountDifference = getMinimumAmountDifferenceToCloseDefaultedLoan(
                amountDue,
                loanDefaultedOrUnpausedAtTimeStamp
            );

        require(
            _tokenAmountDifference >= minAmountDifference,
            "Insufficient tokenAmountDifference"
        );


        if (minAmountDifference > 0) {
            //this is used when the collateral value is higher than the principal (rare)
            //the loan will be completely made whole and our contract gets extra funds too
            uint256 tokensToTakeFromSender = abs(minAmountDifference);
        
        
           uint256 liquidationProtocolFee = Math.mulDiv( 
                tokensToTakeFromSender , 
                ISmartCommitmentForwarder(SMART_COMMITMENT_FORWARDER)
                    .getLiquidationProtocolFeePercent(),
                 10000)  ;
           

            IERC20(principalToken).safeTransferFrom(
                msg.sender,
                address(this),
                amountDue + tokensToTakeFromSender - liquidationProtocolFee
            ); 
             
            address protocolFeeRecipient = ITellerV2(address(TELLER_V2)).getProtocolFeeRecipient();

              IERC20(principalToken).safeTransferFrom(
                msg.sender,
                address(protocolFeeRecipient),
                 liquidationProtocolFee
            );

            totalPrincipalTokensRepaid += amountDue;
            tokenDifferenceFromLiquidations += int256(tokensToTakeFromSender - liquidationProtocolFee );


        } else {
          
           
            uint256 tokensToGiveToSender = abs(minAmountDifference);

           
            IERC20(principalToken).safeTransferFrom(
                msg.sender,
                address(this),
                amountDue - tokensToGiveToSender  
            );

            totalPrincipalTokensRepaid += amountDue;

            //this will make tokenDifference go more negative
            tokenDifferenceFromLiquidations -= int256(tokensToGiveToSender);

           
        }

        //this will give collateral to the caller
        ITellerV2(TELLER_V2).lenderCloseLoanWithRecipient(_bidId, msg.sender);
    
    
         emit DefaultedLoanLiquidated(
            _bidId,
            msg.sender,
            amountDue, 
            _tokenAmountDifference
        );
    }


    function getLastUnpausedAt() 
    public view 
    returns (uint256) {


        return Math.max(
            lastUnpausedAt,
            IPausableTimestamp(SMART_COMMITMENT_FORWARDER).getLastUnpausedAt() //this counts tellerV2 pausing
            )
        ;
 

    }


    function setLastUnpausedAt() internal {
        lastUnpausedAt =  block.timestamp;
    }

    

    function _getAmountOwedForBid(uint256 _bidId )
        internal
        view
        virtual
        returns (uint256 amountDue)
    {
        (,,,, amountDue, , ,  )
         = ITellerV2(TELLER_V2).getLoanSummary(_bidId);

       
    }


    function getTokenDifferenceFromLiquidations() public view returns (int256){

        return tokenDifferenceFromLiquidations;

    }
    

    /*
       * @dev This function will calculate the incentive amount (using a uniswap bonus plus a timer)
        of principal tokens that will be given to incentivize liquidating a loan 

       * @dev As time approaches infinite, the output approaches -1 * AmountDue .  
    */
    function getMinimumAmountDifferenceToCloseDefaultedLoan(
        uint256 _amountOwed,
        uint256 _loanDefaultedTimestamp
    ) public view virtual returns (int256 amountDifference_) {
        require(
            _loanDefaultedTimestamp > 0,
            "Loan defaulted timestamp must be greater than zero"
        );
        require(
            block.timestamp > _loanDefaultedTimestamp,
            "Loan defaulted timestamp must be in the past"
        );

        uint256 secondsSinceDefaulted = block.timestamp -
            _loanDefaultedTimestamp;

        //this starts at 764% and falls to -100% 
        int256 incentiveMultiplier = int256(86400 - 10000) -
            int256(secondsSinceDefaulted);

        if (incentiveMultiplier < -10000) {
            incentiveMultiplier = -10000;
        }

        amountDifference_ =
            (int256(_amountOwed) * incentiveMultiplier) /
            int256(10000);
    }

    function abs(int x) private pure returns (uint) {
        return x >= 0 ? uint(x) : uint(-x);
    }


    function calculateCollateralRequiredToBorrowPrincipal(  
        uint256 _principalAmount
    ) public
        view
        virtual
        returns (uint256) {

        uint256 baseAmount = calculateCollateralTokensAmountEquivalentToPrincipalTokens(
                _principalAmount
        ); 

        //this is an amount of collateral
        return baseAmount.percent(collateralRatio);
    }

    /* 
    * @dev this is expanded by 10e18
    * @dev this logic is very similar to that used in LCFA 
    */
    function calculateCollateralTokensAmountEquivalentToPrincipalTokens(
        uint256 principalAmount 
    ) public view virtual returns (uint256 collateralTokensAmountToMatchValue) {
   
        uint256 pairPriceWithTwapFromOracle = UniswapPricingLibrary
            .getUniswapPriceRatioForPoolRoutes(poolOracleRoutes);
       
       
        uint256 principalPerCollateralAmount = maxPrincipalPerCollateralAmount == 0  
                ? pairPriceWithTwapFromOracle   
                : Math.min(
                    pairPriceWithTwapFromOracle,
                    maxPrincipalPerCollateralAmount //this is expanded by uniswap exp factor  
                );


        return
            getRequiredCollateral(
                principalAmount,
                principalPerCollateralAmount   
            );
    }


     function getUniswapPriceRatioForPoolRoutes(
       IUniswapPricingLibrary.PoolRouteConfig[] memory poolOracleRoutes
    ) external view virtual returns (uint256 ) {
   
        uint256 pairPriceWithTwapFromOracle = UniswapPricingLibrary
            .getUniswapPriceRatioForPoolRoutes(poolOracleRoutes);
       

        return pairPriceWithTwapFromOracle;
    }

      function getPrincipalForCollateralForPoolRoutes(
        IUniswapPricingLibrary.PoolRouteConfig[] memory poolOracleRoutes
    ) external view virtual returns (uint256 ) {
   
        uint256 pairPriceWithTwapFromOracle = UniswapPricingLibrary
            .getUniswapPriceRatioForPoolRoutes(poolOracleRoutes);
       
       
        uint256 principalPerCollateralAmount = maxPrincipalPerCollateralAmount == 0  
                ? pairPriceWithTwapFromOracle   
                : Math.min(
                    pairPriceWithTwapFromOracle,
                    maxPrincipalPerCollateralAmount //this is expanded by uniswap exp factor  
                );


        return principalPerCollateralAmount;
    }


   function getRequiredCollateral(
        uint256 _principalAmount,
        uint256 _maxPrincipalPerCollateralAmount 
        
    ) public view virtual returns (uint256) {
         
         return
            MathUpgradeable.mulDiv(
                _principalAmount,
                STANDARD_EXPANSION_FACTOR,
                _maxPrincipalPerCollateralAmount,
                MathUpgradeable.Rounding.Up
            );  
    }
 

    /*
    This  callback occurs when a TellerV2 repayment happens or when a TellerV2 liquidate happens 

    lenderCloseLoan does not trigger a repayLoanCallback 

    It is important that only teller loans FOR THIS POOL can call this !
     */
    function repayLoanCallback(
        uint256 _bidId,
        address repayer,
        uint256 principalAmount,
        uint256 interestAmount
    ) external onlyTellerV2 whenForwarderNotPaused whenNotPaused bidIsActiveForGroup(_bidId) { 
        totalPrincipalTokensRepaid += principalAmount;
        totalInterestCollected += interestAmount;

         emit LoanRepaid(
            _bidId,
            repayer,
            principalAmount,
            interestAmount,
            totalPrincipalTokensRepaid,
            totalInterestCollected
        );
    }


    /*
        If principaltokens get stuck in the escrow vault for any reason, anyone may
        call this function to move them from that vault in to this contract 

        @dev there is no need to increment totalPrincipalTokensRepaid here as that is accomplished by the repayLoanCallback
    */
    function withdrawFromEscrowVault ( uint256 _amount ) external whenForwarderNotPaused whenNotPaused {

        address _escrowVault = ITellerV2(TELLER_V2).getEscrowVault();

        IEscrowVault(_escrowVault).withdraw(address(principalToken), _amount );  

        emit WithdrawFromEscrow(_amount);

    }
 
  
    function getTotalPrincipalTokensOutstandingInActiveLoans()
        public
        view
        returns (uint256)
    {
        return totalPrincipalTokensLended - totalPrincipalTokensRepaid;
    }




    function getCollateralTokenAddress() external view returns (address) {
        return address(collateralToken);
    }

    function getCollateralTokenId() external view returns (uint256) {
        return 0;
    }

    function getCollateralTokenType()
        external
        view
        returns (CommitmentCollateralType)
    {
        return CommitmentCollateralType.ERC20;
    }
 

    function getMarketId() external view returns (uint256) {
        return marketId;
    }

    function getMaxLoanDuration() external view returns (uint32) {
        return maxLoanDuration;
    }


 function getPoolUtilizationRatio(uint256 activeLoansAmountDelta ) public view returns (uint16) {

        if (getPoolTotalEstimatedValue() == 0) {
            return 0;
        }

        return uint16(  Math.min(                          
                            MathUpgradeable.mulDiv( 
                                (getTotalPrincipalTokensOutstandingInActiveLoans() + activeLoansAmountDelta), 
                                10000  ,
                                getPoolTotalEstimatedValue() ) , 
                        10000  ));

    }

  function getMinInterestRate(uint256 amountDelta) public view returns (uint16) {
        return interestRateLowerBound + 
        uint16( uint256(interestRateUpperBound-interestRateLowerBound)
        .percent(getPoolUtilizationRatio(amountDelta )
        
        ) );
    } 
    

    function getPrincipalTokenAddress() external view returns (address) {
        return address(principalToken);
    }

   

    function getPrincipalAmountAvailableToBorrow()
        public
        view
        returns (uint256)
    {     

            return  ( uint256( getPoolTotalEstimatedValue() )).percent(liquidityThresholdPercent) -
            getTotalPrincipalTokensOutstandingInActiveLoans();
     
    }

    /**
     * @notice Lets the DAO/owner of the protocol implement an emergency stop mechanism.
     */
    function pauseLendingPool() public virtual onlyProtocolPauser whenNotPaused {
        _pause();
    }

    /**
     * @notice Lets the DAO/owner of the protocol undo a previously implemented emergency stop.
     */
    function unpauseLendingPool() public virtual onlyProtocolPauser whenPaused {
        setLastUnpausedAt();
        _unpause();
    }
}
