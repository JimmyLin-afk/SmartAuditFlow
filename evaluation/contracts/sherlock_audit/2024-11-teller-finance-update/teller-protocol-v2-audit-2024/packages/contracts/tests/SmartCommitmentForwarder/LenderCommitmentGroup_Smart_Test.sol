import { Testable } from "../Testable.sol";

import { LenderCommitmentGroup_Smart_Override } from "./LenderCommitmentGroup_Smart_Override.sol";

import {TestERC20Token} from "../tokens/TestERC20Token.sol";


import {MarketRegistry} from "../../contracts/MarketRegistry.sol";
import {SmartCommitmentForwarder} from "../../contracts/LenderCommitmentForwarder/SmartCommitmentForwarder.sol";
import {TellerV2SolMock} from "../../contracts/mock/TellerV2SolMock.sol";
import {UniswapV3PoolMock} from "../../contracts/mock/uniswap/UniswapV3PoolMock.sol";
import {UniswapV3FactoryMock} from "../../contracts/mock/uniswap/UniswapV3FactoryMock.sol";
import { PaymentType, PaymentCycleType } from "../../contracts/libraries/V2Calculations.sol";
import { LoanDetails, Payment, BidState , Bid, Terms } from "../../contracts/TellerV2Storage.sol";

import { ILenderCommitmentGroup } from "../../contracts/interfaces/ILenderCommitmentGroup.sol";
import { IUniswapPricingLibrary } from "../../contracts/interfaces/IUniswapPricingLibrary.sol";

import {ProtocolPausingManager} from "../../contracts/pausing/ProtocolPausingManager.sol";


import "lib/forge-std/src/console.sol";

//contract LenderCommitmentGroup_Smart_Mock is ExtensionsContextUpgradeable {}

/*
  

Write tests for a borrower . borrowing money from the group 



- write tests for the LTV ratio and make sure that is working as expected (mock) 
- write tests for the global liquidityThresholdPercent and built functionality for a user-specific liquidityThresholdPercent based on signalling shares.



-write a test that ensures that adding principal then removing it will mean that totalPrincipalCommitted is the net amount 
 

*/

contract LenderCommitmentGroup_Smart_Test is Testable {
    constructor() {}

    User private extensionContract;

    User private borrower;
    User private lender;
    User private liquidator;

    TestERC20Token principalToken;

    TestERC20Token collateralToken;

    LenderCommitmentGroup_Smart_Override lenderCommitmentGroupSmart;

    MarketRegistry _marketRegistry;
    TellerV2SolMock _tellerV2;
    SmartCommitmentForwarder _smartCommitmentForwarder;
    UniswapV3PoolMock _uniswapV3Pool;
    UniswapV3FactoryMock _uniswapV3Factory;

    function setUp() public {
        borrower = new User();
        lender = new User();
        liquidator = new User();

        _tellerV2 = new TellerV2SolMock();
        _marketRegistry = new MarketRegistry();
        _smartCommitmentForwarder = new SmartCommitmentForwarder(
            address(_tellerV2),address(_marketRegistry));
         
        _uniswapV3Pool = new UniswapV3PoolMock();

        _uniswapV3Factory = new UniswapV3FactoryMock();
        _uniswapV3Factory.setPoolMock(address(_uniswapV3Pool));


        ProtocolPausingManager protocolPausingManager = new ProtocolPausingManager();
        protocolPausingManager.initialize();

        _tellerV2.setProtocolPausingManager(address(protocolPausingManager));
      


        principalToken = new TestERC20Token("wrappedETH", "WETH", 1e24, 18);

        collateralToken = new TestERC20Token("PEPE", "pepe", 1e24, 18);

        principalToken.transfer(address(lender), 1e18);
        collateralToken.transfer(address(borrower), 1e18);


        _uniswapV3Pool.set_mockToken0(address(principalToken));
        _uniswapV3Pool.set_mockToken1(address(collateralToken));

        lenderCommitmentGroupSmart = new LenderCommitmentGroup_Smart_Override(
            address(_tellerV2),
            address(_smartCommitmentForwarder),
            address(_uniswapV3Factory)
        );
    }

    function initialize_group_contract() public {
        address _principalTokenAddress = address(principalToken);
        address _collateralTokenAddress = address(collateralToken);
        uint256 _marketId = 1;
        uint32 _maxLoanDuration = 5000000;
        uint16 _interestRateLowerBound = 0;
        uint16 _interestRateUpperBound = 800;
        uint16 _liquidityThresholdPercent = 10000;
        uint16 _collateralRatio = 10000;
       // uint24 _uniswapPoolFee = 3000;
       // uint32 _twapInterval = 5;

         ILenderCommitmentGroup.CommitmentGroupConfig memory groupConfig = ILenderCommitmentGroup.CommitmentGroupConfig({
            principalTokenAddress: _principalTokenAddress,
            collateralTokenAddress: _collateralTokenAddress,
            marketId: _marketId,
            maxLoanDuration: _maxLoanDuration,
            interestRateLowerBound: _interestRateLowerBound,
            interestRateUpperBound: _interestRateUpperBound,
            liquidityThresholdPercent: _liquidityThresholdPercent,
            collateralRatio: _collateralRatio
           // uniswapPoolFee: _uniswapPoolFee,
           // twapInterval: _twapInterval
        });

          bool zeroForOne = false;
          uint32 twapInterval = 0;


          IUniswapPricingLibrary.PoolRouteConfig
            memory routeConfig = IUniswapPricingLibrary.PoolRouteConfig({
                pool: address(_uniswapV3Pool),
                zeroForOne: zeroForOne,
                twapInterval: twapInterval,
                token0Decimals: 18,
                token1Decimals: 18
            });


       IUniswapPricingLibrary.PoolRouteConfig[]
            memory routesConfig = new IUniswapPricingLibrary.PoolRouteConfig[](
                1
            );

        routesConfig[0] = routeConfig; 


        address _poolSharesToken = lenderCommitmentGroupSmart.initialize(
            groupConfig,
            routesConfig
        );

        lenderCommitmentGroupSmart.mock_setFirstDepositMade(true);
    }

    function test_initialize() public {
        address _principalTokenAddress = address(principalToken);
        address _collateralTokenAddress = address(collateralToken);
        uint256 _marketId = 1;
        uint32 _maxLoanDuration = 5000000;
        uint16 _interestRateLowerBound = 100;
        uint16 _interestRateUpperBound = 800;
        uint16 _liquidityThresholdPercent = 10000;
        uint16 _collateralRatio = 10000;
      //  uint24 _uniswapPoolFee = 3000;
      //  uint32 _twapInterval = 5;


      ILenderCommitmentGroup.CommitmentGroupConfig memory groupConfig = ILenderCommitmentGroup.CommitmentGroupConfig({
            principalTokenAddress: _principalTokenAddress,
            collateralTokenAddress: _collateralTokenAddress,
            marketId: _marketId,
            maxLoanDuration: _maxLoanDuration,
            interestRateLowerBound: _interestRateLowerBound,
            interestRateUpperBound: _interestRateUpperBound,
            liquidityThresholdPercent: _liquidityThresholdPercent,
            collateralRatio: _collateralRatio
          //  uniswapPoolFee: _uniswapPoolFee,
          //  twapInterval: _twapInterval
        });

       bool zeroForOne = false;
      uint32 twapInterval = 0;


          IUniswapPricingLibrary.PoolRouteConfig
            memory routeConfig = IUniswapPricingLibrary.PoolRouteConfig({
                pool: address(_uniswapV3Pool),
                zeroForOne: zeroForOne,
                twapInterval: twapInterval,
                token0Decimals: 18,
                token1Decimals: 18
            });



       IUniswapPricingLibrary.PoolRouteConfig[]
            memory routesConfig = new IUniswapPricingLibrary.PoolRouteConfig[](
                1
            );

        routesConfig[0] = routeConfig; 


        address _poolSharesToken = lenderCommitmentGroupSmart.initialize(
             groupConfig,
            routesConfig
        );

        // assertFalse(isTrustedBefore, "Should not be trusted forwarder before");
        // assertTrue(isTrustedAfter, "Should be trusted forwarder after");
    }

    //  https://github.com/teller-protocol/teller-protocol-v1/blob/develop/contracts/lending/ttoken/TToken_V3.sol
    function test_addPrincipalToCommitmentGroup() public {
        //principalToken.transfer(address(lenderCommitmentGroupSmart), 1e18);
        //collateralToken.transfer(address(lenderCommitmentGroupSmart), 1e18);
        lenderCommitmentGroupSmart.set_mockSharesExchangeRate(1e36);

        initialize_group_contract();

        vm.prank(address(lender));
        principalToken.approve(address(lenderCommitmentGroupSmart), 1000000);

        uint256 minAmountOut = 1000000;

        vm.prank(address(lender));
        uint256 sharesAmount_ = lenderCommitmentGroupSmart
            .addPrincipalToCommitmentGroup(1000000, address(borrower), minAmountOut);

        uint256 expectedSharesAmount = 1000000;

        //use ttoken logic to make this better
        assertEq(
            sharesAmount_,
            expectedSharesAmount,
            "Received an unexpected amount of shares"
        );
    }

    function test_addPrincipalToCommitmentGroup_after_interest_payments()
        public
    {
        principalToken.transfer(address(lenderCommitmentGroupSmart), 1e18);
        collateralToken.transfer(address(lenderCommitmentGroupSmart), 1e18);

        lenderCommitmentGroupSmart.set_mockSharesExchangeRate(1e36 * 2);

        initialize_group_contract();

        //lenderCommitmentGroupSmart.set_totalPrincipalTokensCommitted(1000000);
        //lenderCommitmentGroupSmart.set_totalInterestCollected(2000000);

        uint256 minAmountOut = 500000;


        vm.prank(address(lender));
        principalToken.approve(address(lenderCommitmentGroupSmart), 1000000);

        vm.prank(address(lender));
        uint256 sharesAmount_ = lenderCommitmentGroupSmart
            .addPrincipalToCommitmentGroup(1000000, address(borrower),minAmountOut);

        uint256 expectedSharesAmount = 500000;

        //use ttoken logic to make this better
        assertEq(
            sharesAmount_,
            expectedSharesAmount,
            "Received an unexpected amount of shares"
        );
    }

    function test_addPrincipalToCommitmentGroup_after_nonzero_shares()
        public
    {
        principalToken.transfer(address(lenderCommitmentGroupSmart), 1e18);
        collateralToken.transfer(address(lenderCommitmentGroupSmart), 1e18);
 
       
        initialize_group_contract();
        lenderCommitmentGroupSmart.set_totalInterestCollected(1e6 * 1);
        lenderCommitmentGroupSmart.set_totalPrincipalTokensCommitted(1e6 * 1);

        uint256 minAmountOut = 500000;

        vm.prank(address(lender));
        principalToken.approve(address(lenderCommitmentGroupSmart), 1000000);

        vm.prank(address(lender));
        uint256 sharesAmount_ = lenderCommitmentGroupSmart
            .addPrincipalToCommitmentGroup(1000000, address(borrower), minAmountOut);

        uint256 expectedSharesAmount = 1000000;

        //use ttoken logic to make this better
        assertEq(
            sharesAmount_,
            expectedSharesAmount,
            "Received an unexpected amount of shares"
        );
    }

    function test_burnShares_simple() public {
        principalToken.transfer(address(lenderCommitmentGroupSmart), 1e18);
        // collateralToken.transfer(address(lenderCommitmentGroupSmart),1e18);

        initialize_group_contract();

          lenderCommitmentGroupSmart.set_mockSharesExchangeRate( 1e36 );  //this means 1:1 since it is expanded

       // lenderCommitmentGroupSmart.set_totalPrincipalTokensCommitted(1000000);
       // lenderCommitmentGroupSmart.set_totalInterestCollected(0);

      
        lenderCommitmentGroupSmart.set_totalPrincipalTokensCommitted(
            
            1000000
        );

        vm.prank(address(lender));
        principalToken.approve(address(lenderCommitmentGroupSmart), 1000000);

        vm.prank(address(lender));

        uint256 sharesAmount = 1000000;
        //should have all of the shares at this point
        lenderCommitmentGroupSmart.mock_mintShares(
            address(lender),
            sharesAmount
        );


      vm.prank(address(lender));

        lenderCommitmentGroupSmart.prepareSharesForBurn(sharesAmount);

        vm.warp(1000);

        vm.prank(address(lender));

        uint256 minAmountOut = 1000000;
        
         uint256 receivedPrincipalTokens 
          = lenderCommitmentGroupSmart.burnSharesToWithdrawEarnings(
                sharesAmount,
                address(lender),
                minAmountOut
            );

        uint256 expectedReceivedPrincipalTokens = 1000000; // the orig amt !
        assertEq(
            receivedPrincipalTokens,
            expectedReceivedPrincipalTokens,
            "Received an unexpected amount of principaltokens"
        );
    }


    function test_burnShares_simple_with_ratio_math() public {
        principalToken.transfer(address(lenderCommitmentGroupSmart), 1e18);
        // collateralToken.transfer(address(lenderCommitmentGroupSmart),1e18);

        initialize_group_contract();

          lenderCommitmentGroupSmart.set_mockSharesExchangeRate( 2e36 );  //this means 1:1 since it is expanded

       // lenderCommitmentGroupSmart.set_totalPrincipalTokensCommitted(1000000);
       // lenderCommitmentGroupSmart.set_totalInterestCollected(0);

        lenderCommitmentGroupSmart.set_totalPrincipalTokensCommitted(
            
            1000000
        );

        vm.prank(address(lender));
        principalToken.approve(address(lenderCommitmentGroupSmart), 1000000);

        vm.prank(address(lender));

        uint256 sharesAmount = 500000;
        //should have all of the shares at this point
        lenderCommitmentGroupSmart.mock_mintShares(
            address(lender),
            sharesAmount
        );  

         vm.prank(address(lender));

        lenderCommitmentGroupSmart.prepareSharesForBurn(sharesAmount);

        vm.warp(1000);

        uint256 minAmountOut = 900000;
        vm.prank(address(lender));
        
         uint256 receivedPrincipalTokens 
          = lenderCommitmentGroupSmart.burnSharesToWithdrawEarnings(
                sharesAmount,
                address(lender),
                minAmountOut
            );

        uint256 expectedReceivedPrincipalTokens = 1000000; // the orig amt !
        assertEq(
            receivedPrincipalTokens,
            expectedReceivedPrincipalTokens,
            "Received an unexpected amount of principaltokens"
        );
    }

    function test_burnShares_also_get_collateral() public {
        principalToken.transfer(address(lenderCommitmentGroupSmart), 1e18);
        collateralToken.transfer(address(lenderCommitmentGroupSmart), 1e18);

        initialize_group_contract();

        lenderCommitmentGroupSmart.set_mockSharesExchangeRate( 1e36 );  //the default for now 
 

        lenderCommitmentGroupSmart.set_totalPrincipalTokensCommitted(
            
            1000000
        );

        vm.prank(address(lender));
        principalToken.approve(address(lenderCommitmentGroupSmart), 1000000);

        vm.prank(address(lender));

        uint256 minAmountOut = 500000;

        uint256 sharesAmount = 500000;
        //should have all of the shares at this point
        lenderCommitmentGroupSmart.mock_mintShares(
            address(lender),
            sharesAmount
        );



        vm.prank(address(lender));

        lenderCommitmentGroupSmart.prepareSharesForBurn(sharesAmount);

        vm.warp(1000);

        vm.prank(address(lender));
    
            uint256 receivedPrincipalTokens
         = lenderCommitmentGroupSmart.burnSharesToWithdrawEarnings(
                sharesAmount,
                address(lender),
                minAmountOut
            );

        uint256 expectedReceivedPrincipalTokens = 500000; // the orig amt !
        assertEq(
            receivedPrincipalTokens,
            expectedReceivedPrincipalTokens,
            "Received an unexpected amount of principal tokens"
        );
 
    }

     //test this thoroughly -- using spreadsheet data 
    function test_get_shares_exchange_rate_scenario_A() public {
          initialize_group_contract();

        lenderCommitmentGroupSmart.set_totalInterestCollected(0);

        lenderCommitmentGroupSmart.set_totalPrincipalTokensCommitted(
             
            5000000
        );

        uint256 rate = lenderCommitmentGroupSmart.super_sharesExchangeRate();

         assertEq(rate , 1e36, "unexpected sharesExchangeRate");

    }

    function test_get_shares_exchange_rate_scenario_B() public {
          initialize_group_contract();

        lenderCommitmentGroupSmart.set_totalInterestCollected(1000000);

        lenderCommitmentGroupSmart.set_totalPrincipalTokensCommitted(
 
            1000000
        );

         lenderCommitmentGroupSmart.set_totalPrincipalTokensWithdrawn(
            
            1000000
        );

        uint256 rate = lenderCommitmentGroupSmart.super_sharesExchangeRate();

         assertEq(rate , 1e36, "unexpected sharesExchangeRate");

    }

    function test_get_shares_exchange_rate_scenario_C() public {
        initialize_group_contract();


        lenderCommitmentGroupSmart.set_totalPrincipalTokensCommitted(
             
            1000000
        );

        lenderCommitmentGroupSmart.set_totalInterestCollected(1000000);

        uint256 sharesAmount = 500000;


        lenderCommitmentGroupSmart.mock_mintShares(
            address(lender),
            sharesAmount
        );

        uint256 poolTotalEstimatedValue = lenderCommitmentGroupSmart.getPoolTotalEstimatedValue();
        assertEq(poolTotalEstimatedValue ,  2 * 1000000, "unexpected poolTotalEstimatedValue");

        uint256 rate = lenderCommitmentGroupSmart.super_sharesExchangeRate();

        assertEq(rate , 4 * 1e36, "unexpected sharesExchangeRate");

        /*
            Rate should be 2 at this point so a depositor will only get half as many shares for their principal 
        */
    }


 
 
    function test_get_shares_exchange_rate_after_default_liquidation_A() public {
         initialize_group_contract();


        lenderCommitmentGroupSmart.set_totalPrincipalTokensCommitted(
            1000000
        );

        lenderCommitmentGroupSmart.set_totalInterestCollected(1000000);

        lenderCommitmentGroupSmart.set_tokenDifferenceFromLiquidations(-1000000);

        uint256 sharesAmount = 1000000;

        lenderCommitmentGroupSmart.mock_mintShares(
            address(lender),
            sharesAmount
        );

        uint256 poolTotalEstimatedValue = lenderCommitmentGroupSmart.getPoolTotalEstimatedValue();
        assertEq(poolTotalEstimatedValue ,  1 * 1000000, "unexpected poolTotalEstimatedValue");

        uint256 rate = lenderCommitmentGroupSmart.super_sharesExchangeRate();

        assertEq(rate , 1 * 1e36, "unexpected sharesExchangeRate");

    }


 
    function test_get_shares_exchange_rate_after_default_liquidation_B() public {
         initialize_group_contract();


        lenderCommitmentGroupSmart.set_totalPrincipalTokensCommitted(
            1000000
        );

    
        lenderCommitmentGroupSmart.set_tokenDifferenceFromLiquidations(-500000);

        uint256 sharesAmount = 1000000;

        lenderCommitmentGroupSmart.mock_mintShares(
            address(lender),
            sharesAmount
        );

        uint256 poolTotalEstimatedValue = lenderCommitmentGroupSmart.getPoolTotalEstimatedValue();
        assertEq(poolTotalEstimatedValue ,  1 * 500000, "unexpected poolTotalEstimatedValue");

        uint256 rate = lenderCommitmentGroupSmart.super_sharesExchangeRate();

        assertEq(rate ,  1e36 / 2, "unexpected sharesExchangeRate");

    }





    function test_get_shares_exchange_rate_inverse() public {
        lenderCommitmentGroupSmart.set_mockSharesExchangeRate(1000000);
 

        uint256 rate = lenderCommitmentGroupSmart.super_sharesExchangeRateInverse();
    }


   









/*
  make sure we get expected data based on the vm warp 
*/
    function test_getMinimumAmountDifferenceToCloseDefaultedLoan() public {
       initialize_group_contract();

        uint256 bidId = 0;
        uint256 amountDue = 500;

       _tellerV2.mock_setLoanDefaultTimestamp(block.timestamp);
   
        vm.warp(10000);
        uint256 loanDefaultTimestamp = block.timestamp - 2000; //sim that loan defaulted 2000 seconds ago 

        int256 min_amount = lenderCommitmentGroupSmart.super_getMinimumAmountDifferenceToCloseDefaultedLoan(
             
            amountDue,
            loanDefaultTimestamp
        );

      int256 expectedMinAmount = 3720; //based on loanDefaultTimestamp gap 

       assertEq(min_amount,expectedMinAmount,"min_amount unexpected");

    }


      function test_getMinimumAmountDifferenceToCloseDefaultedLoan_zero_time() public {
       initialize_group_contract();

        uint256 bidId = 0;
        uint256 amountDue = 500;

       _tellerV2.mock_setLoanDefaultTimestamp(block.timestamp);
   
        vm.warp(10000);
        uint256 loanDefaultTimestamp = block.timestamp ; //sim that loan defaulted 2000 seconds ago 


        vm.expectRevert("Loan defaulted timestamp must be in the past");
        int256 min_amount = lenderCommitmentGroupSmart.super_getMinimumAmountDifferenceToCloseDefaultedLoan(
           
            amountDue,
            loanDefaultTimestamp
        );

      
      
    }


      function test_getMinimumAmountDifferenceToCloseDefaultedLoan_full_time() public {
       initialize_group_contract();

        uint256 bidId = 0;
        uint256 amountDue = 500;

       _tellerV2.mock_setLoanDefaultTimestamp(block.timestamp);
   
        vm.warp(100000);
        uint256 loanDefaultTimestamp = block.timestamp - 22000 ; //sim that loan defaulted 2000 seconds ago 

        int256 min_amount = lenderCommitmentGroupSmart.super_getMinimumAmountDifferenceToCloseDefaultedLoan(
           
            amountDue,
            loanDefaultTimestamp
        );

      int256 expectedMinAmount = 2720; //based on loanDefaultTimestamp gap 

       assertEq(min_amount,expectedMinAmount,"min_amount unexpected");

    }



    function test_getMinInterestRate() public {
        lenderCommitmentGroupSmart.set_mock_getMaxPrincipalPerCollateralAmount(
            100 * 1e18
        );

        principalToken.transfer(address(lenderCommitmentGroupSmart), 1e18);
        collateralToken.transfer(address(lenderCommitmentGroupSmart), 1e18);

        initialize_group_contract();

        lenderCommitmentGroupSmart.set_totalPrincipalTokensCommitted(1000000);

        uint256 principalAmount = 50;
        uint256 collateralAmount = 50 * 100;

        address collateralTokenAddress = address(
            lenderCommitmentGroupSmart.collateralToken()
        );
        uint256 collateralTokenId = 0;

        uint32 loanDuration = 5000000;
        uint16 interestRate = 800;

        uint256 bidId = 0;

 
      uint16 poolUtilizationRatio = lenderCommitmentGroupSmart.getPoolUtilizationRatio( 
            0
         );


        assertEq(poolUtilizationRatio, 0);

        // submit bid 
        uint16 minInterestRate = lenderCommitmentGroupSmart.getMinInterestRate( 
           0
         );



        assertEq(minInterestRate, 0);
    }




    function test_acceptFundsForAcceptBid() public {


        //this mock no longer helps ! 
       /* lenderCommitmentGroupSmart.set_mock_getMaxPrincipalPerCollateralAmount(
            100 * 1e18
        );*/

        lenderCommitmentGroupSmart.set_mock_requiredCollateralAmount(
            100  
        );

        

        principalToken.transfer(address(lenderCommitmentGroupSmart), 1e18);
        collateralToken.transfer(address(lenderCommitmentGroupSmart), 1e18);

        initialize_group_contract();

        lenderCommitmentGroupSmart.set_totalPrincipalTokensCommitted(1000000);

        uint256 principalAmount = 50;
        uint256 collateralAmount =   100;

        address collateralTokenAddress = address(
            lenderCommitmentGroupSmart.collateralToken()
        );
        uint256 collateralTokenId = 0;

        uint32 loanDuration = 5000000;
        uint16 interestRate = 100;

        uint256 bidId = 0;


       // TellerV2SolMock(_tellerV2).setMarketRegistry(address(marketRegistry));



        // submit bid 
        TellerV2SolMock(_tellerV2).submitBid( 
            address(principalToken),
            0,
            principalAmount,
            loanDuration,
            interestRate,
            "",
            address(this)
         );



        vm.prank(address(_smartCommitmentForwarder));
        lenderCommitmentGroupSmart.acceptFundsForAcceptBid(
            address(borrower),
            bidId,
            principalAmount,
            collateralAmount,
            collateralTokenAddress,
            collateralTokenId,
            loanDuration,
            interestRate
        );
    }

    function test_acceptFundsForAcceptBid_insufficientCollateral() public {
        /*lenderCommitmentGroupSmart.set_mock_getMaxPrincipalPerCollateralAmount(
            100 * 1e18
        );*/

          lenderCommitmentGroupSmart.set_mock_requiredCollateralAmount(
            100  
        );

        principalToken.transfer(address(lenderCommitmentGroupSmart), 1e18);
        collateralToken.transfer(address(lenderCommitmentGroupSmart), 1e18);




        initialize_group_contract();

        lenderCommitmentGroupSmart.set_totalPrincipalTokensCommitted(1000000);

        uint256 principalAmount = 100;
        uint256 collateralAmount = 0;

        address collateralTokenAddress = address(
            lenderCommitmentGroupSmart.collateralToken()
        );
        uint256 collateralTokenId = 0;

        uint32 loanDuration = 5000000;
        uint16 interestRate = 100;

        uint256 bidId = 0;




        vm.expectRevert("Insufficient Borrower Collateral");
        vm.prank(address(_smartCommitmentForwarder));
        lenderCommitmentGroupSmart.acceptFundsForAcceptBid(
            address(borrower),
            bidId,
            principalAmount,
            collateralAmount,
            collateralTokenAddress,
            collateralTokenId,
            loanDuration,
            interestRate
        );
    }


     function test_repayLoanCallback() public {
        uint256 principalAmount = 100;
        uint256 interestAmount = 50;
        address repayer = address(borrower);

        uint256 bidId = 0;

        lenderCommitmentGroupSmart.mock_setBidActive(bidId);
        
        vm.prank(address(_tellerV2));
        lenderCommitmentGroupSmart.repayLoanCallback(
            bidId,
            address(repayer),
            principalAmount,
            interestAmount
        );

        

     }

      function test_repayLoanCallback_bid_not_active() public {
        uint256 principalAmount = 100;
        uint256 interestAmount = 50;
        address repayer = address(borrower);

        uint256 bidId = 0;

        vm.expectRevert("Bid is not active for group");
        vm.prank(address(_tellerV2));
        lenderCommitmentGroupSmart.repayLoanCallback(
            bidId,
            address(repayer),
            principalAmount,
            interestAmount
        );

       

     }

    /*
      improve tests for this 

      
    */


/*
    function test_getCollateralTokensAmountEquivalentToPrincipalTokens_scenarioA() public {
         
        initialize_group_contract();

        uint256 principalTokenAmountValue = 9000;
        uint256 pairPriceWithTwap = 1 * 2**96;
        uint256 pairPriceImmediate = 2 * 2**96;
        bool principalTokenIsToken0 = false;
 
        
        uint256 amountCollateral = lenderCommitmentGroupSmart
            .super_getCollateralTokensAmountEquivalentToPrincipalTokens(
                principalTokenAmountValue,
                pairPriceWithTwap,
                pairPriceImmediate,
                principalTokenIsToken0                
                );

       
        uint256 expectedAmount = 9000;  

        assertEq(
            amountCollateral,
            expectedAmount,
            "Unexpected getCollateralTokensPricePerPrincipalTokens"
        );
    }

    function test_getCollateralTokensAmountEquivalentToPrincipalTokens_scenarioB() public {
         
        initialize_group_contract();

        uint256 principalTokenAmountValue = 9000;
        uint256 pairPriceWithTwap = 1 * 2**96;
        uint256 pairPriceImmediate = 2 * 2**96;
        bool principalTokenIsToken0 = true;
 
        
        uint256 amountCollateral = lenderCommitmentGroupSmart
            .super_getCollateralTokensAmountEquivalentToPrincipalTokens(
                principalTokenAmountValue,
                pairPriceWithTwap,
                pairPriceImmediate,
                principalTokenIsToken0                
                );

       
        uint256 expectedAmount = 18000;  

        assertEq(
            amountCollateral,
            expectedAmount,
            "Unexpected getCollateralTokensPricePerPrincipalTokens"
        );
    }

     function test_getCollateralTokensAmountEquivalentToPrincipalTokens_scenarioC() public {
         
        initialize_group_contract();

        uint256 principalTokenAmountValue = 9000;
        uint256 pairPriceWithTwap = 1 * 2**96;
        uint256 pairPriceImmediate = 60000 * 2**96;
        bool principalTokenIsToken0 = false;
 
        
        uint256 amountCollateral = lenderCommitmentGroupSmart
            .super_getCollateralTokensAmountEquivalentToPrincipalTokens(
                principalTokenAmountValue,
                pairPriceWithTwap,
                pairPriceImmediate,
                principalTokenIsToken0                
                );

       
        uint256 expectedAmount = 9000;  

        assertEq(
            amountCollateral,
            expectedAmount,
            "Unexpected getCollateralTokensPricePerPrincipalTokens"
        );
    }


   function test_getCollateralTokensAmountEquivalentToPrincipalTokens_scenarioD() public {
         
        initialize_group_contract();

        uint256 principalTokenAmountValue = 9000;
        uint256 pairPriceWithTwap = 60000 * 2**96;
        uint256 pairPriceImmediate =  2**96;
        bool principalTokenIsToken0 = false;
 
        
        uint256 amountCollateral = lenderCommitmentGroupSmart
            .super_getCollateralTokensAmountEquivalentToPrincipalTokens(
                principalTokenAmountValue,
                pairPriceWithTwap,
                pairPriceImmediate,
                principalTokenIsToken0                
                );

       
        uint256 expectedAmount = 9000;  

        assertEq(
            amountCollateral,
            expectedAmount,
            "Unexpected getCollateralTokensPricePerPrincipalTokens"
        );
    }
*/




     /*
     

      test for _getUniswapV3TokenPairPrice
    */
    /*
 function test_getPriceFromSqrtX96_scenarioA() public {
         
        initialize_group_contract(); 
 
        uint160 sqrtPriceX96 = 771166083179357884152611;

        uint256 priceX96 = lenderCommitmentGroupSmart
            .super_getPriceFromSqrtX96(
                 sqrtPriceX96               
                );

        uint256 price = priceX96 * 1e18 / 2**96;

        uint256 expectedAmountX96 = 7506133033681329001;
        uint256 expectedAmount = 94740718; 

          assertEq(
            priceX96,
            expectedAmountX96,
            "Unexpected getPriceFromSqrtX96"
        );
        
        assertEq(
            price,
            expectedAmount,
            "Unexpected getPriceFromSqrtX96"
        );
    }

    */

}

contract User {}

/*
contract SmartCommitmentForwarder {

    function paused() external returns (bool){
        return false;
    }

}*/