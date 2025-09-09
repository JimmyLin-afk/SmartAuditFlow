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
 





/*
    make sure both pos and neg branches get run, and tellerV2 is called at the end 
*/
    function test_liquidateDefaultedLoanWithIncentive() public {
          initialize_group_contract();

        principalToken.transfer(address(liquidator), 1e18);
        uint256 originalBalance = principalToken.balanceOf(address(liquidator));

        uint256 amountOwed = 100;
   
        
        uint256 bidId = 0;
    

       lenderCommitmentGroupSmart.set_mockAmountOwedForBid(amountOwed); 

   

         vm.warp(1000);   //loanDefaultedTimeStamp ?

       lenderCommitmentGroupSmart.set_mockBidAsActiveForGroup(bidId,true); 

       //set itself as mock owner for now ..  used for protocol fee 
       _tellerV2.setMockOwner( address(lenderCommitmentGroupSmart)  );

       _tellerV2.setMockProtocolFeeRecipient( address(lenderCommitmentGroupSmart)  );

       vm.prank(address(liquidator));
       principalToken.approve(address(lenderCommitmentGroupSmart), 1e18);

       int256 minAmountDifference = 2000;

       lenderCommitmentGroupSmart.mock_setMinimumAmountDifferenceToCloseDefaultedLoan(minAmountDifference);

        int256 tokenAmountDifference = 4000;
        vm.prank(address(liquidator));
        lenderCommitmentGroupSmart.liquidateDefaultedLoanWithIncentive(
           bidId, 
           tokenAmountDifference           
        );

        uint256 updatedBalance = principalToken.balanceOf(address(liquidator));

        int256 expectedDifference = int256(amountOwed) + minAmountDifference;

        assertEq(originalBalance - updatedBalance , uint256(expectedDifference), "unexpected tokenDifferenceFromLiquidations");


      //make sure lenderCloseloan is called 
       assertEq( _tellerV2.lenderCloseLoanWasCalled(), true, "lender close loan not called");
    }


    //complete me 
     function test_liquidateDefaultedLoanWithIncentive_negative_direction() public {


        initialize_group_contract();

        principalToken.transfer(address(liquidator), 1e18);
        uint256 originalBalance = principalToken.balanceOf(address(liquidator));

        uint256 amountOwed = 1000;
   
        
        uint256 bidId = 0;
    
     _tellerV2.setMockProtocolFeeRecipient( address(lenderCommitmentGroupSmart)  );

       lenderCommitmentGroupSmart.set_mockAmountOwedForBid(amountOwed); 

   
        //time has advanced enough to now have a 50 percent discount s
         vm.warp(1000);   //loanDefaultedTimeStamp ?

       lenderCommitmentGroupSmart.set_mockBidAsActiveForGroup(bidId,true); 
      
       vm.prank(address(liquidator));
       principalToken.approve(address(lenderCommitmentGroupSmart), 1e18);

       lenderCommitmentGroupSmart.mock_setMinimumAmountDifferenceToCloseDefaultedLoan(-500);

        int256 tokenAmountDifference = -500;
        vm.prank(address(liquidator));
        lenderCommitmentGroupSmart.liquidateDefaultedLoanWithIncentive(
           bidId, 
           tokenAmountDifference           
        );

        uint256 updatedBalance = principalToken.balanceOf(address(liquidator));

        require(tokenAmountDifference < 0); //ensure this test is set up properly 

        // we expect it to be amountOwned - abs(tokenAmountDifference ) but we can just test it like this 
        int256 expectedDifference = int256(amountOwed) + ( tokenAmountDifference);

        assertEq(originalBalance - updatedBalance , uint256(expectedDifference), "unexpected tokenDifferenceFromLiquidations");


      //make sure lenderCloseloan is called 
       assertEq( _tellerV2.lenderCloseLoanWasCalled(), true, "lender close loan not called");
     }

  
function test_liquidateDefaultedLoanWithIncentive_increments_amount_repaid_A() public {


        initialize_group_contract();

        _tellerV2.setMockOwner( address(lenderCommitmentGroupSmart)  );


        principalToken.transfer(address(liquidator), 1e18);
        uint256 originalBalance = principalToken.balanceOf(address(liquidator));

        uint256 amountOwed = 1000;
   
        
        uint256 bidId = 0;


       uint256 originalTotalPrincipalTokensCommitted= 6000;
      lenderCommitmentGroupSmart.set_totalPrincipalTokensCommitted(originalTotalPrincipalTokensCommitted);


       lenderCommitmentGroupSmart.set_mockAmountOwedForBid(amountOwed); 
       _tellerV2.setMockProtocolFeeRecipient( address(lenderCommitmentGroupSmart)  );

   
        //time has advanced enough to now have a 50 percent discount s
         vm.warp(1000);   //loanDefaultedTimeStamp ?

       lenderCommitmentGroupSmart.set_mockBidAsActiveForGroup(bidId,true); 
      
       vm.prank(address(liquidator));
       principalToken.approve(address(lenderCommitmentGroupSmart), 1e18);

       uint256 poolTotalEstimatedValueBefore = lenderCommitmentGroupSmart.getPoolTotalEstimatedValue();

        uint256 originalTotalPrincipalTokensRepaid = 2000;
        lenderCommitmentGroupSmart.set_totalPrincipalTokensRepaid(originalTotalPrincipalTokensRepaid);

       lenderCommitmentGroupSmart.mock_setMinimumAmountDifferenceToCloseDefaultedLoan(-500);

        int256 tokenAmountDifference = -500;
        vm.prank(address(liquidator));
        lenderCommitmentGroupSmart.liquidateDefaultedLoanWithIncentive(
           bidId, 
           tokenAmountDifference           
        );

            // this actually doesnt happen in a group liquidation 
       /*  vm.prank(address(_tellerV2));
        lenderCommitmentGroupSmart.repayLoanCallback(
            bidId,
            address(this),
            amountOwed,
            20
        );
        */

        uint256 updatedBalance = principalToken.balanceOf(address(liquidator));

        uint256 totalPrincipalTokensRepaid = lenderCommitmentGroupSmart.totalPrincipalTokensRepaid();
         uint256 poolTotalEstimatedValueAfter = lenderCommitmentGroupSmart.getPoolTotalEstimatedValue();

 
        assertEq(totalPrincipalTokensRepaid, originalTotalPrincipalTokensRepaid + amountOwed, "unexpected totalPrincipalTokensRepaid");
        assertEq( poolTotalEstimatedValueBefore - poolTotalEstimatedValueAfter, 500, "unexpected poolTotalEstimatedValue");


        assertEq(  poolTotalEstimatedValueBefore, 6000, "unexpected poolTotalEstimatedValueBefore");
         assertEq( poolTotalEstimatedValueAfter , 5500, "unexpected poolTotalEstimatedValueAfter");


      //make sure lenderCloseloan is called 
      //how is this passing ? 
       assertEq( _tellerV2.lenderCloseLoanWasCalled(), true, "lender close loan not called");
     }

 

 function test_liquidateDefaultedLoanWithIncentive_increments_amount_repaid_B() public {


        initialize_group_contract();

        principalToken.transfer(address(liquidator), 1e18);
        uint256 originalBalance = principalToken.balanceOf(address(liquidator));

        uint256 amountOwed = 1000;
   
        
        uint256 bidId = 0;
    
       _tellerV2.setMockOwner( address(lenderCommitmentGroupSmart)  );
       _tellerV2.setMockProtocolFeeRecipient( address(lenderCommitmentGroupSmart)  );

       lenderCommitmentGroupSmart.set_mockAmountOwedForBid(amountOwed); 

   
        //time has advanced enough to now have a 50 percent discount s
         vm.warp(1000);   //loanDefaultedTimeStamp ?

       lenderCommitmentGroupSmart.set_mockBidAsActiveForGroup(bidId,true); 
      
       vm.prank(address(liquidator));
       principalToken.approve(address(lenderCommitmentGroupSmart), 1e18);

      uint256 originalTotalPrincipalTokensCommitted= 6000;
      lenderCommitmentGroupSmart.set_totalPrincipalTokensCommitted(originalTotalPrincipalTokensCommitted);



       uint256 poolTotalEstimatedValueBefore = lenderCommitmentGroupSmart.getPoolTotalEstimatedValue();

        uint256 originalTotalPrincipalTokensRepaid = 2000;
        lenderCommitmentGroupSmart.set_totalPrincipalTokensRepaid(originalTotalPrincipalTokensRepaid);

       lenderCommitmentGroupSmart.mock_setMinimumAmountDifferenceToCloseDefaultedLoan(500);

        int256 tokenAmountDifference = 500;
        vm.prank(address(liquidator));
        lenderCommitmentGroupSmart.liquidateDefaultedLoanWithIncentive(
           bidId, 
           tokenAmountDifference           
        );

          

        uint256 updatedBalance = principalToken.balanceOf(address(liquidator));

        uint256 totalPrincipalTokensRepaid = lenderCommitmentGroupSmart.totalPrincipalTokensRepaid();
        uint256 poolTotalEstimatedValueAfter = lenderCommitmentGroupSmart.getPoolTotalEstimatedValue();

 
        assertEq(totalPrincipalTokensRepaid, originalTotalPrincipalTokensRepaid + amountOwed, "unexpected totalPrincipalTokensRepaid");
       
       
       
         assertEq(  poolTotalEstimatedValueBefore, 6000, "unexpected poolTotalEstimatedValueBefore");
         assertEq( poolTotalEstimatedValueAfter , 6500, "unexpected poolTotalEstimatedValueAfter");

        // assertEq( poolTotalEstimatedValueAfter - poolTotalEstimatedValueBefore, 500, "unexpected poolTotalEstimatedValue");


      //make sure lenderCloseloan is called 
      //how is this passing ? 
       assertEq( _tellerV2.lenderCloseLoanWasCalled(), true, "lender close loan not called");
     }


}

contract User {}
 