// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

//import "forge-std/Test.sol";
import "../../contracts/libraries/UniswapPricingLibrary.sol";
 

import { Testable } from "../Testable.sol";

import { UniswapV3PoolMock } from "../../contracts/mock/uniswap/UniswapV3PoolMock.sol";

import "../../contracts/mock/MarketRegistryMock.sol";


import { IUniswapPricingLibrary } from "../../contracts/interfaces/IUniswapPricingLibrary.sol";

import "../../contracts/TellerV2Context.sol";

import "forge-std/console.sol";



contract UniswapPricingLibraryTest is Testable {

    LenderCommitmentForwarderTest_TellerV2Mock private tellerV2Mock;
    MarketRegistryMock mockMarketRegistry;

   // using UniswapPricingLibrary for IUniswapPricingLibrary.PoolRouteConfig[];

    UniswapV3PoolMock mockUniswapPool;
    UniswapV3PoolMock mockUniswapPoolSecondary;

    function setUp() public {


        tellerV2Mock = new LenderCommitmentForwarderTest_TellerV2Mock();
        mockMarketRegistry = new MarketRegistryMock();

        mockUniswapPool = new UniswapV3PoolMock();

        mockUniswapPoolSecondary = new UniswapV3PoolMock();

    }

   function test_getUniswapPriceRatioForPool_same_price() public {
      
        bool zeroForOne = false; // ??

        mockUniswapPool.set_mockSqrtPriceX96(1 * 2**96);

        uint32 twapInterval = 0;

        IUniswapPricingLibrary.PoolRouteConfig
            memory routeConfig = IUniswapPricingLibrary.PoolRouteConfig({
                pool: address(mockUniswapPool),
                zeroForOne: zeroForOne,
                twapInterval: twapInterval,
                token0Decimals: 18,
                token1Decimals: 18
            });

        uint256 priceRatio = UniswapPricingLibrary
            .getUniswapPriceRatioForPool(routeConfig);

        

        assertEq(  1e18 , priceRatio  , "unexpected price ratio") ;
    }


      function test_getUniswapPriceRatioForPool_different_price() public {
      
        bool zeroForOne = false; // ??

        
        //i think this means the ratio is 100:1
        mockUniswapPool.set_mockSqrtPriceX96(10 * 2**96);


        uint32 twapInterval = 0;

        IUniswapPricingLibrary.PoolRouteConfig
            memory routeConfig = IUniswapPricingLibrary.PoolRouteConfig({
                pool: address(mockUniswapPool),
                zeroForOne: zeroForOne,
                twapInterval: twapInterval,
                token0Decimals: 18,
                token1Decimals: 18
            });

        uint256 priceRatio = UniswapPricingLibrary
            .getUniswapPriceRatioForPool(routeConfig);

        

        assertEq(  1e16 , priceRatio  , "unexpected price ratio") ;
    }



  function test_getUniswapPriceRatioForPool_decimal_scenario_A() public {
      
        bool zeroForOne = false; // ??

        uint256  principalTokenDecimals = 18;
        uint256 collateralTokenDecimals = 6;

          

        mockUniswapPool.set_mockSqrtPriceX96(1 * 2**96);

        uint32 twapInterval = 0;

    //decimals dont affect raw ratios 
        IUniswapPricingLibrary.PoolRouteConfig
            memory routeConfig = IUniswapPricingLibrary.PoolRouteConfig({
                pool: address(mockUniswapPool),
                zeroForOne: zeroForOne,
                twapInterval: twapInterval,
                token0Decimals: collateralTokenDecimals,
                token1Decimals: principalTokenDecimals
            });

        uint256 priceRatio = UniswapPricingLibrary
            .getUniswapPriceRatioForPool(routeConfig);

        

        assertEq(  1e18 , priceRatio  , "unexpected price ratio") ;
    }


   function test_getUniswapPriceRatioForPoolRoutes_decimal_scenario_B()
        public
    {
        mockUniswapPool.set_mockSqrtPriceX96(1 * 2**96);

        mockUniswapPoolSecondary.set_mockSqrtPriceX96(1 * 2**96);

        uint32 twapInterval = 0; //for now

        bool zeroForOne = true;

        uint256 principalTokenDecimals = 18;
        uint256 intermediateTokenDecimals = 18;
        uint256 collateralTokenDecimals = 6;

        

        IUniswapPricingLibrary.PoolRouteConfig[]
            memory poolRoutes = new IUniswapPricingLibrary.PoolRouteConfig[](
                2
            );

        poolRoutes[0] = IUniswapPricingLibrary.PoolRouteConfig({
            pool: address(mockUniswapPool),
            zeroForOne: zeroForOne,
            twapInterval: twapInterval,
            token0Decimals: collateralTokenDecimals,
            token1Decimals: intermediateTokenDecimals
        });

        poolRoutes[1] = IUniswapPricingLibrary.PoolRouteConfig({
            pool: address(mockUniswapPoolSecondary),
            zeroForOne: zeroForOne,
            twapInterval: twapInterval,
            token0Decimals: intermediateTokenDecimals,
            token1Decimals: principalTokenDecimals
        });

        uint256 priceRatio = UniswapPricingLibrary
            .getUniswapPriceRatioForPoolRoutes(poolRoutes);

        console.log("price ratio");
        console.logUint(priceRatio);
 
         assertEq(  1e18 , priceRatio  , "unexpected price ratio") ;
    }

    function test_getUniswapPriceRatioForPoolRoutes_decimal_scenario_C()
        public
    {
        mockUniswapPool.set_mockSqrtPriceX96(1 * 2**96);

        mockUniswapPoolSecondary.set_mockSqrtPriceX96(1 * 2**96);

        uint32 twapInterval = 0; //for now

        bool zeroForOne = true;

        uint256 principalTokenDecimals = 6;
        uint256 intermediateTokenDecimals = 18;
        uint256 collateralTokenDecimals = 18;

       
        IUniswapPricingLibrary.PoolRouteConfig[]
            memory poolRoutes = new IUniswapPricingLibrary.PoolRouteConfig[](
                2
            );

        poolRoutes[0] = IUniswapPricingLibrary.PoolRouteConfig({
            pool: address(mockUniswapPool),
            zeroForOne: zeroForOne,
            twapInterval: twapInterval,
            token0Decimals: collateralTokenDecimals,
            token1Decimals: intermediateTokenDecimals
        });

        poolRoutes[1] = IUniswapPricingLibrary.PoolRouteConfig({
            pool: address(mockUniswapPoolSecondary),
            zeroForOne: zeroForOne,
            twapInterval: twapInterval,
            token0Decimals: intermediateTokenDecimals,
            token1Decimals: principalTokenDecimals
        });

        uint256 priceRatio = UniswapPricingLibrary
            .getUniswapPriceRatioForPoolRoutes(poolRoutes);

        
        console.log("price ratio");
        console.logUint(priceRatio);
    

         assertEq(  1e18 , priceRatio  , "unexpected price ratio") ;
    }


     function test_getUniswapPriceRatioForPoolRoutes_price_scenario_A() public {
        mockUniswapPool.set_mockSqrtPriceX96(10 * 2**96);

        uint160 priceTwo = uint160(1 * 2**96) ;
        mockUniswapPoolSecondary.set_mockSqrtPriceX96(priceTwo);

        uint32 twapInterval = 0; //for now

        bool zeroForOne = false;

        IUniswapPricingLibrary.PoolRouteConfig[]
            memory poolRoutes = new IUniswapPricingLibrary.PoolRouteConfig[](
                2
            );

        poolRoutes[0] = IUniswapPricingLibrary.PoolRouteConfig({
            pool: address(mockUniswapPool),
            zeroForOne: zeroForOne,
            twapInterval: twapInterval,
            token0Decimals: 18,
            token1Decimals: 18
        });

        poolRoutes[1] = IUniswapPricingLibrary.PoolRouteConfig({
            pool: address(mockUniswapPoolSecondary),
            zeroForOne: zeroForOne,
            twapInterval: twapInterval,
            token0Decimals: 18,
            token1Decimals: 18
        });

        uint256 priceRatio = UniswapPricingLibrary
            .getUniswapPriceRatioForPoolRoutes(poolRoutes);

        console.log("price ratio");
        console.logUint(priceRatio);
 
      
         assertEq(  1e16 , priceRatio  , "unexpected price ratio") ;
    }


     function test_getUniswapPriceRatioForPoolRoutes_price_scenario_B() public {
        mockUniswapPool.set_mockSqrtPriceX96(1 * 2**96);

        uint32 twapInterval = 0; //for now

        bool zeroForOne = false;

        IUniswapPricingLibrary.PoolRouteConfig[]
            memory poolRoutes = new IUniswapPricingLibrary.PoolRouteConfig[](
                1
            );

        poolRoutes[0] = IUniswapPricingLibrary.PoolRouteConfig({
            pool: address(mockUniswapPool),
            zeroForOne: zeroForOne,
            twapInterval: twapInterval,
            token0Decimals: 18,
            token1Decimals: 18
        });

        uint256 priceRatio = UniswapPricingLibrary
            .getUniswapPriceRatioForPoolRoutes(poolRoutes);

        console.log("price ratio");
        console.logUint(priceRatio);

          
          assertEq(  1e18 , priceRatio  , "unexpected price ratio") ;
    }


 function test_getUniswapPriceRatioForPoolRoutes_price_scenario_C() public {
        mockUniswapPool.set_mockSqrtPriceX96(1 * 2**96);

        uint32 twapInterval = 0; //for now

        bool zeroForOne = true;

        IUniswapPricingLibrary.PoolRouteConfig[]
            memory poolRoutes = new IUniswapPricingLibrary.PoolRouteConfig[](
                1
            );

        poolRoutes[0] = IUniswapPricingLibrary.PoolRouteConfig({
            pool: address(mockUniswapPool),
            zeroForOne: zeroForOne,
            twapInterval: twapInterval,
            token0Decimals: 18,
            token1Decimals: 18
        });

        uint256 priceRatio = UniswapPricingLibrary
            .getUniswapPriceRatioForPoolRoutes(poolRoutes);

        console.log("price ratio");
        console.logUint(priceRatio);

       
       assertEq(  1e18 , priceRatio  , "unexpected price ratio") ;
        
    }

    function test_getUniswapPriceRatioForPoolRoutes_price_scenario_D() public {
        mockUniswapPool.set_mockSqrtPriceX96(81128457937705300000000);

        uint32 twapInterval = 0; //for now

        bool zeroForOne = false;

        
        IUniswapPricingLibrary.PoolRouteConfig[]
            memory poolRoutes = new IUniswapPricingLibrary.PoolRouteConfig[](
                1
            );

        poolRoutes[0] = IUniswapPricingLibrary.PoolRouteConfig({
            pool: address(mockUniswapPool),
            zeroForOne: zeroForOne,
            twapInterval: twapInterval,
            token0Decimals: 6,
            token1Decimals: 18
        });

        uint256 priceRatio = UniswapPricingLibrary
            .getUniswapPriceRatioForPoolRoutes(poolRoutes);

        console.log("price ratio");
        console.logUint(priceRatio);

        uint256 principalAmount = 1000;

     
        assertEq(  953702069891996282433059426929 , priceRatio  , "unexpected price ratio") ;
       
    }





function test_getUniswapPriceRatioForPool_zeroPrice() public {
    mockUniswapPool.set_mockSqrtPriceX96(0);

    uint32 twapInterval = 0;
    bool zeroForOne = false;

    IUniswapPricingLibrary.PoolRouteConfig memory routeConfig = IUniswapPricingLibrary.PoolRouteConfig({
        pool: address(mockUniswapPool),
        zeroForOne: zeroForOne,
        twapInterval: twapInterval,
        token0Decimals: 18,
        token1Decimals: 18
    });

    vm.expectRevert();
    uint256 priceRatio = UniswapPricingLibrary.getUniswapPriceRatioForPool(routeConfig)  ;
}

function test_getUniswapPriceRatioForPool_largePrice() public {
    // Setting a large sqrtPriceX96 to test upper bounds.
    mockUniswapPool.set_mockSqrtPriceX96(type(uint160).max);

    uint32 twapInterval = 0;
    bool zeroForOne = false;

    IUniswapPricingLibrary.PoolRouteConfig memory routeConfig = IUniswapPricingLibrary.PoolRouteConfig({
        pool: address(mockUniswapPool),
        zeroForOne: zeroForOne,
        twapInterval: twapInterval,
        token0Decimals: 18,
        token1Decimals: 18
    });

    uint256 priceRatio = UniswapPricingLibrary.getUniswapPriceRatioForPool(routeConfig);
 

       assertEq( priceRatio,  0 ,   "unexpected price ratio") ;
       
}

function test_getUniswapPriceRatioForPool_invalidDecimals() public {
    // Simulate mismatched decimals for edge case testing.
    mockUniswapPool.set_mockSqrtPriceX96(1 * 2**96);

    uint32 twapInterval = 0;
    bool zeroForOne = false;

    IUniswapPricingLibrary.PoolRouteConfig memory routeConfig = IUniswapPricingLibrary.PoolRouteConfig({
        pool: address(mockUniswapPool),
        zeroForOne: zeroForOne,
        twapInterval: twapInterval,
        token0Decimals: 0, // Invalid decimal value
        token1Decimals: 18
    });

    uint256 priceRatio = UniswapPricingLibrary.getUniswapPriceRatioForPool(routeConfig) ;

     

       assertEq( priceRatio,  1000000000000000000 ,   "unexpected price ratio") ;
       
    
}

function test_getUniswapPriceRatioForPool_twapInterval() public {
    // Test with a valid twapInterval to ensure TWAP computation is working.
    mockUniswapPool.set_mockSqrtPriceX96(2 * 2**96);

    uint32 twapInterval = 60; // 60 seconds TWAP
    bool zeroForOne = false;

    IUniswapPricingLibrary.PoolRouteConfig memory routeConfig = IUniswapPricingLibrary.PoolRouteConfig({
        pool: address(mockUniswapPool),
        zeroForOne: zeroForOne,
        twapInterval: twapInterval,
        token0Decimals: 18,
        token1Decimals: 18
    });

    uint256 priceRatio = UniswapPricingLibrary.getUniswapPriceRatioForPool(routeConfig);

    assert(priceRatio > 0);
}

function test_getUniswapPriceRatioForPoolRoutes_twoPools_differentPrices() public {
    mockUniswapPool.set_mockSqrtPriceX96(4 * 2**96); // Pool 1: sqrtPrice = 4
    mockUniswapPoolSecondary.set_mockSqrtPriceX96(2 * 2**96); // Pool 2: sqrtPrice = 2

    uint32 twapInterval = 0;
    bool zeroForOne = false;
 

    IUniswapPricingLibrary.PoolRouteConfig[]
            memory poolRoutes = new IUniswapPricingLibrary.PoolRouteConfig[](
                2
            );

    poolRoutes[0] = IUniswapPricingLibrary.PoolRouteConfig({
        pool: address(mockUniswapPool),
        zeroForOne: zeroForOne,
        twapInterval: twapInterval,
        token0Decimals: 18,
        token1Decimals: 18
    });

    poolRoutes[1] = IUniswapPricingLibrary.PoolRouteConfig({
        pool: address(mockUniswapPoolSecondary),
        zeroForOne: zeroForOne,
        twapInterval: twapInterval,
        token0Decimals: 18,
        token1Decimals: 18
    });

    uint256 priceRatio = UniswapPricingLibrary.getUniswapPriceRatioForPoolRoutes(poolRoutes);

    //calculatet this more intelligently 
    uint256 expectedPriceRatio = 15625000000000000;

    assertEq(priceRatio, expectedPriceRatio, "Unexpected price ratio for two pools with different prices");
}
 


    
}




contract LenderCommitmentForwarderTest_TellerV2Mock is TellerV2Context {
    constructor() TellerV2Context(address(0)) {}

    function __setMarketRegistry(address _marketRegistry) external {
        marketRegistry = IMarketRegistry(_marketRegistry);
    }

    function getSenderForMarket(uint256 _marketId)
        external
        view
        returns (address)
    {
        return _msgSenderForMarket(_marketId);
    }

    function getDataForMarket(uint256 _marketId)
        external
        view
        returns (bytes calldata)
    {
        return _msgDataForMarket(_marketId);
    }
}
