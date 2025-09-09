// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/utils/Address.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/proxy/beacon/UpgradeableBeacon.sol";


import "../../contracts/TellerV2Context.sol";


import "../../contracts/LenderCommitmentForwarder/extensions/LenderCommitmentGroup/LenderCommitmentGroup_Smart.sol";
import "../../contracts/LenderCommitmentForwarder/extensions/LenderCommitmentGroup/LenderCommitmentGroup_Factory.sol";
import {SmartCommitmentForwarder} from  "../../contracts/LenderCommitmentForwarder/SmartCommitmentForwarder.sol";

import "../tokens/TestERC20Token.sol";  

import { ILenderCommitmentForwarder_U1 } from "../../contracts/interfaces/ILenderCommitmentForwarder_U1.sol";


import { Testable } from "../Testable.sol";

import "../../contracts/interfaces/ILenderCommitmentForwarder.sol";
import { LenderCommitmentForwarder_G2 } from "../../contracts/LenderCommitmentForwarder/LenderCommitmentForwarder_G2.sol";

import { Collateral, CollateralType } from "../../contracts/interfaces/escrow/ICollateralEscrowV1.sol";

import { User } from "../Test_Helpers.sol";

import "../../contracts/mock/MarketRegistryMock.sol";
 
import { UniswapV3PoolMock } from "../../contracts/mock/uniswap/UniswapV3PoolMock.sol";

import { UniswapV3FactoryMock } from "../../contracts/mock/uniswap/UniswapV3FactoryMock.sol";

import { ILenderCommitmentGroup } from "../../contracts/interfaces/ILenderCommitmentGroup.sol";

import { IUniswapPricingLibrary } from "../../contracts/interfaces/IUniswapPricingLibrary.sol";

import "../../contracts/libraries/uniswap/FullMath.sol";

import "forge-std/console.sol";

 

 


contract LenderCommitmentGroupFactory_Test is Testable {
    LenderCommitmentForwarderTest_TellerV2Mock private tellerV2Mock;
    MarketRegistryMock mockMarketRegistry;

    LenderCommitmentGroupFactory factory ;

    User private marketOwner;
    User private lender;
    User private borrower;

    address[] emptyArray;
    address[] borrowersArray;

    TestERC20Token principalToken;
    uint8 principalTokenDecimals = 18;

    TestERC20Token collateralToken;
    uint8 collateralTokenDecimals = 18;

    TestERC20Token intermediateToken;
    uint8 intermediateTokenDecimals = 18;
 

    SmartCommitmentForwarder smartCommitmentForwarder;

    uint256 maxPrincipal;
    uint32 expiration;
    uint32 maxDuration;
    uint16 minInterestRate;
    // address collateralTokenAddress;
    uint256 collateralTokenId;
    uint256 maxPrincipalPerCollateralAmount;
    ILenderCommitmentForwarder_U1.CommitmentCollateralType collateralTokenType;

    uint256 marketId;

    UniswapV3FactoryMock mockUniswapFactory;
    UniswapV3PoolMock mockUniswapPool;
    UniswapV3PoolMock mockUniswapPoolSecondary;

    //  address principalTokenAddress;

    constructor() {}

    function setUp() public {
        tellerV2Mock = new LenderCommitmentForwarderTest_TellerV2Mock();
        mockMarketRegistry = new MarketRegistryMock();

        mockUniswapFactory = new UniswapV3FactoryMock();
        mockUniswapPool = new UniswapV3PoolMock();

        mockUniswapPoolSecondary = new UniswapV3PoolMock();

        smartCommitmentForwarder = new SmartCommitmentForwarder(
            address(tellerV2Mock),
            address(mockMarketRegistry)
            //address(mockUniswapFactory)
        );

        smartCommitmentForwarder.initialize();
    
    
        LenderCommitmentGroup_Smart lenderGroupPoolImplementation = new LenderCommitmentGroup_Smart(
            address(tellerV2Mock),
            address(smartCommitmentForwarder),
            address(mockUniswapFactory)

        );
         // Step 4: Deploy the Beacon with the implementation
        UpgradeableBeacon lenderGroupPoolBeacon = new UpgradeableBeacon(address(lenderGroupPoolImplementation));



        factory = new LenderCommitmentGroupFactory();
        factory.initialize( address(lenderGroupPoolBeacon) );

        marketOwner = new User( address(tellerV2Mock)  );
        borrower = new User( address(tellerV2Mock)  );
        lender = new User( address(tellerV2Mock)  );

        tellerV2Mock.__setMarketRegistry(address(mockMarketRegistry));
        mockMarketRegistry.setMarketOwner(address(marketOwner));

        //tokenAddress = address(0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174);
        marketId = 2;
        maxPrincipal = 100000000000000000000;
        maxPrincipalPerCollateralAmount = 100;
        maxDuration = 2480000;
        minInterestRate = 3000;
        expiration = uint32(block.timestamp) + uint32(64000);

        marketOwner.setTrustedMarketForwarder(
            marketId,
            address(smartCommitmentForwarder)
        );
        lender.approveMarketForwarder(
            marketId,
            address(smartCommitmentForwarder)
        );

        borrowersArray = new address[](1);
        borrowersArray[0] = address(borrower);

        principalToken = new TestERC20Token(
            "Test Wrapped ETH",
            "TWETH",
            1e32,
            principalTokenDecimals
        );

         

        collateralToken = new TestERC20Token(
            "Test USDC",
            "TUSDC",
            1e32,
            collateralTokenDecimals
        );

    }

 

     function test_deployPool() public {

        address _principalTokenAddress = address(principalToken);
        address _collateralTokenAddress = address(collateralToken);

        bool zeroForOne = false; 
        uint32 twapInterval = 0;


        uint256 initialPrincipalAmount = 10000000;


        principalToken.approve( address(factory), initialPrincipalAmount ) ;

         ILenderCommitmentGroup.CommitmentGroupConfig memory groupConfig = ILenderCommitmentGroup.CommitmentGroupConfig({
            principalTokenAddress: _principalTokenAddress,
            collateralTokenAddress: _collateralTokenAddress,
            marketId: marketId,
            maxLoanDuration: maxDuration,
            interestRateLowerBound: minInterestRate,
            interestRateUpperBound: minInterestRate,
            liquidityThresholdPercent: 7500,
            collateralRatio: 10000   //1-1  
        });



           IUniswapPricingLibrary.PoolRouteConfig
            memory routeConfig = IUniswapPricingLibrary.PoolRouteConfig({
                pool: address(mockUniswapPool),
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


 
       address newGroupContract = factory.deployLenderCommitmentGroupPool(
            initialPrincipalAmount,
            groupConfig,
            routesConfig
        );

         assertEq( newGroupContract == address(0x0),  false   );

    }

     function test_deployPool_no_initial_principal() public {

        address _principalTokenAddress = address(principalToken);
        address _collateralTokenAddress = address(collateralToken);

        bool zeroForOne = false; 
        uint32 twapInterval = 0;


        uint256 initialPrincipalAmount = 0;

         ILenderCommitmentGroup.CommitmentGroupConfig memory groupConfig = ILenderCommitmentGroup.CommitmentGroupConfig({
            principalTokenAddress: _principalTokenAddress,
            collateralTokenAddress: _collateralTokenAddress,
            marketId: marketId,
            maxLoanDuration: maxDuration,
            interestRateLowerBound: minInterestRate,
            interestRateUpperBound: minInterestRate,
            liquidityThresholdPercent: 7500,
            collateralRatio: 10000   //1-1  
        });



           IUniswapPricingLibrary.PoolRouteConfig
            memory routeConfig = IUniswapPricingLibrary.PoolRouteConfig({
                pool: address(mockUniswapPool),
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


 
       address newGroupContract = factory.deployLenderCommitmentGroupPool(
            initialPrincipalAmount,
            groupConfig,
            routesConfig
        );

         assertEq( newGroupContract == address(0x0),  false   );

    }

   
  
  

}
 
//Move to a helper file !
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
