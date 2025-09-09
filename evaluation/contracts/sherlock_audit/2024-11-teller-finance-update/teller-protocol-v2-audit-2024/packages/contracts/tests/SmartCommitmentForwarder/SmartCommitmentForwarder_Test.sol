// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/utils/Address.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
//import "../../contracts/TellerV2MarketForwarder_G1.sol";
import "../../contracts/TellerV2Context.sol";
import {SmartCommitmentForwarder} from  "../../contracts/LenderCommitmentForwarder/SmartCommitmentForwarder.sol";

import {LenderCommitmentGroup_SmartMock} from  "../../contracts/mock/LenderCommitmentGroup_SmartMock.sol";

import "../tokens/TestERC20Token.sol"; 
//import "../../contracts/TellerV2Context.sol";

import { ILenderCommitmentForwarder_U1 } from "../../contracts/interfaces/ILenderCommitmentForwarder_U1.sol";


import { Testable } from "../Testable.sol";

import "../../contracts/interfaces/ILenderCommitmentForwarder.sol";
import { LenderCommitmentForwarder_G2 } from "../../contracts/LenderCommitmentForwarder/LenderCommitmentForwarder_G2.sol";

import { Collateral, CollateralType } from "../../contracts/interfaces/escrow/ICollateralEscrowV1.sol";

import { User } from "../Test_Helpers.sol";

import "../../contracts/mock/MarketRegistryMock.sol";
 
import { UniswapV3PoolMock } from "../../contracts/mock/uniswap/UniswapV3PoolMock.sol";

import { UniswapV3FactoryMock } from "../../contracts/mock/uniswap/UniswapV3FactoryMock.sol";
import { TellerV2SolMock } from "../../contracts/mock/TellerV2SolMock.sol";
import { ProtocolPausingManager } from "../../contracts/pausing/ProtocolPausingManager.sol";


import "../../contracts/libraries/uniswap/FullMath.sol";

import "forge-std/console.sol";

 

 


contract SmartCommitmentForwarder_Test is Testable {
    TellerV2SolMock private tellerV2Mock;
    MarketRegistryMock mockMarketRegistry;

    ProtocolPausingManager protocolPausingManager; 

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

    LenderCommitmentGroup_SmartMock lenderPool;

    //  address principalTokenAddress;

    constructor() {}

    function setUp() public {
        tellerV2Mock = new TellerV2SolMock();
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


        protocolPausingManager = new ProtocolPausingManager();
        protocolPausingManager.initialize(); //this becomes the owner 

        marketOwner = new User( address(tellerV2Mock)  );
        borrower = new User( address(tellerV2Mock)  );
        lender = new User( address(tellerV2Mock)  );

        tellerV2Mock.setMarketRegistry(address(mockMarketRegistry));
        mockMarketRegistry.setMarketOwner(address(marketOwner));

        //tokenAddress = address(0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174);
        marketId = 2;
        maxPrincipal = 100000000000000000000;
        maxPrincipalPerCollateralAmount = 100;
        maxDuration = 2480000;
        minInterestRate = 3000;
        expiration = uint32(block.timestamp) + uint32(64000);

// only need these if using the real tellerv2 
     /*
        marketOwner.setTrustedMarketForwarder(
            marketId,
            address(smartCommitmentForwarder)
        );
       lender.approveMarketForwarder(
            marketId,
            address(smartCommitmentForwarder)
        );*/

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


        lenderPool = new LenderCommitmentGroup_SmartMock(
            address(tellerV2Mock),
            address(smartCommitmentForwarder),
            address(mockUniswapFactory)
        );

        
    }


     function test_cant_reinit() public {


        vm.expectRevert("Initializable: contract is already initialized");
        smartCommitmentForwarder.initialize();
     }

    function test_acceptSmartCommitment() public {

        uint256 principalAmount = 1e6;
        uint256 collateralAmount = 1e6; 
        uint16 interestRate = 500;
        uint32 loanDuration = 1e6;

        address recipient = address(this);



        uint256 bidId = smartCommitmentForwarder.acceptSmartCommitmentWithRecipient(
            address(lenderPool),
            principalAmount,
            collateralAmount,
            0,
            address(collateralToken),
            address(recipient),
            interestRate,
            loanDuration
        );



    }

      function test_acceptSmartCommitment_fails_when_paused() public {


        tellerV2Mock.setProtocolPausingManager(address(protocolPausingManager) );

         protocolPausingManager.addPauser(address(this));

         smartCommitmentForwarder.pause();



        uint256 principalAmount = 1e6;
        uint256 collateralAmount = 1e6; 
        uint16 interestRate = 500;
        uint32 loanDuration = 1e6;

        address recipient = address(this);


        vm.expectRevert("Pausable: paused");
        uint256 bidId = smartCommitmentForwarder.acceptSmartCommitmentWithRecipient(
            address(lenderPool),
            principalAmount,
            collateralAmount,
            0,
            address(collateralToken),
            address(recipient),
            interestRate,
            loanDuration
        );



    }




     function test_setLiquidationProtocolFeePercent() public {

          tellerV2Mock.setMockOwner(address(this));
           
         smartCommitmentForwarder.setLiquidationProtocolFeePercent(555);

         uint256 fee = smartCommitmentForwarder.getLiquidationProtocolFeePercent();

         assertEq(fee , 555);

     }


     function test_setLiquidationProtocolFeePercent_not_authorized() public {

      
         
           vm.expectRevert("Sender not authorized");
         smartCommitmentForwarder.setLiquidationProtocolFeePercent(555);

          
     }



     function test_pause() public {

         tellerV2Mock.setProtocolPausingManager(address(protocolPausingManager) );

         protocolPausingManager.addPauser(address(this));

         smartCommitmentForwarder.pause();

     }

     function test_cannot_pause () public {

         tellerV2Mock.setProtocolPausingManager(address(protocolPausingManager) );

         //protocolPausingManager.addPauser(address(this));
          protocolPausingManager.renounceOwnership();

         vm.expectRevert("Sender not authorized");
         smartCommitmentForwarder.pause();  
 
     }

      function test_pause_cannot_double_pause() public {

         tellerV2Mock.setProtocolPausingManager(address(protocolPausingManager) );

         protocolPausingManager.addPauser(address(this));

         smartCommitmentForwarder.pause();  

         vm.expectRevert("Pausable: paused");
         smartCommitmentForwarder.pause();

     }

     function test_unpause() public {

         tellerV2Mock.setProtocolPausingManager(address(protocolPausingManager) );

         protocolPausingManager.addPauser(address(this));

         smartCommitmentForwarder.pause();

         smartCommitmentForwarder.unpause();


     }

     function test_unpause_sets_getLastUnpausedAt() public {

         tellerV2Mock.setProtocolPausingManager(address(protocolPausingManager) );

         protocolPausingManager.addPauser(address(this));

         vm.warp(15000);
         smartCommitmentForwarder.pause();

         smartCommitmentForwarder.unpause();

         vm.warp(25000);
         
         uint256 lastUnpausedAt = smartCommitmentForwarder.getLastUnpausedAt();
 
         assertEq( lastUnpausedAt , 15000 );
     }


     function test_cannot_unpause() public {

         tellerV2Mock.setProtocolPausingManager(address(protocolPausingManager) );

         protocolPausingManager.addPauser(address(this));
 
         smartCommitmentForwarder.pause();

         protocolPausingManager.removePauser(address(this));

        protocolPausingManager.renounceOwnership();

          vm.expectRevert("Sender not authorized");
         smartCommitmentForwarder.unpause();


     }


      function test_unpause_cant_unpause() public {

         tellerV2Mock.setProtocolPausingManager(address(protocolPausingManager) );

         protocolPausingManager.addPauser(address(this));
    
         vm.expectRevert("Pausable: not paused");
         smartCommitmentForwarder.unpause();


     }

     function test_setOracle() public {

          tellerV2Mock.setMockOwner(address(this));

          smartCommitmentForwarder.setOracle(address(this));

     }

 
     function test_setOracle_unauthorized() public {

            vm.expectRevert("Sender not authorized");
          smartCommitmentForwarder.setOracle(address(this));

     }


     function test_setIsStrictMode() public {

        tellerV2Mock.setMockOwner(address(this));

        smartCommitmentForwarder.setIsStrictMode(true);

     }

     function test_setIsStrictMode_unauthorized() public {

        vm.expectRevert("Sender not authorized");
        smartCommitmentForwarder.setIsStrictMode(true);

     }

    /*

    function acceptSmartCommitmentWithRecipient_test(){


        vm.prank(address(borrower));

        smartCommitmentForwarder.acceptSmartCommitmentWithRecipient(
            address(mockLenderGroupPool),
            principalAmount,
            collateralAmount,
            0,
            address(collateralTokenAddress),
            address(borrower),
            interestRate,
            loanDuration
        );


    }


     function pause_test(){


        
    }

      function unpause_test(){


        
    }

    function setOracle_test(){


        
    }
*/


  
  

}
 
//Move to a helper file !
/*
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
*/