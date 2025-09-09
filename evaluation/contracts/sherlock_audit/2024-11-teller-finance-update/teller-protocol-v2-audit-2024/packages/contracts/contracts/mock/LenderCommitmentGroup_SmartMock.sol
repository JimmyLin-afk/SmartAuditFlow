// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/utils/Address.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
//import "../TellerV2MarketForwarder.sol";

import "../TellerV2Context.sol";

//import { LenderCommitmentForwarder } from "../contracts/LenderCommitmentForwarder.sol";

import "../interfaces/ITellerV2.sol";
import "../interfaces/ILenderCommitmentForwarder.sol";
import "../interfaces/ILenderCommitmentGroup.sol";
import "../interfaces/ISmartCommitment.sol";

import "../interfaces/ITellerV2MarketForwarder.sol";

import { Collateral, CollateralType } from "../interfaces/escrow/ICollateralEscrowV1.sol";

import "../mock/MarketRegistryMock.sol";

contract LenderCommitmentGroup_SmartMock is
    ILenderCommitmentGroup,
    ISmartCommitment
{


     constructor(
        address _tellerV2,
        address _smartCommitmentForwarder,
        address _uniswapV3Factory
    )  {
        //TELLER_V2 = _tellerV2;
        //SMART_COMMITMENT_FORWARDER = _smartCommitmentForwarder;
        //UNISWAP_V3_FACTORY = _uniswapV3Factory;
    }



    function initialize(
    CommitmentGroupConfig calldata _commitmentGroupConfig,
    IUniswapPricingLibrary.PoolRouteConfig[] calldata _poolOracleRoutes  
        )
    external
    returns (
        //uint256 _maxPrincipalPerCollateralAmount //use oracle instead

        //ILenderCommitmentForwarder.Commitment calldata _createCommitmentArgs

        address poolSharesToken
    ){


    }

    function addPrincipalToCommitmentGroup(
        uint256 _amount,
        address _sharesRecipient,
        uint256 _minAmountOut
    ) external returns (uint256 sharesAmount_){

        
    }



     function prepareSharesForBurn(
        uint256 _amountPoolSharesTokens 
    ) external returns (bool){

        
    }

     function  burnSharesToWithdrawEarnings(
        uint256 _amountPoolSharesTokens,
        address _recipient,
        uint256 _minAmountOut
    ) external returns (uint256){

        
    }


     function liquidateDefaultedLoanWithIncentive(
        uint256 _bidId,
        int256 _tokenAmountDifference
    ) external{

        
    }

    function getTokenDifferenceFromLiquidations() external view returns (int256){

        
    }





    function getPrincipalTokenAddress() external view returns (address){

        
    }

    function getMarketId() external view returns (uint256){

        
    }

    function getCollateralTokenAddress() external view returns (address){

        
    }

    function getCollateralTokenType()
        external
        view
        returns (CommitmentCollateralType){

        
    }

    function getCollateralTokenId() external view returns (uint256){

        
    }

    function getMinInterestRate(uint256 _delta) external view returns (uint16){

        
    }

    function getMaxLoanDuration() external view returns (uint32){

        
    }

    function getPrincipalAmountAvailableToBorrow()
        external
        view
        returns (uint256){

        
    }

    function getRequiredCollateral(
        uint256 _principalAmount,
        uint256 _maxPrincipalPerCollateralAmount
        
        )
        external
        view
        returns (uint256){

        
    }

    
    function acceptFundsForAcceptBid(
        address _borrower,
        uint256 _bidId,
        uint256 _principalAmount,
        uint256 _collateralAmount,
        address _collateralTokenAddress,
        uint256 _collateralTokenId,
        uint32 _loanDuration,
        uint16 _interestRate
    ) external{

        
    }
}
