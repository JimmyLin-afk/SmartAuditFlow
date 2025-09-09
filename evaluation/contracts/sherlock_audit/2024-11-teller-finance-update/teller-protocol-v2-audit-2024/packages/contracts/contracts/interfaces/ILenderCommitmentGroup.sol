// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;


import {IUniswapPricingLibrary} from "../interfaces/IUniswapPricingLibrary.sol";
interface ILenderCommitmentGroup {



    struct CommitmentGroupConfig {
        address principalTokenAddress;
        address collateralTokenAddress;
        uint256 marketId;
        uint32 maxLoanDuration;
        uint16 interestRateLowerBound;
        uint16 interestRateUpperBound;
        uint16 liquidityThresholdPercent;
        uint16 collateralRatio; //essentially the overcollateralization ratio.  10000 is 1:1 baseline ?
        
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
        );

    function addPrincipalToCommitmentGroup(
        uint256 _amount,
        address _sharesRecipient,
        uint256 _minAmountOut
    ) external returns (uint256 sharesAmount_);



     function prepareSharesForBurn(
        uint256 _amountPoolSharesTokens 
    ) external returns (bool);

     function  burnSharesToWithdrawEarnings(
        uint256 _amountPoolSharesTokens,
        address _recipient,
        uint256 _minAmountOut
    ) external returns (uint256);


     function liquidateDefaultedLoanWithIncentive(
        uint256 _bidId,
        int256 _tokenAmountDifference
    ) external ;

    function getTokenDifferenceFromLiquidations() external view returns (int256);

}
