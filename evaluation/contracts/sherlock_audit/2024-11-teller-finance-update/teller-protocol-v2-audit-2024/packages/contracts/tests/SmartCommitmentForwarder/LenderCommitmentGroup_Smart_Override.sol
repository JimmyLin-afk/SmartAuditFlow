// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/utils/Address.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

import { LenderCommitmentGroup_Smart } from "../../contracts/LenderCommitmentForwarder/extensions/LenderCommitmentGroup/LenderCommitmentGroup_Smart.sol";

contract LenderCommitmentGroup_Smart_Override is LenderCommitmentGroup_Smart {
    //  bool public submitBidWasCalled;
    // bool public submitBidWithCollateralWasCalled;
    //  bool public acceptBidWasCalled;

    uint256 mockMaxPrincipalPerCollateralAmount;
    uint256 mockRequiredCollateralAmount;
    uint256 mockSharesExchangeRate;
    int256 mockMinimumAmountDifferenceToCloseDefaultedLoan;

    uint256 mockAmountOwed;

    address mockToken0;
    address mockToken1;

    constructor(address _tellerV2, address _smartCommitmentForwarder, address _uniswapV3Pool)
        LenderCommitmentGroup_Smart(_tellerV2,_smartCommitmentForwarder, _uniswapV3Pool)
    {}

    function set_mockSharesExchangeRate(uint256 _mockRate) public {
        mockSharesExchangeRate = _mockRate;
    }

    function set_mockBidAsActiveForGroup(uint256 _bidId,bool _active) public {
        activeBids[_bidId] = _active;
    }
 


     function mock_setMinimumAmountDifferenceToCloseDefaultedLoan(
        int256 _amt
    ) external   returns (uint256){
       mockMinimumAmountDifferenceToCloseDefaultedLoan = _amt;
    } 



   /* function mock_prepareSharesForWithdraw(
        uint256 _amountPoolSharesTokens
    ) external   {
        poolSharesPreparedToWithdrawForLender[msg.sender] = _amountPoolSharesTokens; 
        poolSharesPreparedTimestamp[msg.sender] = block.timestamp;
       
    } */


    function getMinimumAmountDifferenceToCloseDefaultedLoan(
        
        uint256 _amountOwed,
        uint256 _loanDefaultedTimestamp
    ) public view override returns (int256 amountDifference_ ) {

        return mockMinimumAmountDifferenceToCloseDefaultedLoan;
    }
    
    function super_getMinimumAmountDifferenceToCloseDefaultedLoan(
        
        uint256 _amountOwed,
        uint256 _loanDefaultedTimestamp
    ) public view returns (int256  ) {

        return super.getMinimumAmountDifferenceToCloseDefaultedLoan(_amountOwed,_loanDefaultedTimestamp);
    }

    function _getAmountOwedForBid(uint256 _bidId )
     internal override view returns (uint256){
        return mockAmountOwed;

     }

    function set_mockAmountOwedForBid(uint256 _amt) public {
        mockAmountOwed = _amt;
    }


    function set_totalPrincipalTokensRepaid(uint256 _mockAmt) public {
        totalPrincipalTokensRepaid = _mockAmt;
    }

    function set_totalPrincipalTokensCommitted(uint256 _mockAmt) public {
        totalPrincipalTokensCommitted = _mockAmt;
    }

      function set_totalPrincipalTokensWithdrawn(uint256 _mockAmt) public {
        totalPrincipalTokensWithdrawn = _mockAmt;
    }

    function set_totalInterestCollected(uint256 _mockAmt) public {
        totalInterestCollected = _mockAmt;
    }

      function set_tokenDifferenceFromLiquidations(int256 _mockAmt) public {
        tokenDifferenceFromLiquidations = _mockAmt;
    }
 

    function mock_mintShares(address _sharesRecipient, uint256 _mockAmt)
        public
    {
        poolSharesToken.mint(_sharesRecipient, _mockAmt);
    }

    function set_mock_getMaxPrincipalPerCollateralAmount(uint256 amt) public {
        mockMaxPrincipalPerCollateralAmount = amt;
    }

      function set_mock_requiredCollateralAmount(uint256 amt) public {
        mockRequiredCollateralAmount = amt;
    }

    function mock_setFirstDepositMade(bool made) public {
        firstDepositMade = made;

    }

    function sharesExchangeRate() public override view returns (uint256 rate_) {
        
        if (mockSharesExchangeRate > 0){
                 return mockSharesExchangeRate;

        }

        return super.sharesExchangeRate();

    }


    function super_sharesExchangeRate(  ) public view returns (uint256) {

        return super.sharesExchangeRate();
    }



    function super_sharesExchangeRateInverse(  ) public view returns (uint256) {

        return super.sharesExchangeRateInverse();
    }

    function mock_setPoolTokens (address token0, address token1) public {

          mockToken0 = token0;

          mockToken1 = token1;
        
    }

    function mock_setBidActive (uint256 _bidId) public {

         activeBids[_bidId] = true;
        
    }

    
/*
    function _getPoolTokens() internal view override returns (address token0, address token1) {

        return (mockToken0,mockToken1); 
        
    }
*/
  /*  function super_getCollateralTokensAmountEquivalentToPrincipalTokens(
        uint256 principalTokenAmountValue,
        uint256 pairPriceWithTwap 
     //   uint256 pairPriceImmediate,  
      //  bool principalTokenIsToken0
    ) public view returns(uint256){

        return super._getCollateralTokensAmountEquivalentToPrincipalTokens(
            principalTokenAmountValue,
            pairPriceWithTwap 
            //pairPriceImmediate,
           // principalTokenIsToken0
        );

    }

*/


    function getRequiredCollateral(
       uint256 _principalAmount,
       uint256 maxPrincipalPerCollateralAmount 
       
    ) public view override returns (uint256 collateralTokensAmountToMatchValue) {
 
        return  mockRequiredCollateralAmount  ;
    }

/*
    function calculateCollateralTokensAmountEquivalentToPrincipalTokens(
        uint256 principalTokenAmountValue
    ) public view override returns (uint256 collateralTokensAmountToMatchValue) {

            //this is not correct 
        return
            principalTokenAmountValue 
            * mockMaxPrincipalPerCollateralAmount  ;
    }*/




/*
    function super_getPriceFromSqrtX96(uint160 _sqrtPriceX96) public pure returns (uint256 price_) {

        price_ =  super._getPriceFromSqrtX96(_sqrtPriceX96);
    }

*/



    /*
    function _getMaxPrincipalPerCollateralAmount(  ) internal override view  returns (uint256) {

       return mockMaxPrincipalPerCollateralAmount ;

    }

    function _super_getMaxPrincipalPerCollateralAmount(  ) public view  returns (uint256) {

       return super._getMaxPrincipalPerCollateralAmount() ;

    }*/
}