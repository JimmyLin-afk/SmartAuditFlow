pragma solidity >=0.8.0 <0.9.0;
// SPDX-License-Identifier: MIT

 
import "../interfaces/ILenderCommitmentGroup.sol";

import "../libraries/NumbersLib.sol";
  
 import "@openzeppelin/contracts-upgradeable/token/ERC20/extensions/IERC20MetadataUpgradeable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
 
contract LenderGroupMockInteract {
   
      
 
    function addPrincipalToCommitmentGroup(  
        address _target,
        address _principalToken,
        address _sharesToken,
        uint256 _amount,
        address _sharesRecipient,
        uint256 _minSharesAmountOut ) public   {

             // Transfer the ERC20 tokens to this contract
        IERC20(_principalToken).transferFrom(msg.sender, address(this), _amount);

        // Approve the target contract to spend tokens on behalf of this contract
        IERC20(_principalToken).approve(_target, _amount);


            //need to suck in ERC 20 and approve them ? 

        uint256 sharesAmount = ILenderCommitmentGroup(_target).addPrincipalToCommitmentGroup(
            _amount,
            _sharesRecipient,
            _minSharesAmountOut 

        );

       
        IERC20(_sharesToken).transfer(msg.sender, sharesAmount);
    }

        /**
     * @notice Allows the contract owner to prepare shares for withdrawal in the specified LenderCommitmentGroup contract.
     * @param _target The LenderCommitmentGroup contract address.
     * @param _amountPoolSharesTokens The amount of share tokens to prepare for withdrawal.
     */
    function prepareSharesForBurn(
        address _target,
        uint256 _amountPoolSharesTokens
    ) external   {
        ILenderCommitmentGroup(_target).prepareSharesForBurn(
            _amountPoolSharesTokens
        );
    }

    /**
     * @notice Allows the contract owner to burn shares and withdraw principal tokens from the specified LenderCommitmentGroup contract.
     * @param _target The LenderCommitmentGroup contract address.
     * @param _amountPoolSharesTokens The amount of share tokens to burn.
     * @param _recipient The address to receive the withdrawn principal tokens.
     * @param _minAmountOut The minimum amount of principal tokens expected from the withdrawal.
     */
    function burnSharesToWithdrawEarnings(
        address _target,
        address _principalToken,
        address _poolSharesToken,
        uint256 _amountPoolSharesTokens,
        address _recipient,
        uint256 _minAmountOut
    ) external   {



             // Transfer the ERC20 tokens to this contract
        IERC20(_poolSharesToken).transferFrom(msg.sender, address(this), _amountPoolSharesTokens);

        // Approve the target contract to spend tokens on behalf of this contract
        IERC20(_poolSharesToken).approve(_target, _amountPoolSharesTokens);



        uint256 amountOut = ILenderCommitmentGroup(_target).burnSharesToWithdrawEarnings(
            _amountPoolSharesTokens,
            _recipient,
            _minAmountOut
        );

         IERC20(_principalToken).transfer(msg.sender, amountOut);
    }

    /**
     * @notice Allows the contract owner to liquidate a defaulted loan with incentive in the specified LenderCommitmentGroup contract.
     * @param _target The LenderCommitmentGroup contract address.
     * @param _bidId The ID of the defaulted loan bid to liquidate.
     * @param _tokenAmountDifference The incentive amount required for liquidation.
     */
    function liquidateDefaultedLoanWithIncentive(
        address _target,
         address _principalToken,
        uint256 _bidId,
        int256 _tokenAmountDifference,
        int256 loanAmount 
    ) external   {

         // Transfer the ERC20 tokens to this contract
        IERC20(_principalToken).transferFrom(msg.sender, address(this), uint256(_tokenAmountDifference + loanAmount));

        // Approve the target contract to spend tokens on behalf of this contract
        IERC20(_principalToken).approve(_target, uint256(_tokenAmountDifference + loanAmount));
 

        ILenderCommitmentGroup(_target).liquidateDefaultedLoanWithIncentive(
            _bidId,
            _tokenAmountDifference
        );
    }





    
}
