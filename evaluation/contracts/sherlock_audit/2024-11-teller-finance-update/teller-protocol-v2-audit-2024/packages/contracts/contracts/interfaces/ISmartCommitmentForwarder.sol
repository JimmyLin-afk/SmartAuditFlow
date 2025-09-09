

pragma solidity >=0.8.0 <0.9.0;

interface ISmartCommitmentForwarder {
     
    function acceptSmartCommitmentWithRecipient(
        address _smartCommitmentAddress,
        uint256 _principalAmount,
        uint256 _collateralAmount,
        uint256 _collateralTokenId,
        address _collateralTokenAddress,
        address _recipient,
        uint16 _interestRate,
        uint32 _loanDuration
    ) external  returns (uint256 bidId)  ;


    function setLiquidationProtocolFeePercent(uint256 _percent) 
    external;

    function getLiquidationProtocolFeePercent() 
    external view returns (uint256) ;



}