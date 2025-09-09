// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0 <0.9.0;

//tellerv2 should support this
interface ILoanRepaymentCallbacks {
    function setRepaymentListenerForBid(uint256 _bidId, address _listener)
        external;

    function getRepaymentListenerForBid(uint256 _bidId)
        external
        view
        returns (address);
}
