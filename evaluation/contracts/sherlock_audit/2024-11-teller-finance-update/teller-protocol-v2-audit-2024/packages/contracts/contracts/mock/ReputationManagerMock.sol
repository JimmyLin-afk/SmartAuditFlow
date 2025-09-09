pragma solidity ^0.8.0;

// SPDX-License-Identifier: MIT

import "../interfaces/IReputationManager.sol";

contract ReputationManagerMock is IReputationManager {
    constructor() {}

    function initialize(address protocolAddress) external override {}

    function getDelinquentLoanIds(address _account)
        external
        returns (uint256[] memory _loanIds)
    {}

    function getDefaultedLoanIds(address _account)
        external
        returns (uint256[] memory _loanIds)
    {}

    function getCurrentDelinquentLoanIds(address _account)
        external
        returns (uint256[] memory _loanIds)
    {}

    function getCurrentDefaultLoanIds(address _account)
        external
        returns (uint256[] memory _loanIds)
    {}

    function updateAccountReputation(address _account) external {}

    function updateAccountReputation(address _account, uint256 _bidId)
        external
        returns (RepMark)
    {
        return RepMark.Good;
    }
}
