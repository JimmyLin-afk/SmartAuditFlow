// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/utils/Address.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "../../contracts/TellerV2MarketForwarder_G1.sol";

import "../../contracts/TellerV2Context.sol";

import { LenderCommitmentForwarder_G2 } from "../../contracts/LenderCommitmentForwarder/LenderCommitmentForwarder_G2.sol";

import { Collateral, CollateralType } from "../../contracts/interfaces/escrow/ICollateralEscrowV1.sol";

import { User } from "../Test_Helpers.sol";

import "../../contracts/mock/MarketRegistryMock.sol";

contract LenderCommitmentForwarder_Override is LenderCommitmentForwarder_G2 {
    bool public submitBidWasCalled;
    bool public submitBidWithCollateralWasCalled;
    bool public acceptBidWasCalled;

    constructor(address tellerV2, address marketRegistry)
        LenderCommitmentForwarder_G2(tellerV2, marketRegistry)
    {}

    function setCommitment(uint256 _commitmentId, Commitment memory _commitment)
        public
    {
        commitments[_commitmentId] = _commitment;
    }

    function _getEscrowCollateralTypeSuper(CommitmentCollateralType _type)
        public
        returns (CollateralType)
    {
        return super._getEscrowCollateralType(_type);
    }

    function validateCommitmentSuper(uint256 _commitmentId) public {
        super.validateCommitment(commitments[_commitmentId]);
    }

    /*
        Overrider methods 
    */

    function _submitBidWithCollateral(CreateLoanArgs memory, address)
        internal
        override
        returns (uint256 bidId)
    {
        submitBidWithCollateralWasCalled = true;
        return 1;
    }

    function _acceptBid(uint256, address) internal override returns (bool) {
        acceptBidWasCalled = true;

        return true;
    }
}

contract LenderCommitmentForwarderTest_TellerV2Mock is TellerV2Context {
    constructor() TellerV2Context(address(0)) {}

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
