// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { Testable } from "./Testable.sol";

import { CollateralEscrowV1 } from "../contracts/escrow/CollateralEscrowV1.sol";
import "../contracts/mock/WethMock.sol";
import "@openzeppelin/contracts/proxy/beacon/UpgradeableBeacon.sol";
import "@openzeppelin/contracts/proxy/beacon/BeaconProxy.sol";

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "../contracts/interfaces/IWETH.sol";

import "./tokens/TestERC20Token.sol";
import "./tokens/TestERC721Token.sol";
import "./tokens/TestERC1155Token.sol";

import "../contracts/mock/TellerV2SolMock.sol";
import "../contracts/CollateralManager.sol";

contract CollateralManager_Override is CollateralManager {

     using EnumerableSetUpgradeable for EnumerableSetUpgradeable.AddressSet;
     
    bool public checkBalancesWasCalled;
    bool public checkBalanceWasCalled;
    address public withdrawInternalWasCalledToRecipient;
    bool public commitCollateralInternalWasCalled;

    bool bidsCollateralBackedGlobally;
    bool public checkBalanceGlobalValid = true;

    address public globalEscrowProxyAddress;

    bool public deployEscrowInternalWasCalled;
    bool public depositInternalWasCalled;

    //force adds collateral info for a bid even if it doesnt exist (for testing)
    function commitCollateralSuper(
        uint256 bidId,
        Collateral memory collateralInfo
    ) public {
        super._commitCollateral(bidId, collateralInfo);
    }

    function _depositSuper(uint256 _bidId, Collateral memory _collateralInfo)
        public
    {
        super._deposit(_bidId, _collateralInfo);
    }

    function _withdrawSuper(uint256 _bidId, address _receiver) public {
        super._withdraw(_bidId, _receiver);
    }

    function _commitCollateralSuper(
        uint256 _bidId,
        Collateral memory _collateralInfo
    ) public {
        super._commitCollateral(_bidId, _collateralInfo);
    }

    function isBidCollateralBackedSuper(uint256 _bidId) public returns (bool) {
        return super.isBidCollateralBacked(_bidId);
    }

    function _checkBalancesSuper(
        address _borrowerAddress,
        Collateral[] memory _collateralInfo,
        bool _shortCircut
    ) public returns (bool validated_, bool[] memory checks_) {

        checkBalancesWasCalled = true;

        return
            super._checkBalances(
                _borrowerAddress,
                _collateralInfo,
                _shortCircut
            );
    }

    function _checkBalanceSuper(
        address _borrowerAddress,
        Collateral memory _collateralInfo
    ) public returns (bool) {

        checkBalanceWasCalled = true;

        return super._checkBalance(_borrowerAddress, _collateralInfo);
    }

    function setBidsCollateralBackedGlobally(bool _backed) public {
        bidsCollateralBackedGlobally = _backed;
    }

    function _deployEscrowSuper(uint256 _bidId)
        public
        returns (address proxyAddress_, address borrower_)
    {
        return super._deployEscrow(_bidId);
    }

    function forceSetEscrowAddress(uint256 bidId, address _address) public {
        _escrows[bidId] = _address;
    }

    function setCheckBalanceGlobalValid(bool _valid) public {
        checkBalanceGlobalValid = _valid;
    }

    /*
        Overrides
    */

    function isBidCollateralBacked(uint256 _bidId)
        public
        override
        returns (bool)
    {
        return bidsCollateralBackedGlobally;
    }

 

    function _deposit(uint256 _bidId, Collateral memory collateralInfo)
        internal
        override
    {
        depositInternalWasCalled = true;
    }

    function forceSetBidCollateral(

        uint256 _bidId,
        Collateral[] memory newCollateralArray
    ) public {
         CollateralInfo storage collateral = _bidCollaterals[_bidId];

        for (uint256 i; i < newCollateralArray.length; i++) {
            address collateralAddress = newCollateralArray[i]._collateralAddress;

            collateral.collateralAddresses.add(collateralAddress );

            collateral.collateralInfo[collateralAddress] =  newCollateralArray[i];
        }

    }


/*
    function _checkBalances(
        address _borrowerAddress,
        Collateral[] memory _collateralInfo,
        bool _shortCircuit
    ) internal override view returns (bool validated_, bool[] memory checks_) {
        //checkBalancesWasCalled = true;

        validated_ = checkBalanceGlobalValid;
        checks_ = new bool[](0);
    }


    function _checkBalance(
        address _borrowerAddress,
        Collateral memory _collateralInfo
    ) internal override view  returns (bool) {
        //checkBalanceWasCalled = true;

        return checkBalanceGlobalValid;
    }*/

    //for mock purposes
    function setGlobalEscrowProxyAddress(address _address) public {
        globalEscrowProxyAddress = _address;
    }

    function _deployEscrow(uint256 _bidId)
        internal
        override
        returns (address proxyAddress_, address borrower_)
    {
        proxyAddress_ = globalEscrowProxyAddress;
        borrower_ = tellerV2.getLoanBorrower(_bidId);

        deployEscrowInternalWasCalled = true;
    }

    function _withdraw(uint256 _bidId, address recipient) internal override {
        withdrawInternalWasCalledToRecipient = recipient;
    }

    function _commitCollateral(
        uint256 _bidId,
        Collateral memory _collateralInfo
    ) internal override {
        commitCollateralInternalWasCalled = true;
    }
}
