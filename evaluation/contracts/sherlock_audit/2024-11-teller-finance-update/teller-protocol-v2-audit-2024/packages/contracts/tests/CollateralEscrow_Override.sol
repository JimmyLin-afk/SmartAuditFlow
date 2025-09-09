// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { Testable } from "./Testable.sol";

import { CollateralEscrowV1 } from "../contracts/escrow/CollateralEscrowV1.sol";
import "../contracts/mock/WethMock.sol";

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "../contracts/interfaces/IWETH.sol";
import { CollateralType, CollateralEscrowV1 } from "../contracts/escrow/CollateralEscrowV1.sol";

import { Collateral } from "../contracts/interfaces/escrow/ICollateralEscrowV1.sol";

contract CollateralEscrowV1_Override is CollateralEscrowV1 {
    function setStoredBalance(
        CollateralType collateralType,
        address assetContractAddress,
        uint256 assetAmount,
        uint256 tokenId,
        address owner
    ) public {
        collateralBalances[assetContractAddress] = Collateral({
            _amount: assetAmount,
            _collateralAddress: assetContractAddress,
            _collateralType: collateralType,
            _tokenId: tokenId
        });
    }

    function _depositCollateralSuper(
        CollateralType _collateralType,
        address _collateralAddress,
        uint256 _amount,
        uint256 _tokenId
    ) public {
        super._depositCollateral(
            _collateralType,
            _collateralAddress,
            _amount,
            _tokenId
        );
    }

    function _withdrawCollateralSuper(
        Collateral memory _collateral,
        address _collateralAddress,
        uint256 _amount,
        address _recipient
    ) public {
        super._withdrawCollateral(
            _collateral,
            _collateralAddress,
            _amount,
            _recipient
        );
    }
}
