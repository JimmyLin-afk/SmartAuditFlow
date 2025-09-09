// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.26;

// External imports
import {Math} from "@openzeppelin/contracts/utils/math/Math.sol";

// Internal imports
import {ExternalAction} from "src/types/DataTypes.sol";
import {ILeverageToken} from "src/interfaces/ILeverageToken.sol";
import {FeeManagerTest} from "test/unit/FeeManager/FeeManager.t.sol";

contract ComputeTokenFeeTest is FeeManagerTest {
    function test_computeTokenFee_Mint() public {
        ExternalAction action = ExternalAction.Mint;
        uint256 shares = 1 ether;
        uint256 mintTokenFee = 200;
        uint256 redeemTokenFee = 400;
        _setLeverageTokenActionFees(mintTokenFee, redeemTokenFee);

        (uint256 sharesAfterFee, uint256 tokenFee) = feeManager.exposed_computeTokenFee(leverageToken, shares, action);

        uint256 expectedTokenFee = Math.mulDiv(shares, mintTokenFee, MAX_BPS, Math.Rounding.Ceil);
        assertEq(tokenFee, expectedTokenFee);

        assertEq(sharesAfterFee, shares - expectedTokenFee);
    }

    function test_computeTokenFee_Redeem() public {
        ExternalAction action = ExternalAction.Redeem;
        uint256 shares = 1 ether;
        uint256 mintTokenFee = 200;
        uint256 redeemTokenFee = 400;
        _setLeverageTokenActionFees(mintTokenFee, redeemTokenFee);

        (uint256 sharesAfterFee, uint256 tokenFee) = feeManager.exposed_computeTokenFee(leverageToken, shares, action);

        uint256 expectedTokenFee = Math.mulDiv(shares, redeemTokenFee, MAX_BPS, Math.Rounding.Ceil);
        assertEq(tokenFee, expectedTokenFee);

        assertEq(sharesAfterFee, shares + expectedTokenFee);
    }

    function testFuzz_computeTokenFee(uint128 shares, uint256 mintTokenFee, uint256 redeemTokenFee) public {
        ExternalAction action = ExternalAction.Mint;
        mintTokenFee = bound(mintTokenFee, 0, MAX_ACTION_FEE);
        redeemTokenFee = bound(redeemTokenFee, 0, MAX_ACTION_FEE);
        _setLeverageTokenActionFees(mintTokenFee, redeemTokenFee);

        (uint256 sharesAfterFee, uint256 tokenFee) = feeManager.exposed_computeTokenFee(leverageToken, shares, action);

        uint256 expectedTokenFee = Math.mulDiv(
            shares, action == ExternalAction.Mint ? mintTokenFee : redeemTokenFee, MAX_BPS, Math.Rounding.Ceil
        );
        assertEq(tokenFee, expectedTokenFee);

        uint256 expectedSharesAfterFee =
            action == ExternalAction.Mint ? shares - expectedTokenFee : shares + expectedTokenFee;
        assertEq(sharesAfterFee, expectedSharesAfterFee);

        assertLe(tokenFee, shares);
        assertLe(expectedSharesAfterFee, shares);
    }
}
