// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.26;

// Internal imports
import {FeeManagerTest} from "test/unit/FeeManager/FeeManager.t.sol";

contract ChargeTreasuryFeeTest is FeeManagerTest {
    /// forge-config: default.fuzz.runs = 1
    function testFuzz_chargeTreasuryFee(uint256 shares) public {
        vm.prank(feeManagerRole);
        feeManager.setTreasury(treasury);

        feeManager.exposed_chargeTreasuryFee(leverageToken, shares);

        assertEq(leverageToken.balanceOf(treasury), shares);
    }
}
