// SPDX-License-Identifier: BSD-3-Clause
pragma solidity 0.8.22;

import {BaseTest} from "./BaseTest.t.sol";
import "forge-std/Test.sol";

contract ComptrollerTest is BaseTest {
    function setUp() public {
        _deployBaselineContracts();
    }

    function test_updateSupplyIndex() public {}
}
