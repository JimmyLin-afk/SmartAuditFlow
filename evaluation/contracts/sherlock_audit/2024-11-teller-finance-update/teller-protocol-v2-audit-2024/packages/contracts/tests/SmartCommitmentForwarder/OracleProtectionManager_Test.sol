// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "forge-std/Test.sol";
import "../../contracts/oracleprotection/OracleProtectionManager.sol";
import "../../contracts/mock/MockHypernativeOracle.sol";

contract OracleProtectionManagerTest is Test {
    OracleProtectionManagerOverride manager;
    MockHypernativeOracle mockOracle;

    function setUp() public {
        manager = new OracleProtectionManagerOverride();
        mockOracle = new MockHypernativeOracle();
    }

    function test_initialization() public {
        assertEq(manager.getHypernativeOracle(), address(0), "Oracle should be unset initially");
        assertEq(manager.hypernativeOracleIsStrictMode(), false, "Strict mode should be off initially");
    }

    function test_setOracle() public {
        manager.setOracle(address(mockOracle));
        assertEq(manager.getHypernativeOracle(), address(mockOracle), "Oracle address not set correctly");
    }

    function test_setStrictMode() public {
        manager.setStrictMode(true);
        assertEq(manager.hypernativeOracleIsStrictMode(), true, "Strict mode not set correctly");

        manager.setStrictMode(false);
        assertEq(manager.hypernativeOracleIsStrictMode(), false, "Strict mode not unset correctly");
    }



    function test_oracleRegister_withStrictMode() public {
        manager.setOracle(address(mockOracle));
        manager.setStrictMode(true);

        vm.expectCall(address(mockOracle), abi.encodeWithSelector(mockOracle.registerStrict.selector, address(this)));
        manager.oracleRegister(address(this));
    }

    function test_oracleRegister_withoutStrictMode() public {
        manager.setOracle(address(mockOracle));
        manager.setStrictMode(false);

        vm.expectCall(address(mockOracle), abi.encodeWithSelector(mockOracle.register.selector, address(this)));
        manager.oracleRegister(address(this));
    }

    function test_isOracleApproved_withNoOracleSet() public {
        bool approved = manager.isOracleApproved(address(this));
        assertEq(approved, true, "Should approve when no oracle is set");
    }

   /*  function test_isOracleApproved_withOracleSet() public {
        manager.setOracle(address(mockOracle));
        mockOracle.setBlacklistedContext(address(this), false);
        mockOracle.setTimeExceeded(address(this), true);

        bool approved = manager.isOracleApproved(address(this));
        assertEq(approved, true, "Should approve valid address");

        mockOracle.setBlacklistedContext(address(this), true);
        approved = manager.isOracleApproved(address(this));
        assertEq(approved, false, "Should reject blacklisted address");
    } */

    function test_isOracleApprovedAllowEOA() public {
        manager.setOracle(address(mockOracle));
        mockOracle.setBlacklistedContext(tx.origin, false);
        mockOracle.setTimeExceeded(address(this), true);

        bool approved = manager.isOracleApprovedAllowEOA(address(this));
        assertEq(approved, true, "Should approve valid address");

        mockOracle.setBlacklistedContext(tx.origin, true);
        approved = manager.isOracleApprovedAllowEOA(address(this));
        assertEq(approved, false, "Should reject blacklisted address");

        mockOracle.setBlacklistedContext(tx.origin, false);
        mockOracle.setTimeExceeded(address(this), false);
        approved = manager.isOracleApprovedAllowEOA(address(this));
        assertEq(approved, false, "Should reject unregistered contract");
    }

    function test_isOracleApprovedOnlyAllowEOA() public {
        manager.setOracle(address(mockOracle));
        mockOracle.setBlacklistedAccount(tx.origin, false);

        bool approved = manager.isOracleApprovedOnlyAllowEOA(tx.origin);
        assertEq(approved, true, "Should approve EOA address");

        approved = manager.isOracleApprovedOnlyAllowEOA(address(this));
        assertEq(approved, false, "Should reject contract address");

        mockOracle.setBlacklistedAccount(tx.origin, true);
        approved = manager.isOracleApprovedOnlyAllowEOA(tx.origin);
        assertEq(approved, false, "Should reject blacklisted EOA address");
    }
}


 


contract OracleProtectionManagerOverride is OracleProtectionManager {

    function getHypernativeOracle() public view returns (address) {
        return _hypernativeOracle();
    }

    function setOracle(address _oracle) public {
        return _setOracle(_oracle);
    }

     function setStrictMode(bool _mode) public {
        return _setIsStrictMode(_mode);
    }

}