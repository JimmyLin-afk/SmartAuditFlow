// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import {IOracleProtectionManager} from "../interfaces/oracleprotection/IOracleProtectionManager.sol";

abstract contract OracleProtectedChild {
    address public immutable ORACLE_MANAGER;
   

    modifier onlyOracleApproved() {
        
        IOracleProtectionManager oracleManager = IOracleProtectionManager(ORACLE_MANAGER);
        require( oracleManager .isOracleApproved(msg.sender ) , "Oracle: Not Approved");
        _;
    }

    modifier onlyOracleApprovedAllowEOA() {
      
        IOracleProtectionManager oracleManager = IOracleProtectionManager(ORACLE_MANAGER);
        require( oracleManager .isOracleApprovedAllowEOA(msg.sender ) , "Oracle: Not Approved");
        _;
    }
    
 constructor(address oracleManager){
		 ORACLE_MANAGER = oracleManager;
 }
    
}