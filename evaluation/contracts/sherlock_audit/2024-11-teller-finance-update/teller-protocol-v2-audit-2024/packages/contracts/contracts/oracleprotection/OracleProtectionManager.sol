// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;


import {IHypernativeOracle} from "../interfaces/oracleprotection/IHypernativeOracle.sol";

import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
 

import {IOracleProtectionManager} from "../interfaces/oracleprotection/IOracleProtectionManager.sol";
 

abstract contract OracleProtectionManager is 
IOracleProtectionManager, OwnableUpgradeable
{
    bytes32 private constant HYPERNATIVE_ORACLE_STORAGE_SLOT = bytes32(uint256(keccak256("eip1967.hypernative.oracle")) - 1);
    bytes32 private constant HYPERNATIVE_MODE_STORAGE_SLOT = bytes32(uint256(keccak256("eip1967.hypernative.is_strict_mode")) - 1);
    
    event OracleAddressChanged(address indexed previousOracle, address indexed newOracle);
    

    modifier onlyOracleApproved() {
         
        require( isOracleApproved(msg.sender ) , "Oracle: Not Approved");
        _;
    }

    modifier onlyOracleApprovedAllowEOA() {
       
        require( isOracleApprovedAllowEOA(msg.sender ) , "Oracle: Not Approved");
        _;
    }
 

    function oracleRegister(address _account) public virtual {
        address oracleAddress = _hypernativeOracle();
        IHypernativeOracle oracle = IHypernativeOracle(oracleAddress);
        if  (hypernativeOracleIsStrictMode()) {
            oracle.registerStrict(_account);
        }
        else {
            oracle.register(_account);
        }
    } 


     function isOracleApproved(address _sender) public returns (bool) {
        address oracleAddress = _hypernativeOracle();
        if (oracleAddress == address(0)) { 
            return true;
        }
        IHypernativeOracle oracle = IHypernativeOracle(oracleAddress);
        if (oracle.isBlacklistedContext( tx.origin,_sender) || !oracle.isTimeExceeded(_sender)) {
            return false;
        }
        return true;
     }

    // Only allow EOA to interact 
    function isOracleApprovedOnlyAllowEOA(address _sender) public returns (bool){
        address oracleAddress = _hypernativeOracle();
        if (oracleAddress == address(0)) {
             
            return true;
        }

        IHypernativeOracle oracle = IHypernativeOracle(oracleAddress);
        if (oracle.isBlacklistedAccount(_sender) || _sender != tx.origin) {
            return false;
        } 
        return true ;
    }

    // Always allow EOAs to interact, non-EOA have to register
    function isOracleApprovedAllowEOA(address _sender) public returns (bool){
        address oracleAddress = _hypernativeOracle();

        //without an oracle address set, allow all through 
        if (oracleAddress == address(0)) {
            return true;
        }

        // any accounts are blocked if blacklisted
        IHypernativeOracle oracle = IHypernativeOracle(oracleAddress);
        if (oracle.isBlacklistedContext( tx.origin,_sender) ){
            return false;
        }
        
        //smart contracts (delegate calls) are blocked if they havent registered and waited
        if (  _sender != tx.origin  && !oracle.isTimeExceeded(_sender)) {
            return false;
        }

        return true ;
    }
    
    
    /*
    * @dev This must be implemented by the parent contract or else the modifiers will always pass through all traffic

    */
    function _setOracle(address _oracle) internal {
        address oldOracle = _hypernativeOracle();
        _setAddressBySlot(HYPERNATIVE_ORACLE_STORAGE_SLOT, _oracle);
        emit OracleAddressChanged(oldOracle, _oracle);
    }
    

    function _setIsStrictMode(bool _mode) internal {
        _setValueBySlot(HYPERNATIVE_MODE_STORAGE_SLOT, _mode ? 1 : 0);
    }

 

    function _setAddressBySlot(bytes32 slot, address newAddress) internal {
        assembly {
            sstore(slot, newAddress)
        }
    }

    function _setValueBySlot(bytes32 _slot, uint256 _value) internal {
        assembly {
            sstore(_slot, _value)
        }
    }

 
    function hypernativeOracleIsStrictMode() public view returns (bool) {
        return _getValueBySlot(HYPERNATIVE_MODE_STORAGE_SLOT) == 1;
    }

    function _getAddressBySlot(bytes32 slot) internal view returns (address addr) {
        assembly {
            addr := sload(slot)
        }
    }

    function _getValueBySlot(bytes32 _slot) internal view returns (uint256 _value) {
        assembly {
            _value := sload(_slot)
        }
    }

    function _hypernativeOracle() internal view returns (address) {
        return _getAddressBySlot(HYPERNATIVE_ORACLE_STORAGE_SLOT);
    }


}