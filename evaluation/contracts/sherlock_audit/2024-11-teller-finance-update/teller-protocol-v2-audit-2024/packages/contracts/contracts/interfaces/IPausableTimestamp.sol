




// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;


//records the unpause timestamp s
interface IPausableTimestamp {
    
    
    function getLastUnpausedAt() 
    external view 
    returns (uint256)  ;

   // function setLastUnpausedAt() internal;



}
