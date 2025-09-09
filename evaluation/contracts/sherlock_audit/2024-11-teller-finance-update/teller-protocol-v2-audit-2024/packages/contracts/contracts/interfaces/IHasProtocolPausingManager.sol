




// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;


//records the unpause timestamp s
interface IHasProtocolPausingManager {
    
     
   function getProtocolPausingManager() external view returns (address); 

   // function isPauser(address _address) external view returns (bool); 


}
