




// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;


//records the unpause timestamp s
interface IProtocolPausingManager {
    
   function isPauser(address _address) external view returns (bool);
   function protocolPaused() external view returns (bool);
   function liquidationsPaused() external view returns (bool);
   
 

}
