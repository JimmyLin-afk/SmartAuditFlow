// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.7.0) (security/Pausable.sol)

pragma solidity ^0.8.0;



import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";

import "@openzeppelin/contracts-upgradeable/utils/ContextUpgradeable.sol";

import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";



import "@openzeppelin/contracts-upgradeable/utils/math/MathUpgradeable.sol";



import "../interfaces/IProtocolPausingManager.sol";


import "../interfaces/IPausableTimestamp.sol";

/**
 
 TODO:

 Move SCF pausing into here ??


 */
contract ProtocolPausingManager is ContextUpgradeable, OwnableUpgradeable, IProtocolPausingManager , IPausableTimestamp{
    using MathUpgradeable for uint256;

    bool private _protocolPaused; 
    bool private _liquidationsPaused; 
    //bool private _liquidityPoolsPaused;    


    mapping(address => bool) public  pauserRoleBearer;


    uint256 private lastPausedAt;
    uint256 private lastUnpausedAt;


    // Events
    event PausedProtocol(address indexed account);
    event UnpausedProtocol(address indexed account);
    event PausedLiquidations(address indexed account);
    event UnpausedLiquidations(address indexed account);
    event PauserAdded(address indexed account);
    event PauserRemoved(address indexed account);
    
 
    modifier onlyPauser() {
        require( isPauser( _msgSender()) );
        _;
    }


    //need to initialize so owner is owner (transfer ownership to safe)

    function initialize(
        
    ) external initializer {

        __Ownable_init();

    }
 

    /**
     * @dev Returns true if the contract is paused, and false otherwise.
     */
    function protocolPaused() public view virtual returns (bool) {
        return _protocolPaused;
    }


      function liquidationsPaused() public view virtual returns (bool) {
        return _liquidationsPaused;
    }

   


    function pauseProtocol() public virtual onlyPauser {
        require( _protocolPaused == false);
          _protocolPaused = true;
          lastPausedAt = block.timestamp;
        emit PausedProtocol(_msgSender());
    }

     
    function unpauseProtocol() public virtual onlyPauser {
       
        require( _protocolPaused == true);
        _protocolPaused = false;
       lastUnpausedAt = block.timestamp;
        emit UnpausedProtocol(_msgSender());
    }

    function pauseLiquidations() public virtual onlyPauser  {
         
         require( _liquidationsPaused == false);
          _liquidationsPaused = true;
          lastPausedAt = block.timestamp;
        emit PausedLiquidations(_msgSender());
    }

     
    function unpauseLiquidations() public virtual onlyPauser  {
        require( _liquidationsPaused == true);
        _liquidationsPaused = false;
       lastUnpausedAt = block.timestamp;
        emit UnpausedLiquidations(_msgSender());
    }

    function getLastPausedAt() 
    external view 
    returns (uint256) {

        return lastPausedAt;

    } 


    function getLastUnpausedAt() 
    external view 
    returns (uint256) {

        return lastUnpausedAt;

    } 




    // Role Management 


    function addPauser(address _pauser) public virtual onlyOwner   {
       pauserRoleBearer[_pauser] = true;
       emit PauserAdded(_pauser);
    }


    function removePauser(address _pauser) public virtual onlyOwner {
        pauserRoleBearer[_pauser] = false;
        emit PauserRemoved(_pauser);
    }


    function isPauser(address _account) public view returns(bool){
        return pauserRoleBearer[_account] || _account == owner();
    }

  
}
