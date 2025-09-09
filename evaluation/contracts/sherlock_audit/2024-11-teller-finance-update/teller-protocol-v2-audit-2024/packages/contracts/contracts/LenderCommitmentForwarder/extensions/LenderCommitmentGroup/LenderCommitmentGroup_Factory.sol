// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Contracts
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol"; 

import "@openzeppelin/contracts-upgradeable/utils/AddressUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/token/ERC20/IERC20Upgradeable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

// Interfaces
import "../../../interfaces/ITellerV2.sol";
import "../../../interfaces/IProtocolFee.sol";
import "../../../interfaces/ITellerV2Storage.sol";
import "../../../libraries/NumbersLib.sol";

import {IUniswapPricingLibrary} from "../../../interfaces/IUniswapPricingLibrary.sol";

import "@openzeppelin/contracts/proxy/beacon/BeaconProxy.sol";
 
import { ILenderCommitmentGroup } from "../../../interfaces/ILenderCommitmentGroup.sol";

contract LenderCommitmentGroupFactory is OwnableUpgradeable {
    using AddressUpgradeable for address;
    using NumbersLib for uint256;

    
    //this is the beacon proxy
    address public lenderGroupBeacon;


    mapping(address => uint256) public deployedLenderGroupContracts;

    event DeployedLenderGroupContract(address indexed groupContract);

 

      /**
     * @notice Initializes the factory contract.
     * @param _lenderGroupBeacon The address of the beacon proxy used for deploying group contracts.
     */
     function initialize(address _lenderGroupBeacon )
        external
        initializer
    {
        lenderGroupBeacon = _lenderGroupBeacon; 
        __Ownable_init_unchained();
    }


 /**
     * @notice Deploys a new lender commitment group pool contract.
     * @dev The function initializes the deployed contract and optionally adds an initial principal amount.
     * @param _initialPrincipalAmount The initial principal amount to be deposited into the group contract.
     * @param _commitmentGroupConfig Configuration parameters for the lender commitment group.
     * @param _poolOracleRoutes Array of pool route configurations for the Uniswap pricing library.
     * @return newGroupContract_ Address of the newly deployed group contract.
     */
    function deployLenderCommitmentGroupPool(
        uint256 _initialPrincipalAmount,
        ILenderCommitmentGroup.CommitmentGroupConfig calldata _commitmentGroupConfig,
        IUniswapPricingLibrary.PoolRouteConfig[] calldata _poolOracleRoutes
    ) external returns ( address ) {
         

      
        BeaconProxy newGroupContract_ = new BeaconProxy(
                lenderGroupBeacon,
                abi.encodeWithSelector(
                    ILenderCommitmentGroup.initialize.selector,    //this initializes 
                    _commitmentGroupConfig,
                    _poolOracleRoutes

                )
            );

        deployedLenderGroupContracts[address(newGroupContract_)] = block.number; //consider changing this ?
        emit DeployedLenderGroupContract(address(newGroupContract_));



        //it is not absolutely necessary to have this call here but it allows the user to potentially save a tx step so it is nice to have .
         if (_initialPrincipalAmount > 0) {
                _addPrincipalToCommitmentGroup(
                address(newGroupContract_),
                _initialPrincipalAmount,
                _commitmentGroupConfig.principalTokenAddress 
                
            ); 
        } 


          //transfer ownership to msg.sender 
        OwnableUpgradeable(address(newGroupContract_))
            .transferOwnership(msg.sender);

        return address(newGroupContract_) ;
    }


    /**
     * @notice Adds principal tokens to a commitment group.
     * @param _newGroupContract The address of the group contract to add principal tokens to.
     * @param _initialPrincipalAmount The amount of principal tokens to add.
     * @param _principalTokenAddress The address of the principal token contract.
     */
    function _addPrincipalToCommitmentGroup(
        address _newGroupContract,
        uint256 _initialPrincipalAmount,
        address _principalTokenAddress
    ) internal returns (uint256) {


            IERC20(_principalTokenAddress).transferFrom(
                msg.sender,
                address(this),
                _initialPrincipalAmount
            );
            IERC20(_principalTokenAddress).approve(
                _newGroupContract,
                _initialPrincipalAmount
            );

            address sharesRecipient = msg.sender; 

            uint256 sharesAmount_ = ILenderCommitmentGroup(address(_newGroupContract))
                .addPrincipalToCommitmentGroup(
                    _initialPrincipalAmount,
                    sharesRecipient,
                    0 //_minShares
                );

        return sharesAmount_;
    }

}
