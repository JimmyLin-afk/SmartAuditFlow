// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import {AccessControl} from "@openzeppelin/contracts/access/AccessControl.sol";

contract HypernativeOracle is AccessControl {
    struct OracleRecord {
        uint256 registrationTime;
        bool isPotentialRisk;
    }

    bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");
    bytes32 public constant CONSUMER_ROLE = keccak256("CONSUMER_ROLE");
    uint256 internal threshold = 2 minutes;

    mapping(bytes32 /*hashedAccount*/ => OracleRecord) internal accountHashToRecord;
    
    event ConsumerAdded(address consumer);
    event ConsumerRemoved(address consumer);
    event Registered(address consumer, address account);
    event Whitelisted(bytes32[] hashedAccounts);
    event Allowed(bytes32[] hashedAccounts);
    event Blacklisted(bytes32[] hashedAccounts);
    event TimeThresholdChanged(uint256 threshold);

    modifier onlyAdmin() {
        require(hasRole(DEFAULT_ADMIN_ROLE, msg.sender), "Hypernative Oracle error: admin required");
        _;
    }

    modifier onlyOperator() {
        require(hasRole(OPERATOR_ROLE, msg.sender), "Hypernative Oracle error: operator required");
        _;
    }

    modifier onlyConsumer {
        require(hasRole(CONSUMER_ROLE, msg.sender), "Hypernative Oracle error: consumer required");
        _;
    }

    constructor(address _admin) {
        _grantRole(DEFAULT_ADMIN_ROLE, _admin);
    }

    function register(address _account) external onlyConsumer() {
        bytes32 _hashedAccount = keccak256(abi.encodePacked(_account, address(this)));
        require(accountHashToRecord[_hashedAccount].registrationTime == 0, "Account already registered");
        accountHashToRecord[_hashedAccount].registrationTime = block.timestamp;
        emit Registered(msg.sender, _account);
    }

    function registerStrict(address _account) external onlyConsumer() {
        bytes32 _hashedAccount = keccak256(abi.encodePacked(_account, address(this)));
        require(accountHashToRecord[_hashedAccount].registrationTime == 0, "Account already registered");
        accountHashToRecord[_hashedAccount].registrationTime = block.timestamp;
        accountHashToRecord[_hashedAccount].isPotentialRisk = true;
        emit Registered(msg.sender, _account);
    }

    // **
    // * @dev Operator only function, can be used to whitelist accounts in order to pre-approve them
    // * @param hashedAccounts array of hashed accounts
    // */
    function whitelist(bytes32[] calldata hashedAccounts) public onlyOperator() {
        for (uint256 i; i < hashedAccounts.length; i++) {
            accountHashToRecord[hashedAccounts[i]].registrationTime = 1;
            accountHashToRecord[hashedAccounts[i]].isPotentialRisk = false;
        }
        emit Whitelisted(hashedAccounts);
    }

    // **
    // * @dev Operator only function, can be used to blacklist accounts in order to prevent them from interacting with the protocol
    // * @param hashedAccounts array of hashed accounts
    // */
    function blacklist(bytes32[] calldata hashedAccounts) public onlyOperator() {
        for (uint256 i; i < hashedAccounts.length; i++) {
            accountHashToRecord[hashedAccounts[i]].isPotentialRisk = true;
        }
        emit Blacklisted(hashedAccounts);
    }

    // **
    // * @dev Operator only function, when strict mode is being used, this function can be used to allow accounts to interact with the protocol
    // * @param hashedAccounts array of hashed accounts
    // */
    function allow(bytes32[] calldata hashedAccounts) public onlyOperator() {
        for (uint256 i; i < hashedAccounts.length; i++) {
            accountHashToRecord[hashedAccounts[i]].isPotentialRisk = false;
        }
        emit Allowed(hashedAccounts);
    }

    /**
     * @dev Admin only function, can be used to block any interaction with the protocol, meassured in seconds
     */
    function changeTimeThreshold(uint256 _newThreshold) public onlyAdmin() {
        require(_newThreshold >= 2 minutes, "Threshold must be greater than 2 minutes");
        threshold = _newThreshold;
        emit TimeThresholdChanged(threshold);
    }

    function addConsumers(address[] memory consumers) public onlyAdmin() {
        for (uint256 i; i < consumers.length; i++) {
            _grantRole(CONSUMER_ROLE, consumers[i]);
            emit ConsumerAdded(consumers[i]);
        }
    }

    function revokeConsumers(address[] memory consumers) public onlyAdmin() {
        for (uint256 i; i < consumers.length; i++) {
            _revokeRole(CONSUMER_ROLE, consumers[i]);
            emit ConsumerRemoved(consumers[i]);
        }
    }

    function addOperator(address operator) public onlyAdmin() {
        _grantRole(OPERATOR_ROLE, operator);
    }

    function revokeOperator(address operator) public onlyAdmin() {
        _revokeRole(OPERATOR_ROLE, operator);
    }

    function changeAdmin(address _newAdmin) public onlyAdmin() {
        _grantRole(DEFAULT_ADMIN_ROLE, _newAdmin);
        _revokeRole(DEFAULT_ADMIN_ROLE, msg.sender);
    }

    //if true, the account has been registered for two minutes 
    function isTimeExceeded(address account) external onlyConsumer() view returns (bool) {
        bytes32 hashedAccount = keccak256(abi.encodePacked(account, address(this)));
        require(accountHashToRecord[hashedAccount].registrationTime != 0, "Account not registered");
        return block.timestamp - accountHashToRecord[hashedAccount].registrationTime > threshold;
    }

    function isBlacklistedContext(address _origin, address _sender) external onlyConsumer() view returns (bool) {
        bytes32 hashedOrigin = keccak256(abi.encodePacked(_origin, address(this)));
        bytes32 hashedSender = keccak256(abi.encodePacked(_sender, address(this)));
        return accountHashToRecord[hashedOrigin].isPotentialRisk || accountHashToRecord[hashedSender].isPotentialRisk;
    }

    function isBlacklistedAccount(address _account) external onlyConsumer() view returns (bool) {
        bytes32 hashedAccount = keccak256(abi.encodePacked(_account, address(this)));
        return accountHashToRecord[hashedAccount].isPotentialRisk;
    }
}
