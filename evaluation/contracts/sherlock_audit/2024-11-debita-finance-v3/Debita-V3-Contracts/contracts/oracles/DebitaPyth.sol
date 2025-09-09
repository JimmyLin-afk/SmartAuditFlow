// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import "@pythnetwork/pyth-sdk-solidity/IPyth.sol";

contract DebitaPyth {
    mapping(address => bytes32) public priceIdPerToken;
    mapping(bytes32 => bool) public isFeedAvailable;

    // Managers will be able to pause the contract or specific price feeds but not set them
    mapping(address => bool) public isSenderAManager;

    // MultiSig will be able to set price feeds, change multisig, change status of the contract and change status of price feeds
    address public multiSig;
    bool public isPaused;
    IPyth pyth;

    // _pyth contract address and _multisig address
    constructor(address _pyth, address _multisig) {
        pyth = IPyth(_pyth);
        isSenderAManager[msg.sender] = true;
        multiSig = _multisig;
    }

    function getThePrice(address tokenAddress) public view returns (int) {
        // falta hacer un chequeo para las l2
        bytes32 _priceFeed = priceIdPerToken[tokenAddress];
        require(_priceFeed != bytes32(0), "Price feed not set");
        require(!isPaused, "Contract is paused");

        // Get the price from the pyth contract, no older than 90 seconds
        PythStructs.Price memory priceData = pyth.getPriceNoOlderThan(
            _priceFeed,
            600
        );

        // Check if the price feed is available and the price is valid
        require(isFeedAvailable[_priceFeed], "Price feed not available");
        require(priceData.price > 0, "Invalid price");
        return priceData.price;
    }

    /* 
    
    Managers can pause the contract or specific price feeds but not set them or reactivate them (That's for the multisig)
    
    */
    // only the multisig can set the price feeds. The price feeds can only be set once
    function setPriceFeeds(address tokenAddress, bytes32 priceId) public {
        require(msg.sender == multiSig, "Only multiSig can set price feeds");
        require(
            priceIdPerToken[tokenAddress] == bytes32(0),
            "Price feed already set"
        );
        isFeedAvailable[priceId] = true;
        priceIdPerToken[tokenAddress] = priceId;
    }

    function pauseContract() public {
        require(
            isSenderAManager[msg.sender],
            "Only manager can pause contract"
        );
        isPaused = true;
    }
    function reactivateContract() public {
        require(msg.sender == multiSig, "Only multiSig can change status");
        isPaused = false;
    }

    function pauseStatusPriceId(bytes32 priceId) public {
        require(isSenderAManager[msg.sender], "Only manager can change status");
        isFeedAvailable[priceId] = false;
    }

    function reactivateStatusPriceId(bytes32 priceId) public {
        require(msg.sender == multiSig, "Only multiSig can change status");
        isFeedAvailable[priceId] = true;
    }

    function changeMultisig(address newMultisig) public {
        require(msg.sender == multiSig, "Only multiSig can change multisig");
        multiSig = newMultisig;
    }

    function getDecimals(address token) public view returns (uint) {
        bytes32 _priceFeed = priceIdPerToken[token];

        PythStructs.Price memory priceData = pyth.getPriceNoOlderThan(
            _priceFeed,
            1800
        );
        uint decimals = uint(int(priceData.expo) * -1);
        return decimals;
    }

    function changeManager(address newManager, bool available) public {
        require(msg.sender == multiSig, "Only multiSig can change manager");
        isSenderAManager[newManager] = available;
    }
}
