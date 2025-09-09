pragma solidity ^0.8.0;

import {AggregatorV3Interface} from "@chainlink/src/interfaces/feeds/AggregatorV3Interface.sol";
import {AggregatorV2V3Interface} from "@chainlink/src/interfaces/feeds/AggregatorV2V3Interface.sol";
import {console} from "forge-std/console.sol";

contract DebitaChainlink {
    error SequencerDown();
    error GracePeriodNotOver();

    // token => price feed
    mapping(address => address) public priceFeeds;
    // Managers will be able to pause the contract or specific price feeds but not set them
    mapping(address => bool) public isSenderAManager;

    // is feed available?
    mapping(address => bool) public isFeedAvailable;
    address public multiSig;
    bool public isPaused;
    uint constant GRACE_PERIOD_TIME = 1 hours;
    AggregatorV2V3Interface public sequencerUptimeFeed;

    // In case of deploying this contract on a layer 2, sequencer address must be set. Otherwhise just set address(0x0)
    constructor(address _sequencer, address _multisig) {
        isSenderAManager[msg.sender] = true;
        multiSig = _multisig;
        sequencerUptimeFeed = AggregatorV2V3Interface(_sequencer);
    }

    function getThePrice(address tokenAddress) public view returns (int) {
        // falta hacer un chequeo para las l2
        address _priceFeed = priceFeeds[tokenAddress];
        require(!isPaused, "Contract is paused");
        require(_priceFeed != address(0), "Price feed not set");
        AggregatorV3Interface priceFeed = AggregatorV3Interface(_priceFeed);

        // if sequencer is set, check if it's up
        // if it's down, revert
        if (address(sequencerUptimeFeed) != address(0)) {
            checkSequencer();
        }
        (, int price, , , ) = priceFeed.latestRoundData();

        require(isFeedAvailable[_priceFeed], "Price feed not available");
        require(price > 0, "Invalid price");
        return price;
    }

    function checkSequencer() public view returns (bool) {
        (, int256 answer, uint256 startedAt, , ) = sequencerUptimeFeed
            .latestRoundData();

        // Answer == 0: Sequencer is up
        // Answer == 1: Sequencer is down
        bool isSequencerUp = answer == 0;
        if (!isSequencerUp) {
            revert SequencerDown();
        }
        console.logUint(startedAt);
        // Make sure the grace period has passed after the
        // sequencer is back up.
        uint256 timeSinceUp = block.timestamp - startedAt;
        if (timeSinceUp <= GRACE_PERIOD_TIME) {
            revert GracePeriodNotOver();
        }

        return true;
    }

    // only multisig can set the price feeds. The price feeds can only be set once
    function setPriceFeeds(address _token, address _priceFeed) public {
        require(msg.sender == multiSig, "Only manager can set price feeds");
        // only declare it once
        require(priceFeeds[_token] == address(0), "Price feed already set");
        priceFeeds[_token] = _priceFeed;
        isFeedAvailable[_priceFeed] = true;
    }

    function getDecimals(address token) public returns (uint) {
        address _priceFeed = priceFeeds[token];
        AggregatorV3Interface priceFeed = AggregatorV3Interface(_priceFeed);
        return priceFeed.decimals();
    }

    /* 
    
    Managers can pause the contract or specific price feeds but not set them or reactivate them (That's for the multisig)
    
    */
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

    function pauseStatuspriceFeed(address priceFeed) public {
        require(isSenderAManager[msg.sender], "Only manager can change status");
        isFeedAvailable[priceFeed] = false;
    }

    function reactivateStatuspriceFeed(address priceFeed) public {
        require(msg.sender == multiSig, "Only multiSig can change status");
        isFeedAvailable[priceFeed] = true;
    }

    // multiple managers can be added, multisig can decide who can pause the contract
    function changeManager(address newManager, bool available) public {
        require(msg.sender == multiSig, "Only multiSig can change manager");
        isSenderAManager[newManager] = available;
    }

    function changeMultisig(address newMultisig) public {
        require(msg.sender == multiSig, "Only multiSig can change multisig");
        multiSig = newMultisig;
    }

    /* function to stop an oracle or activate it again */
}
