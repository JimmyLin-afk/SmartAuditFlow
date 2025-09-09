pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@contracts/oracles/MixOracle/TarotOracle/interfaces/IUniswapV2Pair.sol";
import "@contracts/DebitaProxyContract.sol";
import {console} from "forge-std/console.sol";

interface IPyth {
    function getThePrice(address tokenAddress) external returns (int);
}

interface ITarotOracle {
    function initialize(address uniswapV2Pair) external;
    function getResult(
        address uniswapV2Pair
    ) external returns (uint224 price, uint32 T);
}

contract MixOracle {
    address multisig; //
    address tarotOracleImplementation; // TarotOracle implementation
    address debitaPythOracle; // DebitaPythOracle
    bool isPaused;
    uint deployedAt;

    mapping(address => bool) public isSenderAManager; // Managers will be able to pause the contract or specific price feeds but not set them
    mapping(address => address) public AttachedTarotOracle; // address of the token ==> address of the TarotOracle
    mapping(address => address) public AttachedUniswapPair; // address of the token ==> address of the UniswapPair
    mapping(address => address) public AttachedPricedToken; // address of the token you want to know the price ==> address of the token that the price will be returned
    mapping(address => bool) public isFeedAvailable; // address of the UniswapPair ==> is the feed available?

    constructor(address _tarotOracleImplementation, address _debitaPythOracle) {
        multisig = msg.sender;
        isSenderAManager[msg.sender] = true;
        tarotOracleImplementation = _tarotOracleImplementation;
        debitaPythOracle = _debitaPythOracle;
        deployedAt = block.timestamp;
    }

    function getThePrice(address tokenAddress) public returns (int) {
        // get tarotOracle address
        address _priceFeed = AttachedTarotOracle[tokenAddress];
        require(_priceFeed != address(0), "Price feed not set");
        require(!isPaused, "Contract is paused");
        ITarotOracle priceFeed = ITarotOracle(_priceFeed);

        address uniswapPair = AttachedUniswapPair[tokenAddress];
        require(isFeedAvailable[uniswapPair], "Price feed not available");
        // get twap price from token1 in token0
        (uint224 twapPrice112x112, ) = priceFeed.getResult(uniswapPair);
        address attached = AttachedPricedToken[tokenAddress];

        // Get the price from the pyth contract, no older than 20 minutes
        // get usd price of token0
        int attachedTokenPrice = IPyth(debitaPythOracle).getThePrice(attached);
        uint decimalsToken1 = ERC20(attached).decimals();
        uint decimalsToken0 = ERC20(tokenAddress).decimals();

        // calculate the amount of attached token that is needed to get 1 token1
        int amountOfAttached = int(
            (((2 ** 112)) * (10 ** decimalsToken1)) / twapPrice112x112
        );

        // calculate the price of 1 token1 in usd based on the attached token
        uint price = (uint(amountOfAttached) * uint(attachedTokenPrice)) /
            (10 ** decimalsToken1);

        require(price > 0, "Invalid price");
        return int(uint(price));
    }

    function setAttachedTarotPriceOracle(address uniswapV2Pair) public {
        require(multisig == msg.sender, "Only multisig can set price feeds");

        require(
            AttachedUniswapPair[uniswapV2Pair] == address(0),
            "Uniswap pair already set"
        );

        address token0 = IUniswapV2Pair(uniswapV2Pair).token0();
        address token1 = IUniswapV2Pair(uniswapV2Pair).token1();
        require(
            AttachedTarotOracle[token1] == address(0),
            "Price feed already set"
        );
        DebitaProxyContract tarotOracle = new DebitaProxyContract(
            tarotOracleImplementation
        );
        ITarotOracle oracle = ITarotOracle(address(tarotOracle));
        oracle.initialize(uniswapV2Pair);
        AttachedUniswapPair[token1] = uniswapV2Pair;
        AttachedTarotOracle[token1] = address(tarotOracle);
        AttachedPricedToken[token1] = token0;
        isFeedAvailable[uniswapV2Pair] = true;
    }

    function setManager(address manager, bool status) public {
        require(multisig == msg.sender, "Only multisig can set manager");
        isSenderAManager[manager] = status;
    }

    function changeMultisig(address _newMultisig) public {
        require(multisig == msg.sender, "Only multisig can change multisig");
        // only in the first 6 hours after deployment
        require(block.timestamp < deployedAt + 6 hours, "Time passed");
        multisig = _newMultisig;
    }

    function pauseContract() public {
        require(
            isSenderAManager[msg.sender],
            "Only manager can pause contract"
        );
        isPaused = true;
    }

    function reactivateContract() public {
        require(multisig == msg.sender, "Only multisig can change status");
        isPaused = false;
    }

    function pauseStatusPriceId(address uniswapPair) public {
        require(isSenderAManager[msg.sender], "Only manager can change status");
        isFeedAvailable[uniswapPair] = false;
    }

    function reactivateStatusPriceId(address uniswapPair) public {
        require(multisig == msg.sender, "Only multisig can change status");
        isFeedAvailable[uniswapPair] = true;
    }
}
