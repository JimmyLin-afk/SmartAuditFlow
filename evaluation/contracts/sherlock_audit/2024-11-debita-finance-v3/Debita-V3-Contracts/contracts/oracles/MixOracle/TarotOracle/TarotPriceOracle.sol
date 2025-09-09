pragma solidity ^0.8.0;

import "@contracts/oracles/MixOracle/TarotOracle/libraries/UQ112x112.sol";
import "@contracts/oracles/MixOracle/TarotOracle/interfaces/IUniswapV2Pair.sol";
import "@contracts/oracles/MixOracle/TarotOracle/interfaces/ITarotPriceOracle.sol";
import {console} from "forge-std/console.sol";
contract TarotPriceOracle is ITarotPriceOracle {
    using UQ112x112 for uint224;

    uint32 public constant MIN_T = 1200;

    struct Pair {
        uint256 priceCumulativeSlotA;
        uint256 priceCumulativeSlotB;
        uint32 lastUpdateSlotA;
        uint32 lastUpdateSlotB;
        bool latestIsSlotA;
        bool initialized;
    }
    mapping(address => Pair) public getPair;

    event PriceUpdate(
        address indexed pair,
        uint256 priceCumulative,
        uint32 blockTimestamp,
        bool latestIsSlotA
    );

    function toUint224(uint256 input) internal pure returns (uint224) {
        return uint224(input);
    }

    function getPriceCumulativeCurrent(
        address uniswapV2Pair
    ) internal view returns (uint256 priceCumulative) {
        priceCumulative = IUniswapV2Pair(uniswapV2Pair)
            .reserve0CumulativeLast();
        (
            uint112 reserve0,
            uint112 reserve1,
            uint32 _blockTimestampLast
        ) = IUniswapV2Pair(uniswapV2Pair).getReserves();
        uint224 priceLatest = UQ112x112.encode(reserve1).uqdiv(reserve0);
        uint32 timeElapsed = getBlockTimestamp() - _blockTimestampLast; // overflow is desired
        // * never overflows, and + overflow is desired
        priceCumulative += (uint256(priceLatest) * timeElapsed);
    }

    function initialize(address uniswapV2Pair) external {
        Pair storage pairStorage = getPair[uniswapV2Pair];
        require(
            !pairStorage.initialized,
            "TarotPriceOracle: ALREADY_INITIALIZED"
        );

        uint256 priceCumulativeCurrent = getPriceCumulativeCurrent(
            uniswapV2Pair
        );
        uint32 blockTimestamp = getBlockTimestamp();
        pairStorage.priceCumulativeSlotA = priceCumulativeCurrent;
        pairStorage.priceCumulativeSlotB = priceCumulativeCurrent;
        pairStorage.lastUpdateSlotA = blockTimestamp;
        pairStorage.lastUpdateSlotB = blockTimestamp;
        pairStorage.latestIsSlotA = true;
        pairStorage.initialized = true;
        emit PriceUpdate(
            uniswapV2Pair,
            priceCumulativeCurrent,
            blockTimestamp,
            true
        );
    }

    function getResult(
        address uniswapV2Pair
    ) external returns (uint224 price, uint32 T) {
        Pair memory pair = getPair[uniswapV2Pair];
        require(pair.initialized, "TarotPriceOracle: NOT_INITIALIZED");
        Pair storage pairStorage = getPair[uniswapV2Pair];

        uint32 blockTimestamp = getBlockTimestamp();
        uint32 lastUpdateTimestamp = pair.latestIsSlotA
            ? pair.lastUpdateSlotA
            : pair.lastUpdateSlotB;
        uint256 priceCumulativeCurrent = getPriceCumulativeCurrent(
            uniswapV2Pair
        );
        uint256 priceCumulativeLast;

        if (blockTimestamp - lastUpdateTimestamp >= MIN_T) {
            // update price
            priceCumulativeLast = pair.latestIsSlotA
                ? pair.priceCumulativeSlotA
                : pair.priceCumulativeSlotB;
            if (pair.latestIsSlotA) {
                pairStorage.priceCumulativeSlotB = priceCumulativeCurrent;
                pairStorage.lastUpdateSlotB = blockTimestamp;
            } else {
                pairStorage.priceCumulativeSlotA = priceCumulativeCurrent;
                pairStorage.lastUpdateSlotA = blockTimestamp;
            }
            pairStorage.latestIsSlotA = !pair.latestIsSlotA;
            emit PriceUpdate(
                uniswapV2Pair,
                priceCumulativeCurrent,
                blockTimestamp,
                !pair.latestIsSlotA
            );
        } else {
            // don't update; return price using previous priceCumulative
            lastUpdateTimestamp = pair.latestIsSlotA
                ? pair.lastUpdateSlotB
                : pair.lastUpdateSlotA;
            priceCumulativeLast = pair.latestIsSlotA
                ? pair.priceCumulativeSlotB
                : pair.priceCumulativeSlotA;
        }

        T = blockTimestamp - lastUpdateTimestamp; // overflow is desired
        require(T >= MIN_T, "TarotPriceOracle: NOT_READY"); //reverts only if the pair has just been initialized
        // / is safe, and - overflow is desired
        price = toUint224((priceCumulativeCurrent - priceCumulativeLast) / T);
    }

    /*** Utilities ***/

    function getBlockTimestamp() public view returns (uint32) {
        return uint32(block.timestamp % 2 ** 32);
    }
}
