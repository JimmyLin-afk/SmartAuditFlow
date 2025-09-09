pragma solidity ^0.8.0;

interface ITarotPriceOracle {
    function MIN_T() external pure returns (uint32);

    function getPair(
        address uniswapV2Pair
    )
        external
        view
        returns (
            uint256 priceCumulativeSlotA,
            uint256 priceCumulativeSlotB,
            uint32 lastUpdateSlotA,
            uint32 lastUpdateSlotB,
            bool latestIsSlotA,
            bool initialized
        );

    function initialize(address uniswapV2Pair) external;

    function getResult(
        address uniswapV2Pair
    ) external returns (uint224 price, uint32 T);

    function getBlockTimestamp() external view returns (uint32);
}
