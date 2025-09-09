
contract UniswapV3PoolMock {
    //this represents an equal price ratio
    uint160 mockSqrtPriceX96 = 2 ** 96;

    address mockToken0;
    address mockToken1;


    

    struct Slot0 {
        // the current price
        uint160 sqrtPriceX96;
        // the current tick
        int24 tick;
        // the most-recently updated index of the observations array
        uint16 observationIndex;
        // the current maximum number of observations that are being stored
        uint16 observationCardinality;
        // the next maximum number of observations to store, triggered in observations.write
        uint16 observationCardinalityNext;
        // the current protocol fee as a percentage of the swap fee taken on withdrawal
        // represented as an integer denominator (1/x)%
        uint8 feeProtocol;
        // whether the pool is locked
        bool unlocked;
    }

    function set_mockSqrtPriceX96(uint160 _price) public {
        mockSqrtPriceX96 = _price;
    }

    function set_mockToken0(address t0) public {
        mockToken0 = t0;
    }


    function set_mockToken1(address t1) public {
        mockToken1 = t1;
    }

    function token0() public returns (address) {
        return mockToken0;
    }

    function token1() public returns (address) {
        return mockToken1;
    }


    function slot0() public returns (Slot0 memory slot0) {
        return
            Slot0({
                sqrtPriceX96: mockSqrtPriceX96,
                tick: 0,
                observationIndex: 0,
                observationCardinality: 0,
                observationCardinalityNext: 0,
                feeProtocol: 0,
                unlocked: true
            });
    }

    //mock fn 
   function observe(uint32[] calldata secondsAgos)
        external
        view
        returns (int56[] memory tickCumulatives, uint160[] memory secondsPerLiquidityCumulativeX128s)
    {
        // Initialize the return arrays
        tickCumulatives = new int56[](secondsAgos.length);
        secondsPerLiquidityCumulativeX128s = new uint160[](secondsAgos.length);

        // Mock data generation - replace this with your logic or static values
        for (uint256 i = 0; i < secondsAgos.length; i++) {
            // Generate mock data. Here we're just using simple static values for demonstration.
            // You should replace these with dynamic values based on your testing needs.
            tickCumulatives[i] = int56(1000 * int256(i)); // Example mock data
            secondsPerLiquidityCumulativeX128s[i] = uint160(2000 * i); // Example mock data
        }

        return (tickCumulatives, secondsPerLiquidityCumulativeX128s);
    }
 
      

}