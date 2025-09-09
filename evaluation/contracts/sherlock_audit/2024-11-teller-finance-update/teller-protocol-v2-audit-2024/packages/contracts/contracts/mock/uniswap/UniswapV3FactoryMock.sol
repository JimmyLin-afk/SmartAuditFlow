contract UniswapV3FactoryMock {
    address poolMock;

    function getPool(address token0, address token1, uint24 fee)
        public
        returns (address)
    {
        return poolMock;
    }

    function setPoolMock(address _pool) public {
        poolMock = _pool;
    }
}
