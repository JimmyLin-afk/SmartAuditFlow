// Built off of https://github.com/DeltaBalances/DeltaBalances.github.io/blob/master/smart_contract/deltabalances.sol
pragma solidity ^0.8.23;

// ERC20 contract interface
interface Token {
    function balanceOf(address) external view returns (uint);
}

interface IUniswapV3PoolState {
    function slot0()
        external
        view
        returns (
            uint160 sqrtPriceX96,
            int24 tick,
            uint16 observationIndex,
            uint16 observationCardinality,
            uint16 observationCardinalityNext,
            uint8 feeProtocol,
            bool unlocked
        );
}

interface IUniswapV2Pair {
    function getReserves() external view returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast);
}

contract MultiReader {
    /* Fallback function, don't accept any ETH */
    fallback() external {
        revert("MultiReader does not accept payments");
    }
    receive() external payable {
        revert("MultiReader does not accept payments");
    }

    /*
        Check the token balance of a wallet in a token contract

        Returns the balance of the token for user. Avoids possible errors:
        - return 0 on non-contract address 
        - returns 0 if the contract doesn't implement balanceOf
    */
    function tokenBalance(address user, address token) public view returns (uint) {
        return Token(token).balanceOf(user);
    }

    /*
        Check the token balances of a wallet for multiple tokens.
        Pass 0x0 as a "token" address to get ETH balance.

        Possible error throws:
        - extremely large arrays for user and or tokens (gas cost too high) 
            
        Returns a one-dimensional that's user.length * tokens.length long. The
        array is ordered by all of the 0th users token balances, then the 1th
        user, and so on.
    */
    function balances(address[] calldata users, address[] calldata tokens) external view returns (uint[] memory) {
        uint[] memory addrBalances = new uint[](tokens.length * users.length);
        
        for(uint i = 0; i < users.length; i++) {
            for (uint j = 0; j < tokens.length; j++) {
                uint addrIdx = j + tokens.length * i;
                if (tokens[j] != address(0)) { 
                    addrBalances[addrIdx] = tokenBalance(users[i], tokens[j]);
                } else {
                    addrBalances[addrIdx] = users[i].balance; // ETH balance    
                }
            }  
        }
        return addrBalances;
    }

    /*
        Read the sqrtPriceX96 of multiple Uniswap V3 pools
        Returns an array of sqrtPriceX96 values for each pool
    */
    function readSqrtPriceX96(address[] calldata pools) external view returns (uint160[] memory) {
        uint160[] memory sqrtPrices = new uint160[](pools.length);
        for (uint i = 0; i < pools.length; i++) {
            (uint160 sqrtPriceX96, , , , , , ) = IUniswapV3PoolState(pools[i]).slot0();
            sqrtPrices[i] = sqrtPriceX96;
        }
        return sqrtPrices;
    }
    
    /*
        Read the reserves of multiple Uniswap V2 pairs
        Returns an array of reserves in [reserve0A, reserve1A, reserve0B, reserve1B, ...]
    */
    function readReserves(address[] calldata pairs) external view returns (uint112[] memory) {
        uint112[] memory reserves = new uint112[](pairs.length * 2);
        for (uint i = 0; i < pairs.length; i++) {
            (uint112 reserve0, uint112 reserve1, ) = IUniswapV2Pair(pairs[i]).getReserves();
            reserves[i * 2] = reserve0;
            reserves[i * 2 + 1] = reserve1;
        }
        return reserves;
    }
}