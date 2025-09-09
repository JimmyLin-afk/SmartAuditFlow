##### Gas Comparison

`/test/gas/BrokerGas.t.sol` contains simple swap tests to estimate gas usage. You can get an overview with

`forge test --match-contract BrokerGasTest` and
`forge test --match-contract BrokerGasTest --gas-report`

or a more detailed view of gas usage by function call with

`forge test --match-contract BrokerGasTest -vvvv` or for single tests
`forge test --match-test ${TEST_NAME} -vvvvv`.

##### How to compare gas usage

To compare feature additions and code changes, you can run `/test/gas/BrokerGas.t.sol` on different branches and compare the gas estimates manually.

##### Historical Exchange Gas usage

Mento V1 contracts take around 25% more gas for simple swaps from a Stable to CELO. You can run the gas estimation by
`forge test --match-contract ExchangeGasTest -vvvv` or for single tests
`forge test --match-test ${TEST_NAME} -vvvvv`.

###### 2022/11/21 gas report

Broker.sol
[PASS] test_gas_swapIn_CELOTocEUR() (gas: 237016)
[PASS] test_gas_swapIn_CELOTocUSD() (gas: 237014)
[PASS] test_gas_swapIn_CEURToCelo() (gas: 292394)
[PASS] test_gas_swapIn_CUSDToCelo() (gas: 292395)
[PASS] test_gas_swapIn_cEURToBridgedUSDC() (gas: 309449)
[PASS] test_gas_swapIn_cEURTocUSD() (gas: 269525)
[PASS] test_gas_swapIn_cUSDToBridgedUSDC() (gas: 309470)
[PASS] test_gas_swapIn_cUSDTocEUR() (gas: 269442)

Exchange.sol
[PASS] test_gas_sell_CELO_for_cEUR() (gas: 207711)
[PASS] test_gas_sell_CELO_for_cUSD() (gas: 207752)
[PASS] test_gas_sell_cEUR_for_CELO() (gas: 257494)
[PASS] test_gas_sell_cUSD_for_CELO() (gas: 261958)
