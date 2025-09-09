## Tests 

For running the tests run the following command:

`forge test --fork-url https://mainnet.base.org --fork-block-number 21151256 --no-match-path '**Fantom**'` 

### With Anvil 


test/local --> forge test test/local/... (on localhost)
test/fork/... --> forge test test/fork/... (anvil --fork-url https://mainnet.base.org --fork-block-number 21151256)

Specific tests
test/fork/Loan/ltv/Tarot-Fantom/.. --> anvil --fork-url https://rpc.ftm.tools (Fantom env.)
