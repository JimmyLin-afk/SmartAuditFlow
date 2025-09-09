## Mento Protocol Fork Tests Suite

### Structure

```text
                  +-------------+
                  |    Test     |
                  +-------------+
                         ^
                         |
                  +--------------+
                  | BaseForkTest |
                  +--------------+
                     ^       ^
                    /         \
         +-----------------+   +--------------------+
         |  ChainForkTest  |   |  ExchangeForkTest  |
         +-----------------+   +--------------------+
                 ^                       ^
                 |                       |
    +-------------------------+  +-------------------------+
    | Alfajores_ChainForkTest |  | Alfajores_P0E00_...     |
    +-------------------------+  +-------------------------+
    +--------------------+       +-------------------------+
    | Celo_ChainForkTest |       | Alfajores_P0E01_...     |
    +--------------------+       +-------------------------+
                                 +-------------------------+
                                 | Celo_P0E00_...          |
                                 +-------------------------+
```

> TIL Claude knows how to draw ascii art.

Base Contracts:

- `BaseForkTest` implements fork-related shared setup logic.
- `ChainForkTest` tests for a given chain.
- `ExchangeForkTest` tests for a given exchange of an exchange provider on a given chain.

These contracts are abstract and need to be extended by instance specific contracts which specify the target chain and exchange.
This happens in `ForkTests.t.sol`. For example:

```solidity
contract Alfajores_ChainForkTest is ChainForkTest(ALFAJORES_ID, 1, uints(14)) {}
```

This represents a ChainForkTest for Alfajores, with the expectation that there's a single exchange provider,
and it has 14 exchanges. If the expectations change this will fail and need to be updated.

```solidity
contract Alfajores_P0E00_ExchangeForkTest is ExchangeForkTest(ALFAJORES_ID, 0, 0) {}
```

This represents an ExchangeForkTest for the 0th exchange of the 0th exchange provider on Alfajores.
These tests contracts need to be added manually when we add more pairs or exchange providers, but the
assertions at chain level gives us the heads up when this changes.

### assertions, actions, helpers

Fork tests can get quite complex because we need to understand the current chain state and manipualte it when needed to be able to test our assetions.
That resulted in the past in a mess of unstructure utility functions. This new structrue tries to improve that:

```
    +------------------+    +------------------+
    |  SwapActions     |    | OracleActions    |
    +------------------+    +------------------+
                  ^              ^
                   \            /
                    \          /
                +------------------+
                |     Actions      |
                +------------------+
                  ^              ^
                 /                \
    +------------------+    +------------------+
    |  SwapAssertions  |    |  CircuitBreaker  |
    |                  |    |  Assertions      |
    +------------------+    +------------------+
                  ^              ^
                   \            /
                    \          /
            +----------------------+
            | Celo_P0E00_...       |
            +----------------------+
```

#### `Assertions`

Assertions are contracts that implement high-level assertions about the protocol.
For example `SwapAssertions` contains `assert_swapIn` which asserts a swap is possible, or `assert_swapInFails` which asserts a swap fails with a given revert reason.

These reasuable building blocks are used inside of the actual tests defined in the `ExchangeForkTest` contract.
All assertions extend and make us of `Actions`.

If you're writing a test and you want to express an assertion about the protocol that either gets too complex, or should be reused, it can become an assertion.
Otherwise you can also simply use `Actions` in the tests and assert their outcome.

#### `Actions`

Actions are contract that implement utilities which modify the chain state, e.g. executing a swap, changing an oracle rate feed, swapping repeteadly until a limit, etc.

If it's more than just calling a function on a contract, it can become an action.

#### `Helpers`

Helpers are libraries that just read chain state and expose it in an useful manner.
They're imported on all levels of the struture by doing:

```solidity
contract OracleActions {
  using OracleHelpers for *;
  using SwapHelpers for *;
  using TokenHelpers for *;
  using TradingLimitHelpers for *;
  using LogHelpers for *;
}
```

Most of them attach to `ExchangeForkTest` and are accessed using the `ctx` variable.
For example `ctx.tradingLimitsState(asset)` will load the trading limits state for the asset.
This works because the ctx contains all information about the current `Exchange`.

### `ctx`

To make it easy to get access to the current test context everywhere in the utility contracts, they all implement a private `ctx` var as:

```solidity
ExchangeForkTest private ctx = ExchangeForkTest(address(this));
```

This is because in the end this whole inheritance structure collapses to a single ExchangeForkTest contract and we already know this.
So we can introduce this magic `ctx` variable which gets you access to all assertions and actions (defined as public),
and all of the public variables of `ExchangeForkTest`, meaning all loaded contracts, the current exchange, etc.
