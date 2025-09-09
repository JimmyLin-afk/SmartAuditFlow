# Smart Contract Upgrading Notes

## UUPS Pattern Upgrading

When using the UUPS (Universal Upgradeable Proxy Standard) pattern for upgrading smart contracts, the stored data in the contract remains intact after an upgrade. This is because the UUPS pattern, like other proxy-based upgrade patterns, separates the contract's state (storage) from its logic (code).

### How UUPSUpgradeable Pattern Works

1. **Proxy Contract**: The proxy contract holds the state (storage) of the smart contract. It does not contain the actual implementation logic but instead delegates all calls to an implementation contract. The proxy's storage layout is fixed and does not change across upgrades.

2. **Implementation Contract (Logic Contract)**: This contract contains the logic that the proxy delegates calls to. When you deploy version 1 of your contract (e.g., `MyContractV1`), the proxy points to this implementation contract.

3. **Upgrading Process**: When you want to upgrade to a new version (e.g., `MyContractV2`), you deploy the new implementation contract and update the proxy to point to this new implementation using the `upgradeTo` function. The proxy's storage does not change; only the implementation contract's logic does.

### What Happens to Stored Data?

When you upgrade from version 1 to version 2:

- **Storage Remains the Same**: The proxy retains the same storage layout. Any data stored in version 1 remains in place and accessible after the upgrade. This is because the proxy contract, which holds the storage, is not redeployed or reset. The storage variables maintain their values and positions.

- **Compatibility**: It’s crucial that the storage layout in the new implementation (version 2) remains compatible with the old one (version 1). If you change the order, types, or remove storage variables in version 2, it can lead to data corruption or unexpected behavior.

### Example

Suppose you have `MyContractV1` with the following storage layout:

**`MyContractV1.sol`**:
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";

contract MyContractV1 is Initializable, UUPSUpgradeable, OwnableUpgradeable {
    uint256 public value;

    function initialize(uint256 _value) public initializer {
        __Ownable_init();
        value = _value;
    }

    function increment() public {
        value += 1;
    }

    function _authorizeUpgrade(address newImplementation) internal override onlyOwner {}
}
```

When you upgrade to `MyContractV2`, the storage in the proxy will still have the value set in `MyContractV1`:

**`MyContractV2.sol`**:
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
import "./MyContractV1.sol";

contract MyContractV2 is Initializable, UUPSUpgradeable, OwnableUpgradeable, MyContractV1 {
    function decrement() public {
        value -= 1;
    }

    function _authorizeUpgrade(address newImplementation) internal override onlyOwner {}
}
```

In `MyContractV2`, you added a `decrement` function. After upgrading, if the `value` was `42` in `MyContractV1`, it will still be `42` in `MyContractV2` unless changed by logic in `MyContractV2`.

### Key Considerations for Upgrades

- **Storage Layout Consistency**: Ensure the storage layout in the new implementation matches the old one. New storage variables should be added at the end of the contract, and existing variables should not be removed or reordered.
- **Data Integrity**: Any logic that modifies the stored data should be carefully tested to ensure it does not inadvertently corrupt the data.
- **Testing**: It is essential to test upgrades thoroughly on a test network or local environment (like Anvil) before deploying them on a live network.

By following these practices, you can safely upgrade your contracts using the UUPS pattern without losing any stored data.
