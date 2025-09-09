// SPDX-License-Identifier: MIT
pragma solidity >=0.6.2 <0.9.0;

pragma experimental ABIEncoderV2;

// 💬 ABOUT
// Standard Library's default Test

// 🧩 MODULES
import { console } from "forge-std/console.sol";
import { console2 } from "forge-std/console2.sol";
import { StdAssertions } from "forge-std/StdAssertions.sol";
import { StdCheats } from "forge-std/StdCheats.sol";
import { stdError } from "forge-std/StdError.sol";

import { StdChains } from "./StdChains.sol"; //custom contract to override initialize
import { stdJson } from "forge-std/StdJson.sol";
import { stdMath } from "forge-std/StdMath.sol";
import { StdStorage, stdStorage } from "forge-std/StdStorage.sol";
import { StdUtils } from "forge-std/StdUtils.sol";
import { Vm } from "forge-std/Vm.sol";

// 📦 BOILERPLATE
import { TestBase } from "forge-std/Base.sol";
import { DSTest } from "ds-test/test.sol";

// ⭐️ TEST
abstract contract Test is
    DSTest,
    StdAssertions,
    StdChains,
    StdCheats,
    StdUtils,
    TestBase
{
    // Note: IS_TEST() must return true.
    // Note: Must have failure system, https://github.com/dapphub/ds-test/blob/cd98eff28324bfac652e63a239a60632a761790b/src/test.sol#L39-L76.
}
