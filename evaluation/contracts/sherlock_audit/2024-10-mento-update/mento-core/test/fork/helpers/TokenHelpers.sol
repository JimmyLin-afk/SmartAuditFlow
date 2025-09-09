// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

import { console } from "forge-std/console.sol";
import { IERC20 } from "contracts/interfaces/IERC20.sol";
import { FixidityLib } from "celo/contracts/common/FixidityLib.sol";
import { Vm } from "forge-std/Vm.sol";
import { VM_ADDRESS } from "mento-std/Constants.sol";

library TokenHelpers {
  Vm internal constant vm = Vm(VM_ADDRESS);
  bool internal constant DEBUG = true;

  using FixidityLib for FixidityLib.Fraction;

  function toSubunits(int48 units, address token) internal view returns (int256) {
    if (DEBUG) {
      console.log(
        "\tTokenHelpers.toSubunits: units=%s, token=%s, decimals=%s",
        vm.toString(units),
        IERC20(token).symbol(),
        vm.toString(IERC20(token).decimals())
      );
    }
    int256 tokenBase = int256(10 ** uint256(IERC20(token).decimals()));
    return int256(units) * tokenBase;
  }

  function toSubunits(FixidityLib.Fraction memory subunits, address token) internal view returns (uint256) {
    uint256 tokenScaler = 10 ** uint256(24 - IERC20(token).decimals());
    return subunits.unwrap() / tokenScaler;
  }

  function toSubunits(uint256 units, address token) internal view returns (uint256) {
    uint256 tokenBase = 10 ** uint256(IERC20(token).decimals());
    return units * tokenBase;
  }

  function toUnits(uint256 subunits, address token) internal view returns (uint256) {
    uint256 tokenBase = 10 ** uint256(IERC20(token).decimals());
    return subunits / tokenBase;
  }

  function toUnitsFixed(uint256 subunits, address token) internal view returns (FixidityLib.Fraction memory) {
    uint256 tokenBase = 10 ** uint256(IERC20(token).decimals());
    return FixidityLib.newFixedFraction(subunits, tokenBase);
  }

  function symbol(address token) internal view returns (string memory) {
    return IERC20(token).symbol();
  }
}
