// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.18;
// solhint-disable func-name-mixedcase

import { GoodDollarExpansionController } from "contracts/goodDollar/GoodDollarExpansionController.sol";

contract GoodDollarExpansionControllerHarness is GoodDollarExpansionController {
  constructor(bool disabled) GoodDollarExpansionController(disabled) {}

  function exposed_getReserveRatioScalar(ExchangeExpansionConfig calldata config) external returns (uint256) {
    return _getReserveRatioScalar(config);
  }
}
