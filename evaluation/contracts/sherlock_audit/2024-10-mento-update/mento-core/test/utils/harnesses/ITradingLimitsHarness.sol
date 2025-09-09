// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity >0.5.13 <0.9;
pragma experimental ABIEncoderV2;

import { ITradingLimits } from "contracts/interfaces/ITradingLimits.sol";

interface ITradingLimitsHarness {
  function validate(ITradingLimits.Config calldata config) external view;

  function verify(ITradingLimits.State calldata state, ITradingLimits.Config calldata config) external view;

  function reset(
    ITradingLimits.State calldata state,
    ITradingLimits.Config calldata config
  ) external view returns (ITradingLimits.State memory);

  function update(
    ITradingLimits.State calldata state,
    ITradingLimits.Config calldata config,
    int256 netflow,
    uint8 decimals
  ) external view returns (ITradingLimits.State memory);
}
