// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8;

import { IStableTokenV2 } from "./IStableTokenV2.sol";

/**
 * @title IStableTokenV2DeprecatedInit
 * @notice Interface for the deprecated initialize function in StableTokenV2
 * @dev In order to improve our DX and get rid of `via-ir` interfaces we
 * are deprecating the old initialize function in favor of the new one.
 * Keeping this interface for backwards compatibility, in fork tests,
 * because in practice we will never be able to call this function again, anyway.
 * More details: https://github.com/mento-protocol/mento-core/pull/502
 */
interface IStableTokenV2DeprecatedInit is IStableTokenV2 {
  function initialize(
    string calldata _name,
    string calldata _symbol,
    uint8, // deprecated: decimals
    address, // deprecated: registryAddress,
    uint256, // deprecated: inflationRate,
    uint256, // deprecated:  inflationFactorUpdatePeriod,
    address[] calldata initialBalanceAddresses,
    uint256[] calldata initialBalanceValues,
    string calldata // deprecated: exchangeIdentifier
  ) external;
}
