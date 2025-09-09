// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity >0.5.13 <0.9;

import { IBreakerBox } from "./IBreakerBox.sol";

interface ISortedOracles {
  enum MedianRelation {
    Undefined,
    Lesser,
    Greater,
    Equal
  }

  function addOracle(address, address) external;

  function removeOracle(address, address, uint256) external;

  function report(address, uint256, address, address) external;

  function removeExpiredReports(address, uint256) external;

  function isOldestReportExpired(address token) external view returns (bool, address);

  function numRates(address) external view returns (uint256);

  function medianRate(address) external view returns (uint256, uint256);

  function numTimestamps(address) external view returns (uint256);

  function medianTimestamp(address) external view returns (uint256);

  function getOracles(address) external view returns (address[] memory);

  function getRates(address token) external view returns (address[] memory, uint256[] memory, MedianRelation[] memory);

  function getTimestamps(
    address token
  ) external view returns (address[] memory, uint256[] memory, MedianRelation[] memory);

  function initialize(uint256) external;

  function setBreakerBox(IBreakerBox) external;

  function getTokenReportExpirySeconds(address token) external view returns (uint256);

  function oracles(address, uint256) external view returns (address);

  function breakerBox() external view returns (IBreakerBox);
}
