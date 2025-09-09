// SPDX-License-Identifier: MIT
pragma solidity 0.8.26;

import { LMSR } from "../utils/LMSR.sol";

contract TestLMSR {
  using LMSR for *;

  function getOdds(
    uint256 yesVotes,
    uint256 noVotes,
    uint256 liquidityParameter,
    bool isTrust
  ) external pure returns (uint256) {
    return LMSR.getOdds(yesVotes, noVotes, liquidityParameter, isTrust);
  }

  function getCost(
    uint256 currentYesVotes,
    uint256 currentNoVotes,
    uint256 outcomeYesVotes,
    uint256 outcomeNoVotes,
    uint256 liquidityParameter
  ) external pure returns (int256) {
    return
      LMSR.getCost(
        currentYesVotes,
        currentNoVotes,
        outcomeYesVotes,
        outcomeNoVotes,
        liquidityParameter
      );
  }
}
