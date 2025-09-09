// SPDX-License-Identifier: MIT

pragma solidity 0.8.18;

import "../LidoVault.sol";

contract EchidnaLidoVault is LidoVault(false) {
  function echidna_start_time_less_than_end_time() public view returns (bool) {
    // if startTime is set startTime should always be < endTime
    return startTime == 0 || startTime < endTime;
  }

  function echidna_eth_deposit_less_than_fixed_side_capacity() public view returns (bool) {
    // if initialized the total fixed eth deposits should always be <= fixed side capacity
    return fixedSideCapacity == 0 || fixedETHDepositTokenTotalSupply <= fixedSideCapacity;
  }

  function echidna_variable_bearer_token_less_than_variable_side_capacity() public view returns (bool) {
    return fixedSideCapacity == 0 || variableBearerTokenTotalSupply <= variableSideCapacity;
  }
  function echidna_protocol_fee_less_than_withdrawn_staking_earnings() public view returns (bool) {
    // since protocol fee is only as a percentage of earnings it should always be less
    return appliedProtocolFee <= withdrawnStakingEarnings;
  }

  function echidna_protocol_fee_less_than_vault_ended_staking_earnings() public view returns (bool) {
    return appliedProtocolFee <= withdrawnStakingEarnings + vaultEndedStakingEarnings;
  }
}
