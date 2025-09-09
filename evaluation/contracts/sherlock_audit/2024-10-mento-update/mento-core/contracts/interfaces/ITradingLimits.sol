// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity >0.5.13 <0.9;

interface ITradingLimits {
  /**
   * @dev The State struct contains the current state of a trading limit config.
   * @param lastUpdated0 The timestamp of the last reset of netflow0.
   * @param lastUpdated1 The timestamp of the last reset of netflow1.
   * @param netflow0 The current netflow of the asset for limit0.
   * @param netflow1 The current netflow of the asset for limit1.
   * @param netflowGlobal The current netflow of the asset for limitGlobal.
   */
  struct State {
    uint32 lastUpdated0;
    uint32 lastUpdated1;
    int48 netflow0;
    int48 netflow1;
    int48 netflowGlobal;
  }

  /**
   * @dev The Config struct contains the configuration of trading limits.
   * @param timestep0 The time window in seconds for limit0.
   * @param timestep1 The time window in seconds for limit1.
   * @param limit0 The limit0 for the asset.
   * @param limit1 The limit1 for the asset.
   * @param limitGlobal The global limit for the asset.
   * @param flags A bitfield of flags to enable/disable the individual limits.
   */
  struct Config {
    uint32 timestep0;
    uint32 timestep1;
    int48 limit0;
    int48 limit1;
    int48 limitGlobal;
    uint8 flags;
  }
}
